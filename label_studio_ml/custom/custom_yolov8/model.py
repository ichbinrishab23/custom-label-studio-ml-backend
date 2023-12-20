import os
from typing import Dict, List, Optional
from uuid import uuid4

from label_studio_converter import brush
from numpy import zeros
from torch import uint8
from ultralytics import SAM, YOLO
from ultralytics.models.sam.amg import batched_mask_to_box

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import (
    get_image_local_path,
    get_image_size,
    get_single_tag_keys,
)

LABEL_STUDIO_HOST = os.environ.get("LABEL_STUDIO_HOST")
LABEL_STUDIO_ACCESS_TOKEN = os.environ.get("LABEL_STUDIO_ACCESS_TOKEN")

DEVICE = "cuda"


class YoloModel(LabelStudioMLBase):
    def __init__(self, project_id, **kwargs):
        super().__init__(**kwargs)
        self.yolo_detector = YOLO("/models/best_v3.pt")
        self.sam_detector = SAM("/models/sam_b.pt")

        self.yolo_detector.to(DEVICE)
        self.sam_detector.to(DEVICE)

    def predict(
        self,
        tasks: List[Dict],
        context: Optional[Dict] = None,
        **kwargs,
    ) -> List[Dict]:
        print("--- RUN PREDICTION ---")
        # print(f"Tasks ({len(tasks)}):\n{tasks}")
        # print("---")
        # print("Context:\n", context)
        # print("---------------")

        if context is None:
            return self.perform_full_prediction(tasks)

        return self.perform_custom_prediction(tasks, context)

    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated
        You can run your logic here to update the model and persist it to the cache
        It is not recommended to perform long-running operations here, as it will block the main thread
        Instead, consider running a separate process or a thread (like RQ worker) to perform the training
        :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED')
        :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
        """

        # use cache to retrieve the data from the previous fit() runs
        old_data = self.get("my_data")
        old_model_version = self.get("model_version")
        print(f"Old data: {old_data}")
        print(f"Old model version: {old_model_version}")

        # store new data to the cache
        self.set("my_data", "my_new_data_value")
        self.set("model_version", "my_new_model_version")
        print(f'New data: {self.get("my_data")}')
        print(f'New model version: {self.get("model_version")}')

        print("fit() completed successfully.")

    def perform_full_prediction(self, tasks: List[Dict]):
        print("Automatic labeling -> Yolo + SAM")
        ls_results = []

        # Extract info from parsed-label-config
        print(f"parsed_label_config:\n{self.parsed_label_config}")
        if not self.parsed_label_config or len(self.parsed_label_config) < 1:
            print("Parsed label config not found.")
            return []

        # from_name, to_name, value, labels = get_single_tag_keys(
        #     {"tag0": self.parsed_label_config["tag0"]}, "RectangleLabels", "Image"
        # )

        for task in tasks:
            path_img = task["data"]["image"]

            path_img = get_image_local_path(
                path_img,
                label_studio_access_token=LABEL_STUDIO_ACCESS_TOKEN,
                label_studio_host=LABEL_STUDIO_HOST,
            )
            print(f"Predict: {path_img}")

            yolo_results = self.yolo_detector.predict(
                path_img,
                stream=True,
                device=DEVICE,
                conf=0.2,
                verbose=False,
                classes=[0],
            )

            label = "Apple"  # TODO: Implement mapping from yolo -> label studio
            prob = 0.2

            for yolo_result in yolo_results:
                height, width, _ = yolo_result.orig_img.shape

                sam_result = self.sam_detector(
                    yolo_result.orig_img,
                    bboxes=yolo_result.boxes.xyxy,
                    verbose=False,
                    device=DEVICE,
                )[0]

                bboxes_sam = batched_mask_to_box(sam_result.masks.data).cpu().numpy()
                masks_sam = sam_result.masks.data.cpu().numpy()

                masks_sam = masks_sam.astype(int) * 255

                for bbox, mask in zip(bboxes_sam, masks_sam):
                    rle = brush.mask2rle(mask)

                    ls_results += [
                        {
                            "id": str(uuid4())[:4],
                            "from_name": "tag0",
                            "to_name": "image",
                            "original_width": width,
                            "original_height": height,
                            "image_rotation": 0,
                            "score": prob,
                            "type": "rectanglelabels",
                            "readonly": False,
                            "value": {
                                "rotation": 0,
                                "x": 100 * bbox[0] / width,  # Top-left
                                "y": 100 * bbox[1] / height,  # Top-left
                                "width": 100 * (bbox[2] - bbox[0]) / width,
                                "height": 100 * (bbox[3] - bbox[1]) / height,
                                "rectanglelabels": [label],
                            },
                        },
                        {
                            "id": str(uuid4())[:4],
                            "from_name": "tag1",
                            "to_name": "image",
                            "original_width": width,
                            "original_height": height,
                            "image_rotation": 0,
                            "score": 0.5,
                            "type": "brushlabels",
                            "readonly": False,
                            "value": {
                                "format": "rle",
                                "rle": rle,
                                "brushlabels": [label],
                            },
                        },
                    ]

        return [{"result": ls_results, "model_version": "yolov8_sam_v1"}]

    def perform_custom_prediction(self, tasks: List[Dict], context: Dict):
        print("Manual labeling -> SAM support")
        print(f"Tasks ({len(tasks)}):\n{tasks}")
        print("---")
        print("Context:\n", context)
        print("---------------")

        ls_results = []

        from_name, to_name, value = self.get_first_tag_occurence(
            "RectangleLabels", "Image"
        )

        image_width = context["result"][0]["original_width"]
        image_height = context["result"][0]["original_height"]

        # collect context information
        point_coords = []
        point_labels = []
        input_box = None
        selected_label = None
        for ctx in context["result"]:
            x = ctx["value"]["x"] * image_width / 100
            y = ctx["value"]["y"] * image_height / 100
            ctx_type = ctx["type"]
            selected_label = ctx["value"][ctx_type][0]
            if ctx_type == "keypointlabels":
                point_labels.append(int(ctx["is_positive"]))
                point_coords.append([int(x), int(y)])
            elif ctx_type == "rectanglelabels":
                box_width = ctx["value"]["width"] * image_width / 100
                box_height = ctx["value"]["height"] * image_height / 100
                input_box = [int(x), int(y), int(box_width + x), int(box_height + y)]

        if not input_box:
            return []

        path_img = tasks[0]["data"][value]
        path_img = get_image_local_path(
            path_img,
            label_studio_access_token=LABEL_STUDIO_ACCESS_TOKEN,
            label_studio_host=LABEL_STUDIO_HOST,
        )
        print(f"Predict: {path_img}")

        sam_result = self.sam_detector(
            path_img,
            bboxes=input_box,
            verbose=False,
            device=DEVICE,
        )[0]

        height,width,_ = sam_result.orig_img.shape
        bboxes_sam = batched_mask_to_box(sam_result.masks.data).cpu().numpy()
        masks_sam = sam_result.masks.data.cpu().numpy()

        masks_sam = masks_sam.astype(int) * 255

        for bbox, mask in zip(bboxes_sam, masks_sam):
            rle = brush.mask2rle(mask)

            ls_results += [
                {
                    "id": str(uuid4())[:4],
                    "from_name": "tag0",
                    "to_name": "image",
                    "original_width": width,
                    "original_height": height,
                    "image_rotation": 0,
                    "score": 1.0,
                    "type": "rectanglelabels",
                    "readonly": False,
                    "value": {
                        "rotation": 0,
                        "x": 100 * bbox[0] / width,  # Top-left
                        "y": 100 * bbox[1] / height,  # Top-left
                        "width": 100 * (bbox[2] - bbox[0]) / width,
                        "height": 100 * (bbox[3] - bbox[1]) / height,
                        "rectanglelabels": [selected_label],
                    },
                },
                {
                    "id": str(uuid4())[:4],
                    "from_name": "tag1",
                    "to_name": "image",
                    "original_width": width,
                    "original_height": height,
                    "image_rotation": 0,
                    "score": 0.5,
                    "type": "brushlabels",
                    "readonly": False,
                    "value": {
                        "format": "rle",
                        "rle": rle,
                        "brushlabels": [selected_label],
                    },
                },
            ]

        return [{"result": ls_results, "model_version": "yolov8_sam_v1"}]
