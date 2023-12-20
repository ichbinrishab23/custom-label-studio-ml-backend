# Quickstart

1. Addition of NNs:
   - Add your pre-trained model as well as SAM to the models folder
   - Go to model.py and set the paths to yolo and sam accordingly 

2. Changes to docker-compose.yml:
   - Update your local IPv4 Address in the variable LABEL_STUDIO_HOST
   - Update your label studio access token in the variable LABEL_STUDIO_ACCESS_TOKEN

3. Build and start Machine Learning backend on 'http://localhost:9090'

```bash
docker compose build
```

```bash
docker compose up
```

4. Validate that backend is running

```bash
$ curl http://localhost:9090/health
{"status":"UP"}
```

5. Open another terminal and start label studio using the following command

```bash
label-studio start
```

6. Go to the label studio frontend on 'http://localhost:8080' and connect to the backend from Label Studio:
    - go to your project `Settings -> Machine Learning -> Add Model`
    - specify 'http://localhost:9090' as a URL

8. Have fun labeling
