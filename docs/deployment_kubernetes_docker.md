---
title: "Deploying to Kubernetes with BentoML"
sidebar: home_sidebar
---


# Kubernetes Deployment

In this guide, you will deploy the pet classification model from lesson one as a REST
API server to a Kubernetes cluster using BentoML.

## Setup

1. A Kubernetes enabled cluster or machine.
    * This guide uses Kubernetes' recommend learning environment, `minikube`.
    `minikube` installation: https://kubernetes.io/docs/setup/learning-environment/minikube/
    * learn more about kubernetes installation: https://kubernetes.io/docs/setup/
       * Managed kubernetes cluster by Cloud providers
         * AWS: https://aws.amazon.com/eks/
         * Google: https://cloud.google.com/kubernetes-engine/
         * Azure: https://docs.microsoft.com/en-us/azure/aks/intro-kubernetes
    * `kubectl` CLI tool: https://kubernetes.io/docs/tasks/tools/install-kubectl/
2. Docker and Docker Hub is properly installed and configured on your local system
    * Docker installation instruction: https://www.docker.com/get-started
    * Docker Hub: https://hub.docker.com
3. Python (3.6 or above) and required packages: `bentoml`, `fastai`, `torch`, and `torchvision`
    * ```pip install bentoml fastai==1.0.57 torch==1.4.0 torchvision=0.5.0```

## Build model API server with BentoML

Run the following code from the Fastai lesson one notebook:

```python
from fastai.vision import *

path = untar_data(URLs.PETS)
path_img = path/'images'
fnames = get_image_files(path_img)
bs=64
np.random.seed(2)
pat = r'/([^/]+)_\d+.jpg$'
data = ImageDataBunch.from_name_re(
    path_img,
    fnames,
    pat,
    num_workers=0,
    ds_tfms=get_transforms(),
    size=224,
    bs=bs
).normalize(imagenet_stats)
learn = create_cnn(data, models.resnet50, metrics=error_rate)
learn.fit_one_cycle(8)
learn.unfreeze()
learn.fit_one_cycle(3, max_lr=slice(1e-6,1e-4))
```

Use BentoML to create an API model server for the pet classification model:

```python
# pet_classification.py file

from bentoml import BentoService, api, env, artifacts
from bentoml.artifact import FastaiModelArtifact
from bentoml.handlers import FastaiImageHandler

@artifacts([FastaiModelArtifact('pet_classifier')])
@env((auto_pip_dependencies=True)
class PetClassification(BentoService):

    @api(FastaiImageHandler)
    def predict(self, image):
        result = self.artifacts.pet_classifier.predict(image)
        return str(result)
```

This code defines a model server using `Fastai` model, asks BentoML to figure out
the required PyPi packages automatically. It also defines an API called `predict`, that
is the entry point to access this model server. The API expects a `Fastai`
`ImageData` object as its input data.

Run the following code to create a BentoService SavedBundle with the pet classification
model. A BentoService SavedBundle is a versioned file archive ready for production
deployment. The archive contains the model service defined above, python code
dependencies and PyPi dependencies, and the trained pet classification model:

```python
# Create a PetClassification instance
from pet_classification import PetClassification
service = PetClassification()

#  Pack the newly trained model artifact
service.pack('pet_classifier', learn)

# Save the prediction service to disk for model serving
service.save()
```
Use BentoML CLI to start a local API model server:

```bash
bentoml serve PetClassification:latest
```

BentoML automatically process the incoming data into required data format defined in the
API. For the pet classifier BentoService defined above, incoming data will transform to
fastai `ImageData` object.

Use `curl` request in another terminal to get the prediction result:

```bash
# Replace PATH_TO_TEST_IMAGE_FILE with one of the image from {path_img}
# An example path: /Users/user_name/.fastai/data/oxford-iiit-pet/images/shiba_inu_122.jpg
curl -i \
    --request POST \
    --header "Content-Type: multipart/form-data" \
    -F "image=@PATH_TO_TEST_IMAGE_FILE" \
    localhost:5000/predict
```

## Deploy model server to Kubernetes

### Build model server image

Find the file directory of the SavedBundle with `bentoml get` command, which is
directory structured as a docker build context. Running docker build with this
directory produces a docker image containing the model API server. Replace
docker_username with your Docker Hub username and run the following code:

```bash
saved_path=$(bentoml get PetClassifier:latest -q | jq -r ".uri.uri")

docker build -t {docker_username}/pet-classifier .

docker push {docker_username}/pet-classifier
```

Use `docker run` command to test the docker image locally:

```bash
docker run -p 5000:5000 {docker_username}/pet-classifier
```

In another terminal window, use the `curl` command from above to get the prediction result.

### Deploy to Kubernetes

The following is an example YAML file for specifying the resources required to run and
expose a BentoML model server in a Kubernetes cluster. Replace `{docker_username}` with
your Docker Hub username and save it to `pet-classifier.yaml` file:

```yaml
apiVersion: v1
kind: Service
metadata:
  labels:
    app: pet-classifier
  name: pet-classifier
spec:
  ports:
  - name: predict
    port: 5000
    targetPort: 5000
  selector:
    app: pet-classifier
  type: LoadBalancer
---
apiVersion: apps/v1
kind: Deployment
metadata:
  labels:
    app: pet-classifier
  name: pet-classifier
spec:
  selector:
    matchLabels:
      app: pet-classifier
  template:
    metadata:
      labels:
        app: pet-classifier
    spec:
      containers:
      - image: {docker_username}/pet-classifier
        name: pet-classifier
        ports:
        - containerPort: 5000
```

Use `kubectl apply` command to deploy the model server to kubernetes cluster.

```bash
kubectl apply -f pet-classifier.yaml
```

Check deployment status with `kubectl`:

```bash
kubectl get svc pet-classifier
```

### Send prediction request

Make prediction request with `curl`:

```bash
# If you are not using minikube, replacing ${minikube ip} with your Kubernetes cluster's IP

# Replace PATH_TO_TEST_IMAGE_FILE
curl -i \
    --request POST \
    --header "Content-Type: multipart/form-data" \
    -F "image=@PATH_TO_TEST_IMAGE_FILE" \
    ${minikube ip}:5000/predict
```

### Delete deployment from Kubernetes cluster

```bash
kubectl delete -f pet-classifier.yaml
```


## Monitor model server metrics with Prometheus

### Setup
  * A cluster with Prometheus installed.
    * For Kubernetes installation: https://github.com/coreos/kube-prometheus
    * For Prometheus installation in other environments: https://prometheus.io/docs/introduction/first_steps/#starting-prometheus

BentoML model server has built-in Prometheus metrics endpoint. Users can also customize
metrics fit their needs when building a model server with BentoML.

For monitoring metrics with Prometheus enabled Kubernetes cluster, update the annotations
in deployment spec with `prometheus.io/scrape: true` and `prometheus.io/port: 5000`.

For example:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  labels:
    app: pet-classifier
  name: pet-classifier
spec:
  selector:
    matchLabels:
      app: pet-classifier
  template:
    metadata:
      labels:
        app: pet-classifier
      annotations:
        prometheus.io/scrape: true
        prometheus.io/port: 5000
    spec:
      containers:
      - image: {docker_username}/pet-classifier
        name: pet-classifier
        ports:
        - containerPort: 5000
```

For monitoring metrics in other environments, update Prometheus scraping config.

An example of a scraping job inside Prometheus configuration:

```
job_name: pet-classifier
host: MODEL_SERVER_IP:5000
```

## Additional information

* BentoML documentation: https://docs.bentoml.org/en/latest
* Deployment tutorials to other platforms or services: https://docs.bentoml.org/en/latest/deployment/index.html