---
title: "Deploying on Amazon SageMaker"
sidebar: home_sidebar
---

# Amazon SageMaker Deployment

This is a quick guide to deploy your trained models using the [Amazon SageMaker](https://aws.amazon.com/sagemaker/) model hosting service. 

Deploying a model in SageMaker is a three-step process:
1. Create a model in SageMaker
1. Create an endpoint configuration
1. Create an endpoint

For more information on how models are deployed to Amazon SageMaker checkout the documentation [here](https://docs.aws.amazon.com/sagemaker/latest/dg/ex1-deploy-model.html).

We will be using the [Amazon SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/) which makes this easy and automates a few of the steps.

## Pricing

Sagemaker deployment pricing information can be found [here](https://aws.amazon.com/sagemaker/pricing/). In short: you pay an hourly rate depending on the instance type that you choose. Be careful because this can add up fast - for instance, the smallest P3 instance costs >$2000/month. Also note that the AWS free tier only provides enough hours to run an m4.xlarge instance for 5 days.

## Setup your SageMaker notebook instance

Setup your notebook instance where you have trained your fastai model on a SageMaker notebook instance. To setup a new SageMaker notebook instance with fastai installed follow the steps outlined [here](https://course.fast.ai/start_sagemaker.html).

Ensure you have the Amazon SageMaker Python SDK installed in the kernel named *Python 3*. An example command to run is the following:

`pip install sagemaker`

## Per-project setup

**Train your model on your notebook instance**

Create a Jupyter notebook on your SageMaker notebook instance for your project to train your fastai model. 

An example based on the pets lesson 1 exercise is the following:

```python
from fastai.vision import *
path = untar_data(URLs.PETS)
path_img = path/'images'
fnames = get_image_files(path_img)
pat = re.compile(r'/([^/]+)_\d+.jpg$')
bs=64
data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(),
                                   size=299, bs=bs//2).normalize(imagenet_stats)
learn = create_cnn(data, models.resnet50, metrics=error_rate)
learn.fit_one_cycle(8)
learn.unfreeze()
learn.fit_one_cycle(3, max_lr=slice(1e-6,1e-4))
```

**Export your model**

Now that you have trained your `learn` object you can export the `data` object and save the model weights with the following commands:

```python
learn.export(path_img/'models/resnet50.pkl')
```

**Zip model artefacts and upload to S3**

Now we have exported our model artefacts we can zip them up and upload to S3.

```python
import tarfile
with tarfile.open(path_img/'models/model.tar.gz', 'w:gz') as f:
    t = tarfile.TarInfo('models')
    t.type = tarfile.DIRTYPE
    f.addfile(t)
    f.add(path_img/'models/resnet50.pkl', arcname='resnet50.pkl')
```

Now we can upload them to S3 with the following commands. 

```python
import sagemaker
from sagemaker.utils import name_from_base
sagemaker_session = sagemaker.Session()
bucket = sagemaker_session.default_bucket()
prefix = f'sagemaker/{name_from_base("fastai-pets-model")}'
model_artefact = sagemaker_session.upload_data(path=str(path_img/'models/model.tar.gz'), bucket=bucket, key_prefix=prefix)
```

## Create model serving script

Now we are ready to deploy our model to the SageMaker model hosting service. We will use the [SageMaker Python SDK](https://github.com/aws/sagemaker-python-sdk) with the Amazon SageMaker [open-source PyTorch container](https://github.com/aws/sagemaker-pytorch-container) as this container has support for the fast.ai library. Using one of the pre-defined Amazon SageMaker containers makes it easy to write a script and then run it in Amazon SageMaker in just a few steps.

To serve models in SageMaker, we need a script that implements 4 methods: `model_fn`, `input_fn`, `predict_fn` & `output_fn`. 
* The `model_fn` method needs to load the PyTorch model from the saved weights from disk. 
* The `input_fn` method needs to deserialze the invoke request body into an object we can perform prediction on. 
* The `predict_fn` method takes the deserialized request object and performs inference against the loaded model.
* The `output_fn` method takes the result of prediction and serializes this according to the response content type.

The methods `input_fn` and `input_fn` are optional and if obmitted SageMaker will assume the input and output objects are of type [NPY](https://docs.scipy.org/doc/numpy/neps/npy-format.html) format with Content-Type `application/x-npy`.

For more information on how the PyTorch model serving works check the project page [here](https://github.com/aws/sagemaker-python-sdk/tree/master/src/sagemaker/pytorch#sagemaker-pytorch-model-server).

An example script to serve a vision resnet model can be found below:

```python
import logging, requests, os, io, glob, time
from fastai.vision import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

JSON_CONTENT_TYPE = 'application/json'
JPEG_CONTENT_TYPE = 'image/jpeg'

# loads the model into memory from disk and returns it
def model_fn(model_dir):
    logger.info('model_fn')
    path = Path(model_dir)
    learn = load_learner(model_dir, fname='resnet50.pkl')
    return learn

# Deserialize the Invoke request body into an object we can perform prediction on
def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):
    logger.info('Deserializing the input data.')
    # process an image uploaded to the endpoint
    if content_type == JPEG_CONTENT_TYPE: return open_image(io.BytesIO(request_body))
    # process a URL submitted to the endpoint
    if content_type == JSON_CONTENT_TYPE:
        img_request = requests.get(request_body['url'], stream=True)
        return open_image(io.BytesIO(img_request.content))
    raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))

# Perform prediction on the deserialized object, with the loaded model
def predict_fn(input_object, model):
    logger.info("Calling model")
    start_time = time.time()
    predict_class,predict_idx,predict_values = model.predict(input_object)
    print("--- Inference time: %s seconds ---" % (time.time() - start_time))
    print(f'Predicted class is {str(predict_class)}')
    print(f'Predict confidence score is {predict_values[predict_idx.item()].item()}')
    return dict(class = str(predict_class),
        confidence = predict_values[predict_idx.item()].item())

# Serialize the prediction result into the desired response content type
def output_fn(prediction, accept=JSON_CONTENT_TYPE):        
    logger.info('Serializing the generated output.')
    if accept == JSON_CONTENT_TYPE: return json.dumps(prediction), accept
    raise Exception('Requested unsupported ContentType in Accept: {}'.format(accept))    
```

Save the script into a python such as `serve.py`

## Deploy to SageMaker

First we need to create a RealTimePredictor class to accept jpeg images as input and output JSON. The default behaviour is to accept a numpy array.

```python
class ImagePredictor(RealTimePredictor):
    def __init__(self, endpoint_name, sagemaker_session):
        super().__init__(endpoint_name, sagemaker_session=sagemaker_session, serializer=None, 
                         deserializer=json_deserializer, content_type='image/jpeg')
```

We need to get the IAM role ARN to give SageMaker permissions to read our model artefact from S3.

```python
role = sagemaker.get_execution_role()
```

In this example we will deploy our model to the instance type `ml.m4.xlarge`. We will pass in the name of our serving script e.g. `serve.py`. We will also pass in the S3 path of our model that we uploaded earlier.

```python
model=PyTorchModel(model_data=model_artefact, name=name_from_base("fastai-pets-model"),
    role=role, framework_version='1.0.0', entry_point='serve.py', predictor_cls=ImagePredictor)

predictor = model.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')
```

It will take a while for SageMaker to provision the endpoint ready for inference. 


**Test the endpoint**

Now you can make inference calls against the deployed endpoint with a call such as:

```python
url = <some url of an image to test>
img_bytes = requests.get(url).content
predictor.predict(img_bytes); response
```

## Local testing
In case you want to test the endpoint before deploying to SageMaker you can run the following `deploy` command changing the parameter name `instance_type` value to `local`.

```python
predictor = model.deploy(initial_instance_count=1, instance_type='local')
```

You can call the `predictor.predict()` the same as earlier but it will call the local endpoint.

