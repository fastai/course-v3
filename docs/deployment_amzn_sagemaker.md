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

With Amazon SageMaker, you pay only for what you use. Hosting is billed by the second, with no minimum fees and no upfront commitments. As part of the [AWS Free Tier](https://aws.amazon.com/free), you can get started with Amazon SageMaker for free. For the first two months after sign-up, you are offered a total of 125 hours of m4.xlarge for deploying your machine learning models for real-time inferencing and batch transform with Amazon SageMaker. The pricing for model deployment per region can be found [here](https://aws.amazon.com/sagemaker/pricing/).

### Example

Say we have a vision based model which is expected to receive one inference call from clients per minute. We could deploy to two *ml.t2.medium* instances for reliable multi-AZ hosting. Each request submits a image of average size 100 KB and returns a response of 100 bytes. We will use the N. Virginia (*us-east-1*) region.


    Hours per month of hosting = 24 * 31 * 2 = 1488
    Hosting instances = ml.t2.medium
    Cost per hour = $0.065

    Monthly hosting cost = $96.72

There is also a charge for data processing (i.e. the data pulled in and out of your model hosting instances). It is calculated at $0.016 per GB for the N. Virginia region. In this example if we assume a request each minute and each image is 100 KB and each response object is 100 bytes, then the data processing charges would be the following:

    Total data IN = 0.0001 GB * 60 * 24 * 31 = 4.464 GB
    Cost per GB IN =  $0.016
    Cost for Data IN = $0.0714

    Total data OUT = 1e-7 * 60 * 24 * 31 = 0.00044 GB
    Cost per GB OUT =  $0.016
    Cost for Data OUT = $0.000007

    Monthly Data Processing cost = $0.0714

There is also a charge for storing your model on S3. If we assume we are using the S3 Standard storage type then this is $0.023 per GB per month. If we assume a model size of 350 MB then the charges are as follows:

    Total storage anount = 0.35 GB
    Cost per GB = $0.023
    Monthly Cost for S3 storage = $0.00805

**Total monthly cost for hosting this model on SageMaker is $96.72 + $0.0714 + $0.00805 = $96.80**

## Setup your SageMaker notebook instance

Setup your notebook instance where you have trained your fast.ai model on a SageMaker notebook instance. To setup a new SageMaker notebook instance with fast.ai installed follow the steps outlined [here](https://course.fast.ai/start_sagemaker.html).

Ensure you have the Amazon SageMaker Python SDK installed in the kernel named *Python 3*. An example command to run is the following:

`pip install sagemaker`

## Per-project setup

**Train your model on your notebook instance**

Create a Jupyter notebook on your SageMaker notebook instance for your project to train your fast.ai model. 

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
data.export()
learn.save('resnet50')
```

**Zip model artefacts and upload to S3**

Now we have exported our model artefacts we can zip them up and upload to S3.

```python
import tarfile
with tarfile.open(path_img/'models/model.tar.gz', 'w:gz') as f:
    t = tarfile.TarInfo('models')
    t.type = tarfile.DIRTYPE
    f.addfile(t)
    f.add(path_img/'models/resnet50.pth', arcname='resnet50.pth')
    f.add(path_img/'export.pkl', arcname='export.pkl')
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

Now we are ready to deploy our model to the SageMaker model hosting service. We will use the SageMaker Pytthon SDK to do this.

Since fast.ai is built on Pytorch we can use the built-in support SageMaker has for Pytorch. All we need to do is create a script for the model serving logic. For more information on how the PyTorch model serving works check the project page [here](https://github.com/aws/sagemaker-python-sdk/tree/master/src/sagemaker/pytorch#sagemaker-pytorch-model-server).

To serve models in SageMaker, we need a script that implements 4 methods: `model_fn`, `input_fn`, `predict_fn` & `output_fn`. 
* The `model_fn` method needs to load the PyTorch model from the saved weights from disk. 
* The `input_fn` method needs to deserialze the invoke request body into an object we can perform prediction on. 
* The `predict_fn` method takes the deserialized request object and performs inference against the loaded model.
* The `output_fn` method takes the result of prediction and serializes this according to the response content type.

The methods `input_fn` and `input_fn` are optional and if obmitted SageMaker will assume the input and output objects are of type [NPY](https://docs.scipy.org/doc/numpy/neps/npy-format.html) format with Content-Type `application/x-npy`.

An example script to serve a vision resnet model can be found below:

```python
import logging
import requests

import os
import io
import glob
import time

from fastai.vision import *

# setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# set the constants for the content types
JSON_CONTENT_TYPE = 'application/json'
JPEG_CONTENT_TYPE = 'image/jpeg'

# Load the fast.ai model
def model_fn(model_dir):
    logger.info('model_fn')
    path = Path(model_dir)
    print('Creating DataBunch object')
    empty_data = ImageDataBunch.load_empty(path)
    arch_name = os.path.splitext(os.path.split(glob.glob(f'{model_dir}/resnet*.pth')[0])[1])[0]
    print(f'Model architecture is: {arch_name}')
    arch = getattr(models, arch_name)    
    learn = create_cnn(empty_data, arch, pretrained=False).load(path/f'{arch_name}')
    return learn

# Deserialize the Invoke request body into an object we can perform prediction on
def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):
    logger.info('Deserializing the input data.')
    # process an image uploaded to the endpoint
    if content_type == JPEG_CONTENT_TYPE:
        img = open_image(io.BytesIO(request_body))
        return img
    # process a URL submitted to the endpoint
    if content_type == JSON_CONTENT_TYPE:
        img_request = requests.get(request_body['url'], stream=True)
        img = open_image(io.BytesIO(img_request.content))
        return img        
    raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))

# Perform prediction on the deserialized object, with the loaded model
def predict_fn(input_object, model):
    logger.info("Calling model")
    start_time = time.time()
    predict_class,predict_idx,predict_values = model.predict(input_object)
    print("--- Inference time: %s seconds ---" % (time.time() - start_time))
    print(f'Predicted class is {str(predict_class)}')
    print(f'Predict confidence score is {predict_values[predict_idx.item()].item()}')
    response = {}
    response['class'] = str(predict_class)
    response['confidence'] = predict_values[predict_idx.item()].item()
    return response

# Serialize the prediction result into the desired response content type
def output_fn(prediction, accept=JSON_CONTENT_TYPE):        
    logger.info('Serializing the generated output.')
    if accept == JSON_CONTENT_TYPE:
        output = json.dumps(prediction)
        return output, accept
    raise Exception('Requested unsupported ContentType in Accept: {}'.format(accept))    
```

Save the script into a python such as `serve.py`

## Deploy to SageMaker

First we need to create a RealTimePredictor class to accept jpeg images as input and output JSON. The default behaviour is to accept a numpy array.

```python
class ImagePredictor(RealTimePredictor):
    def __init__(self, endpoint_name, sagemaker_session):
        super(ImagePredictor, self).__init__(endpoint_name, sagemaker_session=sagemaker_session, serializer=None, 
                                            deserializer=json_deserializer, content_type='image/jpeg')
```

We need to get the IAM role ARN to give SageMaker permissions to read our model artefact from S3.

```python
role = sagemaker.get_execution_role()
```

In this example we will deploy our model to the instance type `ml.m4.xlarge`. We will pass in the name of our serving script e.g. `serve.py`. We will also pass in the S3 path of our model that we uploaded earlier.

```python
model=PyTorchModel(model_data=model_artefact,
                        name=name_from_base("fastai-pets-model"),
                        role=role,
                        framework_version='1.0.0',
                        entry_point='serve.py',
                        predictor_cls=ImagePredictor)

predictor = model.deploy(initial_instance_count=1,
                         instance_type='ml.m4.xlarge')
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
predictor = model.deploy(initial_instance_count=1,
                         instance_type='local')
```

You can call the `predictor.predict()` the same as earlier but it will call the local endpoint.

---
