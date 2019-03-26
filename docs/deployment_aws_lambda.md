---
title: "Deploying on AWS Lambda"
sidebar: home_sidebar
---

# AWS Lambda Deployment

This is a quick guide to deploy your fastai model into production using [Amazon API Gateway](https://aws.amazon.com/api-gateway/) & [AWS Lambda](https://aws.amazon.com/lambda/). This guide will use the [Serverless Application Model (SAM)](https://aws.amazon.com/serverless/sam/) as the framework for building the application that will interfact with the Lambda and API Gateway AWS services.

**[AWS Lambda](https://aws.amazon.com/lambda/)** lets you run code without provisioning or managing servers. You pay only for the compute time you consume - there is no charge when your code is not running.

**[Amazon API Gateway](https://aws.amazon.com/api-gateway/)** is a fully managed service that makes it easy for developers to create, publish, maintain, monitor, and secure APIs at any scale. 

## Pricing

With **AWS Lambda**, you pay only for what you use. You are charged based on the number of requests for your functions and the duration, the time it takes for your code to execute. For more information on AWS Lambda pricing checkout the page [here](https://aws.amazon.com/lambda/pricing/).

With **Amazon API Gateway**, you only pay when your APIs are in use. For HTTP/REST APIs, you pay only for the API calls you receive and the amount of data transferred out. More details on the pricing are available [here](https://aws.amazon.com/api-gateway/pricing/).

## Requirements

You will need to have the following applications installed on your local machine.

* [AWS CLI](https://aws.amazon.com/cli/) already configured with Administrator permission
* [Python 3 installed](https://www.python.org/downloads/)
* [Docker installed](https://www.docker.com/community-edition)
* [AWS SAM CLI](https://aws.amazon.com/serverless/sam/) already installed. See guide for instructions.

## Preparation steps

### Create S3 Bucket

First, we need a `S3 bucket` where we can upload our model artefacts as well as our Lambda functions/layers packaged as ZIP files before we deploy anything - If you don't have a S3 bucket to store model and code artifacts then this is a good time to create one:

```bash
aws s3 mb s3://REPLACE_WITH_YOUR_BUCKET_NAME
```

### Export your trained model and upload to S3

You wil need to start with a pretrained model, most likely on a Jupyter notebook server. The SAM application expects a PyTorch model in [TorchScript](https://pytorch.org/docs/stable/jit.html?highlight=jit#module-torch.jit) format to be saved to S3 along with a classes text file with the output class names. 

An example Python code snippet of how you can export a fastai vision model is shown below.

```python
# export model to TorchScript format
trace_input = torch.ones(1,3,299,299).cuda()
jit_model = torch.jit.trace(learn.model.float(), trace_input)
model_file='resnet50_jit.pth'
output_path = str(path_img/f'models/{model_file}')
torch.jit.save(jit_model, output_path)
# export classes text file
save_texts(path_img/'models/classes.txt', data.classes)
tar_file=path_img/'models/model.tar.gz'
classes_file='classes.txt'
# create a tarfile with the exported model and classes text file
with tarfile.open(tar_file, 'w:gz') as f:
    f.add(path_img/f'models/{model_file}', arcname=model_file)
    f.add(path_img/f'models/{classes_file}', arcname=classes_file)
```

Now we are ready to upload the model artefacts to S3.

```python
import boto3
s3 = boto3.resource('s3')
# replace 'mybucket' with the name of your S3 bucket
s3.meta.client.upload_file(tar_file, 'REPLACE_WITH_YOUR_BUCKET_NAME', 'fastai-models/lesson1/model.tar.gz')
```

A full example of training a model, exporting it to the JIT format and uploading to S3 based on **Lesson 1** can be found [here](https://github.com/fastai/course-v3/blob/master/docs/production/lesson-1-export-jit.ipynb).

### Grab the SAM example project

We will be using SAM to deploy our application. First we need to download the example project using the commands below.

```bash
wget https://github.com/fastai/course-v3/raw/master/docs/production/aws-lambda.zip

unzip aws-lambda.zip
```

## Application overview

The example application makes inference calls on a computer vision model. When the lambda function is loaded it will download the PyTorch model from S3 and load the model into memory. 

It takes a JSON object as input containing the URL of an image somewhere on the internet. The application downloads the image, converts the pixels into a PyTorch Tensor object and passes it through the PyTorch model. It then returns the class name with the highest output score from the model and also returns a confidence level.

The structure of this application is the following:

```bash
.
├── event.json          <-- Event payload for local testing
├── pytorch             <-- Folder for source code
│   ├── __init__.py
│   ├── app.py          <-- Lambda function code
└──template.yaml        <-- SAM Template
```

For information on the SAM Templates view the documentation [here](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-sam-template-basics.html).

For information on programming Lambda functions using the Python programming language see the documentation [here](https://docs.aws.amazon.com/lambda/latest/dg/python-programming-model-handler-types.html).

**Request Body format**

The Lambda function expects as input a JSON string body containing the URL of an image to classify.

Example:
```json
{
    "url": "REPLACE_THIS_WITH_AN_IMAGE_URL"
}
```
**Response format**

The Lambda function will return a JSON object containing the a status code (e.g. 200 for success) and in the body a predicted class and confidence score.

Example:
```json
{
    "statusCode": 200,
    "body": {
        "class": "english_cocker_spaniel",
        "confidence": 0.99
    }
}
```

You may modify this file based on your application to take different input/output formats.

**Lambda Layer**

You can configure your Lambda function to pull in additional code and content in the form of [Lambda layers](https://docs.aws.amazon.com/lambda/latest/dg/configuration-layers.html). A layer is a ZIP archive that contains libraries, a custom runtime, or other dependencies. With layers, you can use libraries in your function without needing to include them in your deployment package.

In this project we will be using a publicly accessible Lambda layer that contains the necessary PyTorch libraries needed to run our application. These layers are deployed to the following regions: *us-west-2,us-east-1,us-east-2,eu-west-1,ap-southeast-1,ap-southeast-2,ap-northeast-1,eu-central-1*. The default region is *us-east-1*.

If you are not running your model in the default region (i.e. *us-east-1*) You will need to update the file `template.yaml` with the correct region code by replacing the text `AWS_REGION` with the correct region (e.g. us-west-2).

```yaml
...
  LambdaLayerArn:
    Type: String
    Default: "arn:aws:lambda:AWS_REGION:934676248949:layer:pytorchv1-py36:1"
        ...
```

## Local development

**Creating test Lambda Environment Variables**

First create a file called `env.json` with the payload similar to the following substituting the values for the S3 Bucket and Key where your PyTorch model has been uploaded to S3 in the earlier step.

```json
{
    "PyTorchFunction": {
      "MODEL_BUCKET": "REPLACE_WITH_YOUR_BUCKET_NAME",  
      "MODEL_KEY": "fastai-models/lesson1/model.tar.gz"      
    }
}
```

**Invoking function locally using a local sample payload**

Edit the file named `event.json` and enter a value for the JSON value `url` to the image you want to classify.

Call the following sam command to test the function locally.

```bash
sam local invoke PyTorchFunction -n env.json -e event.json
```

If you are modifying your expected request body for your application then also modify the file `event.json` to match the correct input format.

**Invoking function locally through local API Gateway**

```bash
sam local start-api -n env.json
```

If the previous command ran successfully you should now be able to send a post request to the local endpoint.

An example for the computer vision applicaiton is the following:

```bash
curl -d "{\"url\":\"REPLACE_THIS_WITH_AN_IMAGE_URL\"}" \
    -H "Content-Type: application/json" \
    -X POST http://localhost:3000/invocations
```


**SAM CLI** is used to emulate both Lambda and API Gateway locally and uses our `template.yaml` to understand how to bootstrap this environment (runtime, where the source code is, etc.) - The following excerpt is what the CLI will read in order to initialize an API and its routes:

```yaml
...
Events:
    PyTorch:
        Type: Api # More info about API Event Source: https://github.com/awslabs/serverless-application-model/blob/master/versions/2016-10-31.md#api
        Properties:
            Path: /invocations
            Method: post
```

## Packaging and deployment

AWS Lambda Python runtime requires a flat folder with all dependencies including the application. SAM will use `CodeUri` property to know where to look up for both application and dependencies:

```yaml
...
    PyTorchFunction:
        Type: AWS::Serverless::Function
        Properties:
            CodeUri: pytorch/
            ...
```

Next, run the following command to package our Lambda function to S3:

```bash
sam package \
    --output-template-file packaged.yaml \
    --s3-bucket REPLACE_THIS_WITH_YOUR_S3_BUCKET_NAME
```

Next, the following command will create a Cloudformation Stack and deploy your SAM resources. You will need to override the default parameters for the bucket name and object key. This is done by passing the `--parameter-overrides` option to the `deploy` command as shown below.

```bash
sam deploy \
    --template-file packaged.yaml \
    --stack-name pytorch-sam-app \
    --capabilities CAPABILITY_IAM \
    --parameter-overrides BucketName=REPLACE_WITH_YOUR_BUCKET_NAME ObjectKey=fastai-models/lesson1/model.tar.gz
```

> **See [Serverless Application Model (SAM) HOWTO Guide](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-quick-start.html) for more details in how to get started.**

After deployment is complete you can run the following command to retrieve the API Gateway Endpoint URL:

```bash
aws cloudformation describe-stacks \
    --stack-name pytorch-sam-app \
    --query 'Stacks[].Outputs[?OutputKey==`PyTorchApi`]' \
    --output table
``` 

## Fetch, tail, and filter Lambda function logs

To simplify troubleshooting, SAM CLI has a command called sam logs. sam logs lets you fetch logs generated by your Lambda function from the command line. In addition to printing the logs on the terminal, this command has several nifty features to help you quickly find the bug.

`NOTE`: This command works for all AWS Lambda functions; not just the ones you deploy using SAM.

```bash
sam logs -n PyTorchFunction --stack-name pytorch-sam-app --tail
```

You can find more information and examples about filtering Lambda function logs in the [SAM CLI Documentation](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-sam-cli-logging.html).


## Cleanup

In order to delete our Serverless Application recently deployed you can use the following AWS CLI Command:

```bash
aws cloudformation delete-stack --stack-name pytorch-sam-app
```
