---
title: "Deploying on Azure Functions"
sidebar: home_sidebar
---

# Microsoft Azure Functions Deployment

This is a quick guide to deploy your trained models using [Microsoft Azure Functions](https://docs.microsoft.com/en-us/azure/azure-functions/functions-overview).

This guide will upload a trained image classifier to Azure Functions.  The inference function will be triggered by a HTTP POST request method call that includes the URL of an image that is to be classified.  The result of the prediction will be returned in the HTTP Response.

## Microsoft Azure Functions

Microsoft Azure Functions is the serverless architecture that Microsoft offers.  You don't need to provision the servers or maintain the OS, but just need to upload your code and define any packages it depends on.

### Pricing

Microsoft Azure offers two kinds of pricing, [Consumption plan](https://docs.microsoft.com/en-us/azure/azure-functions/functions-scale#consumption-plan) and [App Service plan](https://docs.microsoft.com/en-us/azure/app-service/overview-hosting-plans).  The main difference is that the Consumption plan allows you to pay only when your function runs.  It will scale the architecture for you if needed but you don't have any control on how it scales.  See [here](https://azure.microsoft.com/en-us/pricing/details/functions/) for the Consumption plan pricing.

With the App Service plan, you can pick the level of computing resources that you want your function to run on.  You are then charged for as long as your resources are defined, regardless of whether your function is running or not.  See [here](https://azure.microsoft.com/en-us/pricing/details/app-service/windows/) for the App Service plan pricing.

Currently, python is still in preview stage in Azure Functions and fastai only works when you provide your own custom Docker image on the App Service plan.

## Requirements

### Software

- Linux (Windows WSL isn't sufficient as fastai won't compile properly.  This guide has been tested with [Ubuntu 18.04](http://releases.ubuntu.com/18.04/))
- [Python 3.6](https://www.python.org/downloads/) (only Python runtime currently supported by Azure Functions)
- [Azure Functions Core Tools version 2.x](https://docs.microsoft.com/en-us/azure/azure-functions/functions-run-local#v2)
- [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli)
- [Docker](https://www.docker.com/)

### Accounts

- [Microsoft Azure Account](https://azure.microsoft.com/en-us/)    
- [Docker Hub Account](https://hub.docker.com/)

## 1 - Local Setup

### Setup Project Directory

Replace `<PROJECT_DIR>` with your own project directory name.

```bash
mkdir <PROJECT_DIR>
cd <PROJECT_DIR>
python -m venv .env
source .env/bin/activate
```

### Create Functions Project

Create an Azure Function Project that uses the Python runtime.

```bash
func init --docker
```

When prompted:  

1. `Select a worker runtime:` **python**

### Create Function

Create a function with name <FUNCTION_NAME> using the template "HttpTrigger".  Replace `<FUNCTION_NAME>` with your own function name.

```bash
func new --name <FUNCTION_NAME> --template "HttpTrigger"
```

### Install fastai & Dependencies

Install fastai and any other dependencies your app needs in the virtual environment.  

Then output all the dependencies to requirements.txt which will be used when you build the Docker image.

```bash
pip install fastai  # install other dependencies here
pip freeze > requirements.txt
```

### Update Function

Modify the following files in the <PROJECT_DIR> directory:

#### \<FUNCTION_NAME>/__init__.py

This is where your inference function lives.  The following is an example of using a trained image classification model, of which you use to replace the default file.

```python
import logging
import os

import azure.functions as func
from fastai.vision import *
import requests


def main(req: func.HttpRequest) -> func.HttpResponse:

    path = Path.cwd()
    learn = load_learner(path)

    request_json = req.get_json()
    r = requests.get(request_json['url'])

    if r.status_code == 200:
        temp_image_name = "temp.jpg"        
        with open(temp_image_name, 'wb') as f:
            f.write(r.content)
    else:
        return func.HttpResponse(f"Image download failed, url: {request_json['url']}")

    img = open_image(temp_image_name)
    pred_class, pred_idx, outputs = learn.predict(img)

    return func.HttpResponse(f"request_json['url']: {request_json['url']}, pred_class: {pred_class}")
```

#### \<FUNCTION_NAME>/function.json

Update the function authorization so that it can be called without any additional security key.  Replace the corresponding line in the file with the following:

```json
...
      "authLevel": "anonymous",
...
```

#### export.pkl

Copy your trained model file, `export.pkl`, to <PROJECT_DIR>.

### Test Function

Run the following command to start the function on your local machine:

```bash
func host start
```

This will give you an output with the URL for testing:

```bash
Now listening on: http://0.0.0.0:7071
Application started. Press Ctrl+C to shut down.

Http Functions:

	inference_function: [GET,POST] http://localhost:7071/api/<FUNCTION_NAME>
```

### Check Test Outputs

To check that your function is running properly, visit http://localhost:7071 and you should see the following:

![Azure Local Running](https://i.imgur.com/fyVVkKp.png)

You can send a HTTP POST method to  `http://localhost:7071/api/<FUNCTION_NAME>` to check that your inference function is working.  Replace `<URL_TO_IMAGE>` with a URL that points to the image:

```json
POST http://localhost:7071/api/<FUNCTION_NAME> HTTP/1.1
content-type: application/json

{
    "url": "<URL_TO_IMAGE>"
}
```

You should then see a HTTP response:

```http
HTTP/1.1 200 OK
Connection: close
Date: Sun, 17 Mar 2019 06:30:29 GMT
Content-Type: text/plain; charset=utf-8
Server: Kestrel
Content-Length: 216

request_json['url']: <URL_TO_IMAGE>, pred_class: <PREDICTED_CLASS>
```

You should see <PREDICTED_CLASS> be replaced with the class that your inference function predicts.

You can press `Ctrl+C` to stop the testing when you're ready.

## 2 - Docker Setup

### Build Docker image

You can now build the Docker image that will contain your app and all the python libraries that it needs to run:

```bash
docker build --tag <DOCKER_HUB_ID>/<DOCKER_IMAGE_NAME>:<TAG> .
```
If the build throws error like 
```
unable to execute 'gcc': No such file or directory
```
Add following codes into Dockerfile **before** the last RUN command.
```
RUN apt-get update && \
    apt-get install -y build-essential
```

### Test Docker image

The following will run the Docker image on your local machine for testing:

```bash
docker run -p 8080:80 -it <DOCKER_HUB_ID>/<DOCKER_IMAGE_NAME>:<TAG>
```

You app in the Docker image is now running at the URL: `localhost:8080`.  You can run the same tests in [**Check Test Outputs**](###Check-Test-Outputs) with the new URL and you should see the same test output as before.

### Push Docker image to Docker Hub

Use the following command to log in to Docker from the command prompt.  Enter your Docker Hub password when prompted.

```bash
docker login --username <DOCKER_HUB_ID>
```

You can now push the Docker image created earlier to Docker Hub:

```bash
docker push <DOCKER_HUB_ID>/<DOCKER_IMAGE_NAME>:<TAG>
```

## 3 - Azure Setup

### Setup Azure Resources

Login to Microsoft Azure with Azure CLI if you haven't already:

```bash
az login
```

You can now run the following commands to create the Azure resources necessary to run the inference app on Azure Functions.  

The following example uses the lowest pricing tier, B1.

Replace the following placeholders with your own names:

- <RESOURCE_GROUP>
  - name of the Resource Group that all other Azure Resources created for this app will fall under
  - e.g. `ResourceGroup`
- <LOCATION_ID>
  - run the following command to see the list of available locations:
    - `az appservice list-locations --sku B1 --linux-workers-enabled`
  - e.g. `centralus`
- <STORAGE_ACCOUNT>
  - name of the Azure Storage Account which is a general-purpose account to maintain information about your function
  - must be between 3 and 24 characters in length and may contain numbers and lowercase letters only
  - e.g. `inferencestorage`
- <FUNCTION_APP>
  - name of the Azure Function App that you will be creating
  - will be the default DNS domain and must be unique across all apps in Azure
  - e.g. `inferenceapp123`

#### Create Resource Group

```bash
az group create \
--name <RESOURCE_GROUP> \
--location <LOCATION_ID>
```

#### Create Storage Account

```bash
az storage account create \
--name <STORAGE_ACCOUNT> \
--location <LOCATION_ID> \
--resource-group <RESOURCE_GROUP> \
--sku Standard_LRS
```

#### Create a Linux App Service Plan

```bash
az appservice plan create \
--name <APP_PLAN_NAME> \
--resource-group <RESOURCE_GROUP> \
--sku B1 \
--is-linux
```

#### Create the App & Deploy the Docker image from Docker Hub

```bash
az functionapp create \
--resource-group <RESOURCE_GROUP> \
--name <FUNCTION_APP> \
--storage-account  <STORAGE_ACCOUNT> \
--plan <APP_PLAN_NAME> \
--deployment-container-image-name <DOCKER_HUB_ID>/<DOCKER_IMAGE_NAME>:<TAG>
```

#### Configure the function app

The following assumes the Docker image uploaded earlier in your Docker Hub profile is public.  If you have set it to private, you can see [here](https://docs.microsoft.com/en-us/azure/azure-functions/functions-create-function-linux-custom-image#configure-the-function-app) to add your Docker credentials so that Azure can access the image.

```bash
storageConnectionString=$(az storage account show-connection-string \
--resource-group <RESOURCE_GROUP> \
--name <STORAGE_ACCOUNT> \
--query connectionString --output tsv)  

az functionapp config appsettings set --name <FUNCTION_APP> \
--resource-group <RESOURCE_GROUP> \
--settings AzureWebJobsDashboard=$storageConnectionString \
AzureWebJobsStorage=$storageConnectionString
```

### Run your Azure Function

After the previous command, it will generally take 15-20 minutes for the app to deploy on Azure.  You can also see your app in the [Microsoft Azure Portal](https://portal.azure.com/) under Function Apps.

The URL for your app will be:

`https://<FUNCTION_APP>.azurewebsites.net/api/<FUNCTION_NAME>`

You can run the same tests in [**Check Test Outputs**](###Check-Test-Outputs) with the new URL and you should see the output as before.

### Delete Resource Group

When you are done, delete the resource group.

```bash
az group delete \
--name <RESOURCE_GROUP> \
--yes
```

Remember that with the App Service plan, you are being charged for as long as you have resources running, even if you are not calling the function.  So it is best to delete the resource group when you are not calling the function to avoid unexpected charges.

## References:

[Create your first Python function in Azure (preview)](https://docs.microsoft.com/en-us/azure/azure-functions/functions-create-first-function-python)

[Create a function on Linux using a custom image](https://docs.microsoft.com/en-us/azure/azure-functions/functions-create-function-linux-custom-image)

[Azure Functions Python developer guide](https://docs.microsoft.com/en-us/azure/azure-functions/functions-reference-python)
