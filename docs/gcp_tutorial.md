---

title: GCP
sidebar: home_sidebar


---

# Welcome to GCP!

This guide explains how to set up Google Cloud Platform (GCP) to use PyTorch 1.0.0 and fastai 1.0.2. At the end of this tutorial you will be able to use both with Jupyter Lab.

![jupyter](https://cdn-images-1.medium.com/max/1000/1*AKAQ25dYfnYnY0gKzcsWKw.png)


## Step 1: Creating your account
Cloud computing allows users access to virtual CPU or GPU resources on an hourly rate, depending on the hardware configuration. The instance we use here charges at around $1 per hour. Find more information in the [Google Cloud Platform documentation](https://cloud.google.com/compute/). In case you don't have a GCP account yet, you can create one [here](https://cloud.google.com/),  which comes with $300 worth of usage credits for free. 

>  **Potential roadblock**: Even though GCP provides a $300 initial credit, you must enable billing to use it. For a new bank account it will take several days for the activation. 

Furthermore, the project on which you are going to run the image needs to be linked with your billing account. For this navigate to the [billing dashboard](https://console.cloud.google.com/billing/projects), click the '**...**' (https://console.cloud.google.com/marketplace/details/click-to-deploy-images/deeplearningaction menu and choose '**change billing account**'.

<!--- ![](https://raw.githubusercontent.com/andandandand/images-for-colab-notebooks/master/enable-billing.png) 

![create billing account](https://raw.githubusercontent.com/andandandand/images-for-colab-notebooks/master/create-billing-account.png)

![Verify your bank account](https://raw.githubusercontent.com/andandandand/images-for-colab-notebooks/master/verify-your-bank-account-gcp.png)-->

## Step 2: Start an instance
First, go to the [marketplace page](https://console.cloud.google.com/marketplace/details/click-to-deploy-images/deeplearning) of Deep Learning images and click 'launch on compute engine'. Then, select a project, if you want/need to add a new one you can do that with the '+' button on the top right. Finally, select 'Pytorch 1.0 Preview/FastAi 1.0' in the 'Frameworks' section, fill out the rest and click 'Deploy'.

## Step 3: Stop an instance

![stop-vm-instance](https://i.imgflip.com/17koi7.jpg)

You will be charged if you don't stop the instance while it's 'idle' (e.g. not training a network). To stop an instance out of Google Cloud's online interface go [here](https://console.cloud.google.com/compute/instances), click the '...' icon to the right of the instance and choose 'Stop'.

![gcp-stop-instance](https://raw.githubusercontent.com/andandandand/images-for-colab-notebooks/master/gcp-stop-instance.png)

## How to use Jupyter Lab
To connect to Jupyter Lab, you need to install Google Cloud's command line interface (CLI) following [this guide](https://cloud.google.com/sdk/docs/#install_the_latest_cloud_tools_version_cloudsdk_current_version).  

As soon as your instance is created you can use SSH to connect to it:

``gcloud compute ssh $INSTANCE_NAME -- -L 8080:localhost:8080``

You can get this command for your instance name by clicking on `Connect -> View gcloud command`

and open your browser at http://localhost:8080.

That's it! You will be able to access the preloaded notebooks and write new ones using PyTorch and fastai. 

![jupyterlab-screenshot](https://raw.githubusercontent.com/andandandand/images-for-colab-notebooks/master/jupyterlab-screenshot.png)



## References

+ [Setting up PyTorch and Fast.ai in GCP](https://blog.kovalevskyi.com/google-compute-engine-now-has-images-with-pytorch-1-0-0-and-fastai-1-0-2-57c49efd74bb)

+ [Launching a PyTorch Deep Learning VM Instance](https://cloud.google.com/deep-learning-vm/docs/pytorch_start_instance)

+ [Google Cloud SDK Quickstart for Debian and Ubuntu](https://cloud.google.com/sdk/docs/quickstart-debian-ubuntu)

+ [Installing the latest Cloud SDK version](https://cloud.google.com/sdk/docs/#install_the_latest_cloud_tools_version_cloudsdk_current_version)

+ [Installing Google Cloud SDK (StackOverflow question)](https://stackoverflow.com/questions/46822766/sudo-apt-get-update-sudo-apt-get-install-google-cloud-sdk-cannot-be-done)
+ [sudo apt-get update && sudo apt-get install google-cloud-sdk cannot be done (StackOverflow answer)](https://stackoverflow.com/a/47908542/45963)

