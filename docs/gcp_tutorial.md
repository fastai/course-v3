---

title: GCP
sidebar: home_sidebar


---

# Welcome to GCP!

This guide explains how to set up Google Cloud Platform (GCP) to use PyTorch 1.0.0 and fastai 1.0.2. At the end of this tutorial you will be able to use both in a GPU-enabled Jupyter Notebook environment.

## Pricing

A `n1-highmem-2` instance in Google which is what we suggest is $0.1184 per hour. Attaching a Tesla k80 GPU costs $0.45 per hour so both together amount to [$0.684 per hour](https://cloud.google.com/compute/pricing).

## Step 1: Creating your account

Cloud computing allows users access to virtual CPU or GPU resources on an hourly rate, depending on the hardware configuration. Find more information in the [Google Cloud Platform documentation](https://cloud.google.com/compute/). In case you don't have a GCP account yet, you can create one [here](https://cloud.google.com/),  which comes with $300 worth of usage credits for free. 

>  **Potential roadblock**: Even though GCP provides a $300 initial credit, you must enable billing to use it. For a new bank account it will take several days for the activation. 

The project on which you are going to run the image needs to be linked with your billing account. For this navigate to the [billing dashboard](https://console.cloud.google.com/billing/projects), click the '**...**' menu and choose '**change billing account**'.

## Step 2: Start an instance
First, go to the [marketplace page](https://console.cloud.google.com/marketplace/details/click-to-deploy-images/deeplearning) of Deep Learning images and click 'launch on compute engine'. Then, select a project, if you want/need to add a new one you can do that with the '+' button on the top right. Finally, select 'Pytorch 1.0 Preview/FastAi 1.0' in the 'Frameworks' section, check 'Install NVIDIA GPU driver automatically on first startup?' and click 'Deploy'. For now leave all other fields to their default values, as soon as you are getting more profecient using GCP you can read the GCP documentation for all details. 

![image_drivers](images/gcp_tutorial/image_drivers.png)

Leave open the tab that is opened after your instance is deployed since we will need to copy some information from it on the next step.  

## Step 3: Connect to your instance

To be able to connect to your instance from your command line you will first need to install the Google Cloud's command line interface (CLI) which you can do by following [this guide](https://cloud.google.com/sdk/docs/#install_the_latest_cloud_tools_version_cloudsdk_current_version). When prompted in the command line to choose a cloud project to use, choose the number representing your new project. Finally, select 'N' when prompted to configure a default Compute Region and Zone.

Once your CLI is installed and your instance is up, you can use SSH to connect to it by running the command you will find in the right hand side in your webpage after hitting 'Deploy'. You will need to change '8080:localhost:8080' to '8888:localhost:8888'. It will look something like this:

``gcloud compute ssh --project $PROJECT_NAME --zone $ZONE_NAME $INSTANCE_NAME -- -L 8888:localhost:8888``

![ssh](images/gcp_tutorial/ssh.png)

Before you are able to connect, Google Cloud will ask you to create an SSH key and authenticate your account. Just follow the prompts, choose a passphrase and save it somewhere safe.

If everything went ok, you should now be connected to your GCP instance!

## Step 4: Access fast.ai materials

Run `git clone https://github.com/fastai/course-v3` in your terminal to get a folder with all the fast.ai materials. 

Next move into the directory where you will find the materials for the course by running:

`cd course-v3/nbs`

Finally run `jupyter notebook` in your terminal, copy the URL starting with _localhost:_ and paste it in your browser. Voilà! Now you can experiment yourself with fast.ai lessons! If it is your first time with Jupyter Notebook, refer to our [Jupyter Notebook tutorial](http://course-v3.fast.ai/dlami_tutorial.html).

If you have any problem while using the `fastai` library try running `conda update -all`.

## Step 5: Stop an instance

![stop](images/gcp_tutorial/stop_meme.jpg)

You will be charged if you don't stop the instance while it's 'idle' (e.g. not training a network). To stop an instance out of Google Cloud's online interface go [here](https://console.cloud.google.com/compute/instances), click the '...' icon to the right of the instance and choose 'Stop'.

![gcp-stop-instance](images/gcp_tutorial/stop_instance.png)





## References

+ [Setting up PyTorch and Fast.ai in GCP](https://blog.kovalevskyi.com/google-compute-engine-now-has-images-with-pytorch-1-0-0-and-fastai-1-0-2-57c49efd74bb)
+ [Launching a PyTorch Deep Learning VM Instance](https://cloud.google.com/deep-learning-vm/docs/pytorch_start_instance)
+ [Google Cloud SDK Quickstart for Debian and Ubuntu](https://cloud.google.com/sdk/docs/quickstart-debian-ubuntu)
+ [Installing the latest Cloud SDK version](https://cloud.google.com/sdk/docs/#install_the_latest_cloud_tools_version_cloudsdk_current_version)
+ [Installing Google Cloud SDK (StackOverflow question)](https://stackoverflow.com/questions/46822766/sudo-apt-get-update-sudo-apt-get-install-google-cloud-sdk-cannot-be-done)
+ [sudo apt-get update && sudo apt-get install google-cloud-sdk cannot be done (StackOverflow answer)](https://stackoverflow.com/a/47908542/45963)

---

*Many thanks to Marcel Ackermann, Antonio Rueda Toicen, Viacheslav Kovalevskyi and Francisco Ingham for their contributions to this guide.*
