---
title: GCP
sidebar: home_sidebar
---

# Welcome to GCP!

This guide explains how to set up Google Cloud Platform (GCP) to use PyTorch 1.0.0 and fastai 1.0.2. At the end of this tutorial you will be able to use both in a GPU-enabled Jupyter Notebook environment.

## Pricing

A `n1-highmem-8` preemtible instance in Google which is what we suggest is $0.1per hour. Attaching a P100 GPU costs $0.43 per hour so both together amount to [$0.53 per hour](https://cloud.google.com/compute/pricing).

## Step 1: Creating your account

Cloud computing allows users access to virtual CPU or GPU resources on an hourly rate, depending on the hardware configuration. Find more information in the [Google Cloud Platform documentation](https://cloud.google.com/compute/). In case you don't have a GCP account yet, you can create one [here](https://cloud.google.com/),  which comes with $300 worth of usage credits for free. 

>  **Potential roadblock**: Even though GCP provides a $300 initial credit, you must enable billing to use it. For a new bank account it will take several days for the activation. 

![verify bank](/images/gcp/bank_account.png)

The project on which you are going to run the image needs to be linked with your billing account. For this navigate to the [billing dashboard](https://console.cloud.google.com/billing/projects), click the '**...**' menu and choose '**change billing account**'.

## Step 2: Install Google CLI

To create then be able to connect to your instance, you'll need to install Google Cloud's command line interface (CLI) software from Google. For Windows user, we recommend that you use the [Ubuntu terminal](/terminal_tutorial) and follow the same instructions as Ubuntu users (remember you paste with shift + right click in the terminal). 

To install on Linux or Windows (in Ubuntu terminal), follow those four steps:
``` bash
# Create environment variable for correct distribution
export CLOUD_SDK_REPO="cloud-sdk-$(lsb_release -c -s)"

# Add the Cloud SDK distribution URI as a package source
echo "deb http://packages.cloud.google.com/apt $CLOUD_SDK_REPO main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list

# Import the Google Cloud Platform public key
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

# Update the package list and install the Cloud SDK
sudo apt-get update && sudo apt-get install google-cloud-sdk
```
You can find more details on the installation process [here](https://cloud.google.com/sdk/docs/quickstart-debian-ubuntu)

To install on MacOS, follow the instructions detailed in points 1 to 4 [here](https://cloud.google.com/sdk/docs/quickstart-macos). .

In both cases, once the installation is done run this line
``` bash
gcloud init
```

You should then be prompted with this message:
```
To continue, you must log in. Would you like to log in (Y/n)?
```
Type Y then copy the link and paste it to your browser. Choose the google account you used duing step 1, click 'Allow' and you will get a confirmation code to copy and paste to your terminal.

Then, if you are more than one project, you'll be prompted to choose your project:
```
Pick cloud project to use:
 [1] [my-project-1]
 [2] [my-project-2]
 ...
 Please enter your numeric choice:
 ```
Just enter the number next to the project you created on step 1. 

Lastly, you'll be asked if you want to put a default region, choose us-west1-b if you don't have any particular preference.

Once this is done, you should see this message on your terminal:
```
gcloud has now been configured!
You can use [gcloud config] to change more gcloud settings.

Your active configuration is: [default]
```

## Step 3: Create an instance

To create the instance we recommend, just copy and paste the following command in your terminal. You can change \$INSTANCE_NAME to any name you want for your instance.

```bash
export IMAGE_FAMILY="pytorch-1-0-cu92-experimental" # or "pytorch-1-0-cpu-experimental" for non-GPU instances
export ZONE="us-west1-b"
export INSTANCE_NAME="my-fastai-instance"
export INSTANCE_TYPE="n1-standard-8"
gcloud compute instances create $INSTANCE_NAME \
        --zone=$ZONE \
        --image-family=$IMAGE_FAMILY \
        --image-project=deeplearning-platform-release \
        --maintenance-policy=TERMINATE \
        --accelerator='type=nvidia-tesla-p100,count=1' \
        --machine-type=$INSTANCE_TYPE \
        --boot-disk-size=120GB \
        --metadata='install-nvidia-driver=True' \
        --preemtible
```

You will have to wait a little bit until you see informing you the instance has been created. You can see it online [there](https://console.cloud.google.com/compute/instances) (note that this will be the page you have to go to later to start your instance). You can also read more details about instance creation form the command line [here](https://blog.kovalevskyi.com/deep-learning-images-for-google-cloud-engine-the-definitive-guide-bc74f5fb02bc).

Once this is done, you can connect to your instance by typing:
```bash
gcloud compute ssh --zone=$ZONE jupyter@$INSTANCE_NAME -- -L 8080:localhost:8080
```

Before you are able to connect, Google Cloud may ask you to create an SSH key. Just follow the prompts (the passphrase is optional, if you aren't going to be using this key for anything too secure).

If everything went ok, you should now be connected to your GCP instance! To use it, simply go to [localhost:8080/tree](http://localhost:8080/tree). Note that this only work while you maintain the ssh connection. 

## Step 4: Access fast.ai materials

Run 
```bash
git clone https://github.com/fastai/course-v3
``` 
in your terminal to get a folder with all the fast.ai materials. 

Next from your [jupyter notebook](http://localhost:8080/tree), move into the directory 'course-v3/nbs/' where you will find the materials for the course. Now, click on *notebook_tutorial.ipynb* and follow the instructions there; you're now using Jupyter Notebook!

If you have any problem while using the `fastai` library see the [update page](/update/gcp)

## Step 5: Stop an instance

**You will be charged if you don't stop** the instance while it's 'idle' (e.g. not training a network). To stop an instance out of Google Cloud's online interface go [here](https://console.cloud.google.com/compute/instances), click the '...' icon to the right of the instance and choose 'Stop'.

![gcp-stop-instance](/images/gcp/stop_instance.png)

## References

+ [Setting up PyTorch and Fast.ai in GCP](https://blog.kovalevskyi.com/google-compute-engine-now-has-images-with-pytorch-1-0-0-and-fastai-1-0-2-57c49efd74bb)
+ [Launching a PyTorch Deep Learning VM Instance](https://cloud.google.com/deep-learning-vm/docs/pytorch_start_instance)
+ [Google Cloud SDK Quickstart for Debian and Ubuntu](https://cloud.google.com/sdk/docs/quickstart-debian-ubuntu)
+ [Installing the latest Cloud SDK version](https://cloud.google.com/sdk/docs/#install_the_latest_cloud_tools_version_cloudsdk_current_version)
+ [Installing Google Cloud SDK (StackOverflow question)](https://stackoverflow.com/questions/46822766/sudo-apt-get-update-sudo-apt-get-install-google-cloud-sdk-cannot-be-done)
+ [sudo apt-get update && sudo apt-get install google-cloud-sdk cannot be done (StackOverflow answer)](https://stackoverflow.com/a/47908542/45963)

---

*Many thanks to Marcel Ackermann, Antonio Rueda Toicen, Viacheslav Kovalevskyi and Francisco Ingham for their contributions to this guide.*
