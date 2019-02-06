---
title: "Deploying on Google App Engine"
sidebar: home_sidebar
---

# Google App Engine Deployment

This is a quick guide to deploy your trained models using the Google App Engine Custom runtimes. This guide comes with a starter app deploying Jeremy's Bear Image Classification model form Lesson 2.

## Grab Google App Engine starter pack for model deployment

```bash
wget https://github.com/fastai/course-v3/raw/master/docs/production/google-app-engine.zip

unzip google-app-engine.zip

cd google-app-engine/app
```

## Per-project setup

**Upload your trained model file**

Upload your trained model file (for example stage-2.pth) to a cloud service like Google Drive or Dropbox. Copy the download link for the file. Note: the download link is the one which starts the file download directly and is normally different than the share link which presents you with a view to download the file (use https://rawdownload.now.sh/ if needed)

If you want to just test the deployment initially, you can use Jeremy's bear classification model from Lesson 2, you can skip this step since that model's weights URL is already filled in the sample app.


**Customize the app for your model**

Open up the file server.py inside the app directory and update the *model_file_url* variable with the download URL copied above
In the same file, update the line

`classes = ['black', 'grizzly', 'teddys']` 

with the classes you are expecting from your model.

## Upload your customized app to GitHub

Push your customized app directory to GitHub & ***copy the repo URL***


## Deploy

To begin, Open Google Cloud Dashboard, Click ***Create Project*** and then name your new GCP project. Enable billing in your new GCP project by creating a new billing account or setting an existing one. You will be shown following screen:
<img alt="" src="https://cdn-images-1.medium.com/max/1440/1*J_JfUCxs-WAfsNJsW_gXjQ.png" class="screenshot">

After creating a new project, you will be presented with GCP Dashboard page, Go to the far right corner of the page and click ***Activate Cloud Shell***.
<img alt="" src="https://cdn-images-1.medium.com/max/1440/1*X9XC4D-zQLXDTrWPw9csYw.png" class="screenshot">

NOTE: If you have just sign up for a new google cloud account, it may take some minutes, before ***Activate Cloud Shell*** is enable.

A terminal window will open on the same page.
<img alt="" src="https://cdn-images-1.medium.com/max/1440/1*zswXHm5sxmmy5sIj5x60BQ.png" class="screenshot">

In the shell terminal, Create a google app engine application:

```bash 
gcloud app create
```

Then choose geographical region close to you and press enter, after few minutes, it will show something like *"Success! the app is now created. Please use 'gcloud app deploy' to deploy your first app"*
<img alt="" src="https://cdn-images-1.medium.com/max/1440/1*mjRaAbLgGbPxcv2Fzu8YVA.png" class="screenshot">

Download your customized starter pack app repository from Github, for example, here is fast.ai google cloud engine starter pack:

```bash 
git clone https://github.com/pankymathur/google-app-engine
```

Navigate to your app directory:

```bash 
cd google-app-engine
```

Deploy your app to Google App Engine:

```bash 
gcloud app deploy
```

You will be presented with a screen showing "Services to deploy", enter Y
<img alt="" src="https://cdn-images-1.medium.com/max/1440/1*V2drMPZjBsHHh73wctN1cA.png" class="screenshot">


It will take 8~10 minutes for app engine to deploy your docker based app and provide you the app URL. 

**Test the URL of your working app**

To see your final app open http://YOUR_PROJECT_ID.appspot.com or run the following command in browser shell to launch your app in the browser:

```bash 
gcloud app browse
```

## Local testing
In case you want to run the app server locally, or make any changes to the above steps:

```bash
python app/server.py serve
```

Go to http://localhost:8080/ to test your app.

*Thanks to Pankaj Mathur for this guide, and Simon Willison for sample code.*

---
