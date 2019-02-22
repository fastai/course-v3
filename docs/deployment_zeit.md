---
title: "Deploying on Zeit"
sidebar: home_sidebar
---

## THIS IS NOT WORKING

*Unfortunately Zeit changed their API in a way that broke all fastai deployments. We suggest using an alternative service, such as render.com*

<img alt="" src="/images/zeit/zeit_now.png" class="screenshot">

This is quick guide to deploy your trained models using the [Now](https://zeit.co/now) service from [Zeit](https://zeit.co/).  This guide comes with a starter app deploying Jeremy's Bear Image Classification model form Lesson 2.

## One-time setup

### Install Now's CLI (Command Line Interface)

```bash
sudo apt install npm # if not already installed
sudo npm install -g now
```

### Grab starter pack for model deployment

```bash
wget https://github.com/fastai/course-v3/raw/master/docs/production/zeit.tgz
tar xf zeit.tgz
cd zeit
```

## Per-project setup

### Upload your learner file

Upload your trained learner file (for example `export.pkl`) to a cloud service like Google Drive or Dropbox. Copy the download link for the file. **Note:** the download link is the one which starts the file download directly&mdash;and is normally different than the share link which presents you with a view to download the file (use [https://rawdownload.now.sh/](https://rawdownload.now.sh/) if needed)

If you want to just test the deployment initially, you can use Jeremy's bear classification model from Lesson 2, you can skip this step, since that model's weights URL is already filled in the sample app.

### Customize the app for your model

1. Open up the file `server.py` inside the `app` directory and update the `learner_file_url` variable with the url copied above

### Deploy

On the terminal, make sure you are in the `zeit` directory, then type:

```bash
now
```

The first time you run this, it will prompt for your email address and create your Now account for you. After your account is created, run it again to deploy your project.

Every time you deploy with `now` it'll create a unique **deployment URL** for the app. It has a format of `xxx.now.sh`, and is shown while you are deploying the app. When the **deployment finishes** and it shows *"> Success! Deployment ready"* on the terminal, type in the terminal:

```
export NAME='changeme:this-is-your-name-for-the-url'
now alias $NAME
```

This will alias the above mentioned deployment URL to `$NAME.now.sh`. You can do this everytime after you deployed. With that, you have a single URL for your app.

### Scaling

By default all deployment goes to sleep after some inactive time. This is not good for the latest version of your app. So do this:

```
# You only need to do this once.
now scale $NAME.now.sh sfo 1
```

### Test the URL of your working app

Go to `$NAME`.now.sh in your browser and test your app.

## Local testing

In case you want to run the app server locally, make these changes to the above steps:

Instead of

```bash
now
```

type in the terminal:

```bash
python app/server.py serve
```

Go to [http://localhost:5042/](http://localhost:5042/) to test your app.

---

*Thanks to Navjot Matharu for the initial version of this guide, and Simon Willison for sample code.*

