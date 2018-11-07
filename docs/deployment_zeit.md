---
title: Zeit
sidebar: home_sidebar
---

# Deploying Fastai models on Zeit

*(previously known as now.sh)*

<img alt="" src="/images/zeit/zeit_now.png" class="screenshot">

This is quick guide to deploy your trained models using the [Now](https://zeit.co/now) service from [Zeit](https://zeit.co/).  This guide comes with a starter app deploying Jeremy's Bear Image Classification model form Lesson 2.

## Step 1: Create an account and setup Now
For this course's purposes, the free version ([OSS](https://zeit.co/pricing)) should suffice. To get started, create an account [here](https://zeit.co/signup). Make sure you accept the confirmation email they sent you.

## Step 2: Install Now's Desktop app or CLI (Command Line Interface)

Download from [here](https://zeit.co/download). The Desktop app will download the CLI also.

Or simply install the CLI directly:

```bash
sudo apt install npm # if not already installed
sudo npm install -g now
```

## Step 3: Grab starter pack for deployment of your trained model

Download and unzip the starter pack:

```bash
wget https://github.com/fastai/course-v3/raw/master/docs/production/zeit.tgz
tar xf zeit.tgz
cd zeit
```

## Step 5: Upload your trained model file
Upload your trained model file (for example `stage-2.pth`) to a cloud service like Google Drive or Dropbox. Copy the download link for the file. **Note:** the download link is the one which starts the file download directly&mdash;and is normally different than the share link which presents you with a view to download the file (use [https://rawdownload.now.sh/](https://rawdownload.now.sh/) if needed)

If you want to just test the deployment initially, you can use Jeremy's bear classification model from Lesson 2; download the trained model file from [here](https://www.dropbox.com/s/y4kl2gv1akv7y4i/stage-2.pth?raw=1).

## Step 6: Customize the app for your model

1. Open up the file `server.py` inside the `zeit` directory and update the `model_file_url` variable with the url copied from Step 5
1. In the same file, update the line `classes = ['black', 'grizzly', 'teddys']` with the classes you are expecting from your model
1. Open up the file `client.js` inside the `app/static` directory and update the variable `HOSTURL` to something unique for your app&mdash;for example `'https://something-cool.now.sh'` (Note: In the starter app, this variable is set to `'https://deploy-ml-demo.now.sh'`)

## Step 7: Deploy
On the terminal, make sure you are in the local directory you created in Step 3.

To kick off deployment, type in the terminal:
```bash
now
```

Copy the url that Now assigns the project (shown on the terminal right after the above command)&mdash;example [https://myproject-xxxxxxxxxx.now.sh](https://myproject-xxxxxxxxxx.now.sh)

When the **deployment finishes** and it shows *"> Success! Deployment ready"* on the terminal, type in the terminal:
```
now alias https://myproject-xxxxxxxxxx.now.sh https://something-cool.now.sh
```
(**Note:** `https://something-cool.now.sh` above is the unique url you decided for your app in Step 6.3)

## Step 8: Test the URL of your working app
Copy and paste [https://something-cool.now.sh](https://something-cool.now.sh) in your browser and test your app.

---

## Local server
In case you want to run the app server locally, make these changes to the above steps:

### Step 6.3
Update the `HOSTURL` variable to `'http://localhost:5042'`

### Step 7
Instead of
```bash
now
```
type in the terminal:
```bash
python app/server.py serve
```

### Step 8
Copy and paste [http://localhost:5042/](http://localhost:5042/) in your browser and test your app.

