---
title: Zeit
sidebar: home_sidebar
---

# Deploying Fastai models on Zeit (previously known as now.sh)

<img alt="" src="/images/zeit/zeit_now.png" class="screenshot">

This is quick guide to deploy your trained models using the [Now](https://zeit.co/now) service from [Zeit](https://zeit.co/).

This guide comes with a starter app deploying Jeremy's Bear Image Classification model form Lesson 2.

## Step 1: Create an account and setup Now
For this course's purposes, the free version ([OSS](https://zeit.co/pricing)) should suffice. To get started, create an account [here](https://zeit.co/signup).

Make sure you accept the confirmation email they sent you.

## Step 2: Install Now's Desktop app or CLI (Command Line Interface)
From [here](https://zeit.co/download). The Desktop app will download the CLI also.

If on your Operating System, the Desktop app is not installing successfully - try skipping it and directly download the CLI with
```bash
sudo npm install -g now
```

## Step 3: Grab starter pack for deployment of your trained model
Create a new directory on your local machine (for example `myproject`), and copy all 4 files from [here](https://github.com/fastai/course-v3/tree/master/docs/production/zeit) to that directory
1. **app.zip**
2. **Dockerfile**
3. **now.json**
4. **requirements.txt**

## Step 4: Unzip the app
Right click the app.zip file and unzip using the appropriate software on your Operating System.

This should create a directory called **app** in your original directory.

## Step 5: Copy and paste your trained model file into the project
Locate the **stage-X.pth** you had saved while training the model in your jupyter notebook. Copy and paste it at `app/models` (in the **models** directory inside the **app** directory that was created in Step 4).

If you want to just test the deployment initially, you can use Jeremy's bear classification model from Lesson 2,  - download the trained model file from [here](https://www.dropbox.com/s/6zt99q2t3z38zus/stage-2.pth?raw=1).

## Step 6: Customize the app for your model
1. Open up the file **server.py** inside the **app** directory and update `classes = ['black', 'grizzly', 'teddys']` to the classes you are expecting from your model
2. In the same file, make sure this line `learn.load('stage-2')` has the name of the trained model file you copied in Step 5
3. Open up the file **client.js** inside the `app/static` directory and update the variable `HOSTURL` to something unique for your app - for example `'https://something-cool.now.sh'` (Note: In the starter app, this variable is set to `'https://deploy-ml-demo.now.sh'`)

## Step 7: Deploy
On the terminal, make sure you are in the local directory you created in Step 3.

To kick off deployment, type in the terminal:
```bash
now
```

Copy the url that Now assigns the project (shown on the terminal right after the above command) - example [https://myproject-xxxxxxxxxx.now.sh](https://myproject-xxxxxxxxxx.now.sh)

When the **deployment finishes** and it shows *"> Success! Deployment ready"* on the terminal, type in the terminal:
```
now alias https://myproject-xxxxxxxxxx.now.sh https://something-cool.now.sh
```
(**Note:** `https://something-cool.now.sh` above is the unique url you decided for your app in Step 6.3)

## Step 8: Test and share the URL of your working app
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