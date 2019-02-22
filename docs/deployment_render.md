---
title: "Deploying on Render"
sidebar: home_sidebar
---

<img alt="Render" src="/images/render/render.svg" class="provider-logo" width="300">

This is quick guide to deploy your trained models on [Render](https://render.com) in just a few clicks. It comes with a [starter repo](https://github.com/render-examples/fastai-v3) that uses Jeremy's Bear Image Classification model from Lesson 2.

The starter app is deployed at [https://fastai-v3.onrender.com](https://fastai-v3.onrender.com).

## One-time setup

### Fork the starter app on GitHub.

Fork [https://github.com/render-examples/fastai-v3](https://github.com/render-examples/fastai-v3) into your GitHub account.

### Create a Render account

Sign up at [Render](https://render.com/i/fastai-v3) using invite code `fastai-v3`.

Render web services cost $5 per month and are prorated by the second. All new accounts start out with $5 in credits.

## Per-project setup

If you just want to test initial deployment on Render, the starter repo is set up to use Jeremy's bear classification model from Lesson 2 by default. If you want to use your own model, keep reading.

### Upload your trained model file

Upload the trained model file created with `learner.export` (for example `export.pkl`) to a cloud service like Google Drive or Dropbox. Copy the download link for the file.

**Note** the download link should start the file download directly&mdash;and is typically different from the share link (which presents you with a view to download the file).

* Google Drive: Use [this link generator](https://www.wonderplugin.com/online-tools/google-drive-direct-link-generator/).
* Dropbox: Use [this link generator](https://syncwithtech.blogspot.com/p/direct-download-link-generator.html)

### Customize the app for your model

1. Edit the file `server.py` inside the `app` directory and update the `model_file_url` variable with the URL copied above.
2. In the same file, update the line `classes = ['black', 'grizzly', 'teddys']` with the classes you expect from your model.

### Commit and push your changes to GitHub.

Make sure to keep the GitHub repo you created above current. Render integrates with your GitHub repo and automatically builds and deploys changes every time you push a change.

## Deploy

1. Create a new **Web Service** on Render and use the repo you created above. You will need to grant Render permission to access your repo in this step.

2. On the deployment screen, pick a name for your service and use `Docker` for the Environment. The URL will be created using this service name. The service name can be changed if necessary, but the URL initially created can't be edited.

3. Click **Save Web Service**. That's it! Your service will begin building and should be live in a few minutes at the URL displayed in your Render dashboard. You can follow its progress in the deploy logs.


## Test your app

Your app's URL will look like `https://service-name.onrender.com`. You can also monitor the service logs as you test your app.

## Local testing

To run the app server locally, run this command in your terminal:

```bash
python app/server.py serve
```

Go to [http://localhost:5042/](http://localhost:5042/) to test your app.

---

*Thanks to Simon Willison for sample code.*
