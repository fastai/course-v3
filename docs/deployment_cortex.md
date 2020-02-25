---
title: "Deploying on Cortex"
sidebar: home_sidebar
---

# Cortex deployment

This a guide to deploying your models with [Cortex](https://github.com/cortexlabs/cortex), a free and open source platform for deploying models as APIs on AWS. Cortex will automatically manage your APIs, auto scale your instances, and optimize your AWS spend.

Before you begin, you will also need to [install Cortex.](https://www.cortex.dev/install). Once installed, you can begin deploying.

## Configure a Cortex deployment

Cortex's [Python Predictor API](https://www.cortex.dev/deployments/python), the API you'll be using to deploy fastai/PyTorch models, needs the following to deploy your model:

1. A Python script to load your model and serve predictions
2. A YAML file to configure the deployment
3. A text file to list your dependencies

 For this example, we'll be using an ULMFiT model [trained on IMDB reviews](https://docs.fast.ai/text.html#Quick-Start:-Training-an-IMDb-sentiment-model-with-ULMFiT), which you can train and export into Google Drive with [this notebook.](https://colab.research.google.com/drive/1fLnQ5tDp7c37GcSjf2a2n-ezo523uZGE)

Your model serving script needs a `predict()` function that takes queries and serves predictions. For this deployment, you can call your script `predictor.py` and use the following code:

```python
from fastai.text import *
import requests


class PythonPredictor:
    def __init__(self, config):
        # Download model file
        req = requests.get("YOUR MODEL URL")
        with open("export.pkl", "wb") as model:
            model.write(req.content)

        # Initialize model
        self.predictor = load_learner(".")

    def predict(self, payload):
        prediction = self.predictor.predict(payload["text"])
        return prediction[0].obj
```

Now, Cortex needs a YAML file called `cortex.yaml` to configure your deployment. You can read more about the possible configurations of `cortex.yaml` in the [Cortex docs,](https://www.cortex.dev/deployments/python) but for this deployment, the following will work:

```YAML
- kind: deployment
  name: sentiment

- kind: api
  name: analyzer
  predictor:
    type: python
    path: predictor.py
  compute:
    cpu: 1
```

Finally, Cortex needs a file called `requirements.txt` that lists the dependencies needed for `predictor.py` to run. For your deployment, `requirements.txt` can look like this:

```
fastai==1.*
```
## Deploy with the Cortex CLI

Once you have a directory with the above files, you can deploy directly from the command line. If you haven't spun up a cluster yet, you'll need to run
```
$ cortex cluster up
```
Once your cluster is up, deploying and serving predictions with Cortex is easy:

![Cortex Example](https://cortex-readme-gifs.s3-us-west-2.amazonaws.com/cortex-fastai-optimize.gif)
