---
title: Returning to SageMaker
keywords: 
sidebar: home_sidebar
---

To return to your notebook, the basic steps will be:

1. Start your instance
1. Update packages and course repo
1. Open your notebook
1. When done, shut down your instance

## Step by step guide

1. In your [console notebooks](https://www.paperspace.com/console/notebooks) choose the notebook you want to run and click on the button 'Start' under action.

You can choose a different Virtual Machine type on which you'd like to run your Notebook. This can be extremely useful when you want to start on a lower-end machine type, test everything is okay, then move to a more powerful GPU. Also, sometimes the GPU type that you started the notebook on will be unavailable, in which case you can easily fire it up on a different GPU.

![](/images/gradient/restartNotebook.png)

1. Run thing...

    ```bash
    conda update fastai
    ```
1. When you're done, close the notebook tab, and **remember to click stop!** If you don't, you'll keep getting charged until you click the *stop* button.

    <img src="images/sagemaker/23.png" class="screenshot">

