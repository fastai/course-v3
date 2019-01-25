---
title: Returning to Colab
keywords: 
sidebar: home_sidebar
---

## Step by step guide

### Step 1: Accessing Colab and opening notebook

Log in to [Google](https://accounts.google.com/signin/v2/identifier?hl=en-gb&flowName=GlifWebSignIn&flowEntry=ServiceLogin) and head on to the [Colab Welcome Page](https://colab.research.google.com/notebooks/welcome.ipynb#recent=true). 

If you want to open a notebook you worked on beforehand, select 'Google Drive' and then select the notebook you want to work on.

<img alt="stop" src="/images/colab/10.png" class="screenshot">

If you want to open a new course notebook you have not worked on before, click on 'Github'. In the 'Enter a GitHub URL or search by organization or user' line enter 'fastai/course-v3'. You will see all the courses notebooks listed there. Click on the one you are interested in using.

<img alt="stop" src="/images/colab/01.png" class="screenshot">

### Step 2: Updating packages and course repo

To update packages and the course repo, create a code cell in your notebook and run:

```bash
!curl https://course.fast.ai/setup/colab | bash
```

<img alt="" src="/images/colab/07.png" class="screenshot">

Colab terminates your instance after 90 minutes of idle time or after 12 hours of runtime (see [here](https://help.clouderizer.com/running-on-cloud/google-colab/google-colab-faqs)). This script will check if your instance has been terminated and install packages and clone repository again if it has. If it has not (you have been away for less than 90 minutes) the script will just update the packages and repository.

If your notebook has these cells in the top you should delete them before you start working:

```bash
%reload_ext autoreload
%autoreload 2
%matplotlib inline
```

### Step 3: Saving your work

If you are working on a notebook from your Drive you can save by clicking on 'File' and 'Save' or <kbd>CTRL</kbd>+<kbd>S</kbd>.

If you opened a notebook from Github, you will need to save your work to Google Drive. You can do this by clicking on 'File' and then 'Save'. You should see a pop-up with the following message:

<img alt="create" src="/images/colab/09.png" class="screenshot">

Click on 'SAVE A COPY IN DRIVE'. This will open up a new tab with the same file, only this time located in your Drive. If you want to continue working after saving,  use the file in the new tab.
