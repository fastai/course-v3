---
title: Returning to FloydHub
keywords: 
sidebar: home_sidebar
---

To return to your Workspace, the basic steps will be:

1. Resume your Workspace
1. Update the course repo
1. When done, shut down your instance

## Step by step guide

### Resume your workspace

Go to your global [Workspaces list](https://www.floydhub.com/workspaces) and find your fast-ai Workspace and click `Resume`.

<img alt="" src="/images/floydhub/workspaceList.png" class="screenshot">

In a few moments, your Workspace will be Running. You can click the name of the Workspace to go into your live Workspace session.

<img alt="" src="/images/floydhub/workspaceListRunning.png" class="screenshot">

Once you're in your Workspace, you can feel free to toggle your Machine between CPU and GPU Powerups. It is a common pattern to use CPU machines on FloydHub during the initial setup and dataset phases of your work, and then Restarting with a GPU Powerup when you're ready to train a model.

### Update the course repo
 To update the course repo, launch a new terminal from your Workspace launcher.

<img alt="" src="/images/floydhub/terminal.gif" class="screenshot">

This will open a new window, in which you you should run:

``` bash
git pull
``` 

This should give you the latest of the course notebooks and repo. If you modified some of the notebooks in course-v3/nbs directly, GitHub will probably throw you an error. You should type `git stash` to remove your local changes. Remember you should always work on a copy of the lesson notebooks.

### Using the latest fastai library
If you created your Workspace using the "Run on FloydHub" button in the course repo, then there's no need to ever update the fastai library in your Workspace, because FloydHub will always install the latest version of `fastai` when your resume your Workspace. 

To give a little more detail, FloydHub will install any packages that you place in your code's `floyd_requirements.txt` file. FloydHub added `fastai` to this file in the course repo, so you'll always have the latest and greatest `fastai` library whenever you are in your Workspace. 🎉

If you ever want to add more packages to your Workspace, you can add them to the `floyd_requirements.txt` file and simply Restart your workspace. This is better than simply `pip install`-ing them in a Workspace terminal, because they will be persisted across your Workspace sessions. [Learn more in the FloydHub docs](https://docs.floydhub.com/guides/jobs/installing_dependencies/).

### Stop your instance
Once you're finished, click the Shutdown button in your Workspace to stop your Session. You can also shutdown your Workspace from the global Workspaces list.

<img alt="" src="/images/floydhub/shutdown.png" class="screenshot">

 **NOTE: you *will* be charged for the time that your notebook is running. You must stop the notebook to stop incurring charges.**