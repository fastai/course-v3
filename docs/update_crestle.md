---
title: Returning to Crestle.ai
keywords:
sidebar: home_sidebar
---

To return to your notebook, the basic steps will be:

1.  Start your instance
2.  Update the course repo
3.  Update the fastai library
4.  When done, shut down your instance

## Step by step guide

### Start your instance

Sign in to [Crestle](https://www.crestle.ai/) and click on Start Jupyter. The instance should take a min or two to spin up. Your previous work will be automatically loaded.

<img alt="" src="/images/crestle/start_jupyter.png" class="screenshot">

### Update the course repo

To update the course repo, you will need to be in terminal. On the 'Jupyter Notebook' button, launch a new terminal from the jupyter notebook menu.

```bash
cd courses/fast-ai/course-v3/
git pull
```

<img alt="" src="/images/crestle/git_pull.png" class="screenshot">

This should give you the latest of the course notebooks. If you modified some of the notebooks in course-v3/nbs directly, GitHub will probably throw you an error. You should type `git stash` to remove your local changes. Remember you should always work on a copy of the lesson notebooks.

### Update the fastai library

To update the fastai library, open the terminal like before and type:

```bash
conda update conda
conda install -c fastai fastai
```

### Stop your instance

Once you're finished navigate back to the dashboard tab and click Stop Jupyter

**It's not enough to just close your browser or turn off your own computer.**

<img alt="" src="/images/crestle/stop_jupyter.png" class="screenshot">

### Reconnecting to your instance

In order to reconnect in the future you'll just follow the exact same steps listed above, some lesson specific actions may need to be taken though due to updates to the fast.ai course throughout the quarter.

#### Lesson 2

If you just created an instance you're good to go.

If you are a returning user you need to open the terminal and type: `rm -r ~/.datasets/camvid`
