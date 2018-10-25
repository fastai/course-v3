---
title: Returning to Salamander
keywords:
sidebar: home_sidebar
---

To return to your notebook, the basic steps will be:

1.  Start your instance
1.  Update the course repo
1.  Update the fastai library
1.  When done, shut down your instance

## Step by step guide

### Start your instance

Sign in to [Crestle](https://www.crestle.ai/) and choose the instance you want to start...

<img alt="" src="/images/salamander/start.png" class="screenshot">

### Update the course repo

To update the course repo, you will need to be in terminal. If you used the `ssh` method, you're already there, if you clicked on the 'Jupyter Notebook' button, launch a new terminal from the jupyter notebook menu.

```bash
cd fastai-courses/course-v3
git pull
```

<img alt="" src="/images/gradient/update.png" class="screenshot">

This should give you the latest of the course notebooks. If you modified some of the notebooks in course-v3/nbs directly, GitHub will probably throw you an error. You should type `git stash` to remove your local changes. Remember you should always work on a copy of the lesson notebooks.

### Update the fastai library

To update the fastai library, open the terminal like before and type

```bash
conda update conda
conda install -c fastai fastai
```

### Stop your instance

Once you're finished...

**It's not enough to just close your browser or turn off your own computer.**
