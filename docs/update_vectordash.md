---

title: Vectordash
sidebar: home_sidebar


---

To return to your notebook, the basic steps will be:

1. Start your instance
2. Update the course repo
3. Update the fastai library
4. When done, shut down your instance

### Start your instance

1. `vectordash ssh $INSTANCE_ID`

<img alt="vectordash_cli" src="/home/chewing/course-v3/docs/images/vectordash/vectordash_cli.png" class="screenshot">

### Update the course repo

To update the course repo, launch a new terminal from the jupyter notebook menu.

This will open a new window, in which you should run those two instructions:

<img alt="" src="/images/gradient/terminal.png" class="screenshot">

```bash
cd /course-v3
git pull
```

<img alt="" src="/images/gradient/update.png" class="screenshot">

This should give you the latest of the course notebooks. If you modified some of the notebooks in course-v3/nbs directly, GitHub will probably throw you an error. You should type `git stash` to remove your local changes. Remember you should always work on a copy of the lesson notebooks.

### Update the fastai library

To update the fastai library, open the terminal like before and type

```bash
conda upgrade fastai
```

### Stop your instance

You will be charged if you don't stop the instance while it's 'idle' (e.g. not training a network).
To stop an instance on Vectordash, go to the [dashboard](http://vectordash.com/dashboard) and click the instance you would like to stop. Once on the instance page, click 'Stop Instance'. *Please note, stopping
an instance destroys it completely so make sure you save your files locally or in a remote storage location.*

<img alt="stop_instance" src="/home/chewing/course-v3/docs/images/vectordash/stop_instance.png" class="screenshot">