---
title: Deepnote
sidebar: home_sidebar
---

# Fast.ai Deep Learning Course v3 in Deepnote

<img alt="Deepnote + fast.ai" src="/images/deepnote/deepnote_fastai.png" class="screenshot">

This is a quick guide to starting v3 of the Fast.ai course. With [Deepnote](https://deepnote.com/), you get instant access to a python notebook environment with the required fast.ai environment already installed.

If you are returning to work and have previously completed the steps below, please go to the [returning to work](/update_deepnote.html) section.

## Step 1: Duplicate the fast.ai template project

Open [Deepnote's fast.ai v3 project template](https://beta.deepnote.com/project/0fd972c2-a0fe-4ab4-b9a4-730f50f68856) and click Duplicate. This creates your own project that you can modify, run or even share with your friends after signing up to Deepnote.

<img alt="Deepnote duplicate template project" src="/images/deepnote/deepnote_duplicate.png" class="screenshot">

## Step 2: Sign up for Deepnote

Click "Create account" to create a Deepnote account. You'll need to use either Google or GitHub authentication. After that's done, you can edit the project, share it with your friends to collaborate with them, and start the hardware that's needed to execute code.

<img alt="Deepnote sign up" src="/images/deepnote/deepnote_signup.png" class="screenshot">

That's it! There's no need to update the fastai library and pull changes from the fastai repo because the project template is set up in a way that it does it automatically when hardware starts. See the Environment sidebar tab, and open init.ipynb for under-the-hood of this automation.

## Step 3: Start learning Fast.ai!

You should now have a running fast.ai notebook. It only takes a few seconds to provision, and once it's running, you just have to navigate to `nbs/dl1/` in the file tree to see the Deep Learning 1 course notebooks.

<img alt="ready" src="/images/deepnote/deepnote_nbs.png" class="screenshot">

## Step 4: Stopping your Notebook

You don't need to stop your hardware when you're done working. Deepnote keeps it running while it's open in your browser, or if one of your cells is executing code, it keeps running even when you close your browser. Otherwise, the hardware turns off after 15 minutes of inactivity.

---

## Additional considerations:

### Managing Data

Whatever files you upload to your project will stay there. If you need to upload larger datasets, consider creating a 'Data source' in the 'Files & data sources' sidebar.

### Where to get help

Questions or issues related to course content, we recommend posting in the [fast.ai forum](http://forums.fast.ai/).

For Deepnote-specific support, check out [Deepnote docs](https://docs.deepnote.com) or get in touch via in-site chat or email [help@deepnote.com](mailto:help@deepnote.com).

### Resources & Limitations

- Deepnote has a community version that's free. This means you'll be able to use a Python fast.ai environment without worrying about costs.
- Deepnote provides free GPU credits for students and academia. Click 'Change machine' in the environment tab on the left for more details.
- Disk usage is limited to 5 GB per project.
- RAM is limited to 5 GB per project.
