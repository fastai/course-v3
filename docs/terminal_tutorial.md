---

title: Terminal
keywords: 
sidebar: home_sidebar


---
## Getting a terminal on Windows

If you are a Linux or a Mac user, you already have access to a bash terminal. For Windows users, you will need to install it first. Note that this is only necessary if you don't use one of the ready-to-run option!

We recommand WSL and the [Ubuntu App](https://www.microsoft.com/en-us/p/ubuntu/9nblggh4msv6#activetab=pivot:overviewtab) for Windows 10 users. To install it, note that you first need to activate Windows Subsystem for Linux with the following steps:

1. Open the Control Panel
2. Select 'Programs'
3. Click on 'Turn Windows features on or off'
4. Scroll and click next to 'Windows Subsystem for Linux'
5. Reboot your computer

![Turn WSL on](/images/terminal/wsl.png)

Once this is done, you just have to install Ubuntu from the Windows Store and launch the app. 

Useful tip: you can't paste with Ctrl + V in this terminal, you have to use shift + right click.

## General explanation of the terminal

### What is a terminal? 
It’s a black screen that allows you to interact directly with your computer with some lines of code.


![terminal](/images/terminal/what_is_a_terminal.png)

### Navigating between folders

#### How to navigate into folders?
Write in the terminal “cd” and the name of the folder you want to access.

![cd](/images/terminal/terminal_cd_in.png)

On Windows, you can acces 'C:\' or 'D:\' by adding the prefix /mnt/ :

![mnt](/images/terminal/windows_term.png)

#### How to navigate out of folder?
Write in the terminal “cd ..”

![nav](/images/terminal/terminal_cd_out.png)

### Important tips

In order to move faster and avoid typo errors while navigating between folders, you should use Tab. You just need to start to type the file or directory name and once you wrote enough for it to be identified, you can press Tab.

If there is only one folder or file that starts with the characters you wrote, the name will autocompleted. If there are many that start that way, you can press double Tab to list them; in order words, it is like using the command "ls" .

### How to create a folder?
Write in the terminal "mkdir” and add the name of the folder.

![mkdir](/images/terminal/terminal_mkdir.png)

### How to know where I am?
By writing “pwd”, the terminal will tell you where you are; in other words, the file directory you are working on.

![pwd](/images/terminal/terminal_pwd.png)

### How to display files?
Write “ls” in your terminal it will list your files.


![ls](/images/terminal/terminal_ls.png)

### How to copy a folder?
Write in the terminal “cp” with the file you want to copy and add the final destination folder where you want to paste it.

![cp](/images/terminal/terminal_cp.png)

### Git and conda in terminal

#### Cloning fastai repository (Download tha fastai files)

Go to this url: https://github.com/fastai/fastai, and copy the git url. Click on the number 1 and 2 to have it on your clipboard.


![clone](/images/terminal/git_copy_url.png)

Then, in order to clone the fastai repo, use the friendly command "git clone” and 
then paste the url your terminal by pressing <kbd>CTRL</kbd>+<kbd>V</kbd>

![paste](/images/terminal/git_clone_repo_.png)

#### Updating your fastai repository (Downloading the missing files)

When your repo is up-to-date, you should see this message.

![pull1](/images/terminal/git_pull_up_to_date.png)

If there are new missing files, you wil see this.

![pull2](/images/terminal/git_pull_new_files.png)

#### Updating Conda 

On more thing, if you want to update conda, you should write this in the terminal "conda update conda" and you are done.

![update](/images/terminal/conda_update.png)

---

*Many thanks to Kevin Martell for writing this guide.*