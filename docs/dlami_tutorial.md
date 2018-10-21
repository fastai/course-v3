---

title: AWS EC2
keywords: 
sidebar: home_sidebar


---
# Welcome to AWS EC2

AWS EC2 provides preconfigured machine images called [DLAMI](https://aws.amazon.com/machine-learning/amis/), which are servers hosted by Amazon that are specially dedicated to Deep Learning tasks. Setting up an AWS EC2 instance, even with DLAMI, can be daunting. But don't worry, we got you covered. In fact, Amazon has a sweet [step by step guide](https://aws.amazon.com/getting-started/tutorials/get-started-dlami/) to set it up and we are going to draw heavily from their tutorial.

## Pricing
A `p2.xlarge` instance in Amazon which is what we suggest, is [$0.9 an hour](https://aws.amazon.com/ec2/instance-types/p2/).

## Step 1: Sign in or sign up

Visit the [AWS webpage](https://aws.amazon.com/) and click on 'Sign In to the Console'.

![Signin](images/dlami_tutorial/signin.png)

If you do not have an account, the button to press will say 'Sign up' instead of 'Sign in to the Console'.

![Signup](images/dlami_tutorial/signup.png)

Next, enter your credentials if you are signing in or e-mail, account name and password if you need to sign up. If you are signing up you will also need to set your credit card details. This will be the credit card to which all the charges of the instance usage will be applied (if you have free credits you will not be charged until they are over). Note that you will also need to provide a phone number that will be called to verify your identity.

## Step 2: Request service limit

If you just created your account, you'll need to ask for an increase limit in the instance type we need for the course (default is 0). First click on 'Services' and then 'EC2'.

![amiubuntu](images/dlami_tutorial/ec2.png)

Then on the left bar, choose Limits, then scroll trhough the list until you find p2.xlarge. You can skip this step if your limit is already 1 or more, otherwise click on 'Request limit increase'.

![limit](images/dlami_tutorial/request_limit.png)

Fill the form like below, by selecting 'Service Limit Increase', choose 'EC2 instance', your region, then 'p2.xlarge' and ask for a new limit of 1.

![limit](images/dlami_tutorial/increase_limit.png)

Type the message '[FastAI] Limit Increase Request' in the use case description box, then select your preferred language and contact method before clicking 'Sbumit'. You should have an automate reply telling you they'll look in your case, then an approval notice (hopefully quickly).

![limit](images/dlami_tutorial/increase_limit2.png)

## Step 3: Launch an instance

First click on 'Services' and then 'EC2'.

![amiubuntu](images/dlami_tutorial/ec2.png)

You can also search for EC2 in the querry bar. 

Once on the EC2 screen, click launch instance.

![launch instance](images/dlami_tutorial/launch_instance.png)

Search for 'deep learning' and select the first option: Deep Learning AMI (Ubuntu) Version 16.0

![amiubuntu](images/dlami_tutorial/amiubuntu.png)

## Step 4: Choose your instance type and launch

Scroll down until you find 'p2.xlarge' and select it. Then press 'Review and Launch'.

![p2](images/dlami_tutorial/p2.png)

Finally, when in the 'Review' tab press 'Launch'.

![launch](images/dlami_tutorial/launch.png)

## Step 5: Save Key Pair

In the pop-up window's first drop-down menu, select 'create a new key pair' and select a name. **This key represents your access to your instance. If you lose the key, you will have no access to your instance. If someone has your key, he/she can access the instance. It is important that you save it in a secure location.**

![key](images/dlami_tutorial/key.png)

## Step 6: Connect to your instance

In the next window click on 'View Instances'. You will see that you have an instance that says 'running' under 'Instance State'. Amazon charges you by the amount of seconds an instance has been running so you should **always stop an instance when you finish** using it to avoid getting extra charges. More on this, on Step 7.

Now copy your Public DNS address, which you will find in the bottom of the page.

![pubdns](images/dlami_tutorial/pubdns.png)

Now it's time to connect! Open your command line terminal (if you are in Windows you will need Putty, see [here](https://docs.aws.amazon.com/dlami/latest/devguide/setup-jupyter-configure-client-windows.html) for a tutorial on how to set it up) and type the following commands:

`cd ~/Downloads` (replace the address with the address in which you located your pem file, ideally not Downloads)

`chmod 0400 <your .pem filename>` (replace 'your .pem filename' with your .pem filename)

`ssh -L localhost:8888:localhost:8888 -i <your .pem filename> ubuntu@<your instance DNS>` (replace 'your .pem filename' with your .pem file's name and replace 'your instance DNS' with your Public DNS address)

## Step 7: Access fast.ai materials

Run `git clone https://github.com/fastai/course-v3` in your terminal to get a folder with all the fast.ai materials. 

Then run these commands to install the necessary packages for experimenting with fast.ai and PyTorch:

`conda install -c pytorch pytorch-nightly cuda92`
`conda install -c fastai torchvision-nightly`

`conda install -c fastai fastai`

Next move into the directory where you will find the materials for the course by running:

`cd course-v3/nbs`

Finally run `jupyter notebook` in your terminal, copy the URL starting with _localhost:_ and paste it in your browser. Voilà! Now you can experiment yourself with fast.ai lessons! If it is your first time with Jupyter Notebook, refer to our [Jupyter Notebook tutorial](http://course-v3.fast.ai/dlami_tutorial.html).

If you have any problem while using the `fastai` library try running `conda update -all`.

## Step 8: Stop your instance when you are done

When you finish working you must go back to your AWS instance and stop it manually to avoid getting extra charges. A good practice is setting a reminder for yourself (when you close your computer or log off) so you never forget to do it!

![stop](images/dlami_tutorial/stop.png)

If you no longer want to use that instance again, you can just terminate it. This means you will never be able to access the information in it, so be careful. To terminate an instance just choose terminate instead of stop.

![terminate](images/dlami_tutorial/terminate.png)

## References

https://aws.amazon.com/getting-started/tutorials/get-started-dlami/

---

*Many thanks to Francisco Ingham for writing this guide.*
