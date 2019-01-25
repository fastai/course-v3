# FAQ

## Do I need a GPU to do deep learning?

Yes. GPU's are ~10/15 times faster than CPU's in running neural networks. You will need a GPU-enabled server but don't worry, we've got you covered. Check out our tutorials for renting deep learning servers or Jupyter Notebook Platforms [here](https://course.fast.ai).

## What is the difference between a Jupyter Notebook Platform and a Deep Learning Server?

A deep learning server consists of a machine with a GPU into which you can SSH from the command line whereas a Jupyter Notebook Platform features just a Jupyter Notebook environment which you can open in a browser window without the need to SSH.

## Should I use a Jupyter Notebook Platform or a Deep Learning Server? 

If only to run the fast.ai lessons and experiment in Jupyter Notebooks, we suggest you to use a Jupyter Notebook Platform since it require less steps to set up and will have a Jupyter Notebook automatically open as soon as the instance is started.

However, if you use a deep learning server you will learn useful skills like how to SSH and maintain a server and improve your usage of the command line. These skills might prove useful if you ever want to deploy a program to production like for example hosting an App.

## What are the differences between providers?

Different providers offer different machine configurations and different user interfaces. The other meaningful difference is pricing. We suggest to go with the one you find easier to use (if you already have an account) or set-up (if not). That said, Salamander, Crestle, and FloydHub have the useful feature of switching between GPU and CPU without restarting the instance, which is useful to experiment (data pre-processing in CPU and training only in GPU saves money). On the other hand, Salamander and Crestle take longer to boot up because they rely on [Amazon spot instances](https://aws.amazon.com/ec2/spot/) that are not always available.

Whatever provider you choose, you should make sure **from the first day** that you can set up and running a GPU-enabled Jupyter Notebook fairly quickly. Deep Learning is a tiresome endeavor and your development setup cannot be an obstacle. If there is any problem you cannot solve in your own, the [forums](forums.fast.ai) are your best friend, specially the [Install Issues Thread](https://forums.fast.ai/t/fastai-v1-install-issues-thread/24111/87).
