---
title: easyaiforum.cn
keywords: neuralnets deeplearning
sidebar: home_sidebar
---

# Welcome to easyaiforum 
<img alt="" src="/images/easyaiforum/mainpage.png" class="screenshot">

[https://easyaiforum.cn](https://easyaiforum.cn)

easyaiforum.cn provides a ready-to-use machine that you can use with SSH, jupyter, remote desktops and more.You can start your training in less than a minute.

If you are returning to work and have previously completed the steps below, please go to the [returning to work](https://course.fast.ai/update_easyaiforum.html) section.

## Pricing

CPU (RMB 1/h), GPU (RMB 2-6/h), The fee is deducted every half hour.

## Storage

20G, 100G and 500G are available and the cost is 0.5, 2, 5 (RMB/day) respectively. 
The storage cost will be reduced by ￥1 for every ￥5 you have consumed during 24 hours. For example, if you use 100G storage and spend ￥10 in a day, so the storage cost will be free. If you delete it before the fee is deducted, there will be no storage charge.

个人云存储分为20G、100G与500G。
      20G存储费为：0.5元/天，当日消费不低于5元时，免收当日存储费。
      100G存储费为：2元 /天，当日消费不低于10元时，免收当日存储费。
      500G存储费为：5元 /天，当日消费不低于25元时，免收当日存储费。
      当日存储费的收取时间，为次日8:00。【8:01-次日7:59，新建个人云空间并在此期间停用，不收取存储费】

## Step 1: Create an account

Visit [https://www.easyaiforum.cn/register](https://www.easyaiforum.cn/register),If you are not in China, please choose "海外用户注册".

用下面的链接注册账号可以获得10元的免费金额.
Use this link to sign up and you'll get 10 credits [My register Link](https://www.easyaiforum.cn/?re_agent_code=Vc41j0m7CwHi2RIf)

<img alt="" src="/images/easyaiforum/reg.png" class="screenshot">


## Step 2: Earn credits

You can pay the fee by alipay or WeChat.

This is my "推广大使优惠码": Vc41j0m7CwHi2RIf

<img alt="" src="/images/easyaiforum/recharge.png" class="screenshot">

That's not the only way to get the credits, you can get the credits from the following ways:(你可以通过以下方式 来免费获得更多的易学币)

[https://www.easyaiforum.cn/wxRecommend](https://www.easyaiforum.cn/wxRecommend)(活动)

[https://bbs.easyaiforum.cn/thread-512-1-1.html](https://bbs.easyaiforum.cn/thread-512-1-1.html)（活动）

[https://bbs.easyaiforum.cn/thread-1550-1-1.html](https://bbs.easyaiforum.cn/thread-1550-1-1.html)（活动）


## Step 3: Create your server
### Ubuntu Server
When you select an environment, verify that it contains python packages like fastai and pytorch.

当你选择环境的时候，检查这个环境是否包含了fastai和pytorch的python包。

<img alt="" src="/images/easyaiforum/start.png" class="screenshot">

## Step 4: Open Jupyter Notebook
    华中一区:/home/ubuntu/MyFiles/PublicData_1/fast.ai/
	华东二区:/home/ubuntu/MyFiles/PublicData/data_extent_1/fast.ai/
<img alt="" src="/images/easyaiforum/jupyter_open.jpg" class="screenshot">
### Windows Server  [Use Warining](#warining)

All Windows servers have installed the fastai package based on conda virtual environment. And the virtual environment can be activated by typing “conda activate fastai”.

所有的Windows服务器中均安装了基于conda虚拟环境的fastai软件包. 通过输入"conda activate fastai"来激活该虚拟环境.

<img alt="" src="/images/easyaiforum/start_windows.jpg" class="screenshot">

Get the information about how to connect to the windows server

获取连接windows服务器的信息

<img alt="" src="/images/easyaiforum/get_connect.png" class="screenshot">
## Step 5: Study Fast.ai
### Ubuntu Server
如果你只是想运行教程里面的内容，可以直接使用文件夹，如果想要对教程的内容进行修改的尝试，请打开终端复制一份到MyFiles下.

If you just want to run the contents of the tutorial, you can use the folder directly. If you want to modify the contents of the tutorial, please open the terminal and make a copy under MyFiles.

<img alt="" src="/images/easyaiforum/fastai_d.jpg" class="screenshot">

```bash
# 复制目录
cp -r course-v3 /home/ubuntu/MyFiles/
# 或者你也可以复制文件
cp course-v3.tar.gz /home/ubuntu/MyFiles
cd /home/ubuntu/MyFiles && tar -xzf course-v3.tar.gz
```
<img alt="" src="/images/easyaiforum/start_my_fastai.png" class="screenshot">

### Windows Server [Usage Warning](#warning)

After using RDP to enter the windows desktop, open the terminal, activate the fastai environment, run jupyter notebook and specify the running directory as the P drive to start learning fastai. (Note: This content in the P drive is still read-only. If you need to edit notebok, please refer to the use method on ubuntu to copy the files to the desktop. If you want to save files, please store it in the M drive[Do not train model in M drive, it will be very slow]).

使用RDP进入windows桌面后，打开终端，激活fastai环境，运行jupyter并指定运行目录为P盘，即可开始学习fastAI。(注意:P盘里面的内容依旧是只读的，如果需要编辑，请参考ubuntu的使用，将文件复制到桌面。如果要保存请存放在M盘[不要在M盘训练模型，这样会很慢]).

<img alt="" src="/images/easyaiforum/study_w_fastai.jpg" class="screenshot">

## Step 6: Stop instance

<img alt="" src="/images/easyaiforum/stop_instance.png" class="screenshot">

The files under MyFiles will not be lost and will exist after the next boot.

MyFiles下的文件不会丢失，下次开机后还会存在.

## FAQs

### Where can I get more tutorials about easyaifourm.cn? (哪里可以得到更多的关于 easyaiforum.cn 的教程？)
For more tips, visit the tutorial address[ easyaiforum tutorial ](https://bbs.easyaiforum.cn/thread-1039-1-1.html)

If you have trouble using the platform, you can contact customer service to help you solve the problem in the following ways.

如果你在使用中遇到问题，可以采用以下方式联系客服来帮你解决问题:
<img alt="" src="/images/easyaiforum/contact1.png" class="screenshot">

<img alt="" src="/images/easyaiforum/contact2.png" class="screenshot">

### What other connection methods are available at easyaiforum.cn? easyaiforum.cn还有哪些连接方式？

[平台Ubuntu系统(Ubuntu主机)使用教程](https://bbs.easyaiforum.cn/forum.php?mod=viewthread&tid=1151&fromuid=245)
[平台Windows系统(Windows主机)使用教程](https://bbs.easyaiforum.cn/forum.php?mod=viewthread&tid=1056&fromuid=245)


### The dataset location of fast.ai(fast.ai 的数据集的位置)

ubuntu: 
    华中一区 /home/ubuntu/MyFiles/PublicData_1/fast.ai/fast.ai_datasets
	华东二区 /home/ubuntu/MyFiles/PublicData/data_extent_1/fast.ai/fast.ai_datasets

windows: P:/PublicData_1/fast.ai/fast.ai_datasets

### Where can I get help from fast.ai? 哪里可以得到fast.ai的帮助？

[ fast ai ](https://docs.fast.ai/)


### Can I use Windows to learn fastai? 可以用Windows来学习fastai吗？

<span id="warning"> </span>Windows isn't recommended for learning fastai. Because Windows is much slower for running PyTorch than Linux and Windows system grabs more GPU memory.

不推荐你使用windows来学习fastai,因为在Windows里面运行pytorch会比在ubuntu慢,并且在windows中系统会占用更多的显存。
