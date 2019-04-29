# Lesson 1: Image classification
# 第一课 你的宠物图片分类

点击左侧和右侧的蓝色箭头按钮来隐藏panel给你更多观看视频的空间。你可以屏幕下方搜索字幕并进行视频时间跳跃。页面下方还有大量有用资源。如果你有任何建议，可以在最下方的“编辑”链接，增加你想添加的链接或编辑。

## 综述

跟随课程，你需要有一个云端GPU能运行fastai（推荐，目前最便宜的是每小时0.5美元），或者在本地设置自己的GPU（非常费时费事，不推荐）。你需要熟悉Jupyter Notebook的使用环境来做深度学习实验。更多最新的GPU指南可以在[课程官网](https://course.fast.ai)中查看。

本课的核心目标是训练一个图片分类器，将宠物种类识别做到最专业级的精确度。实验成功的关键是迁移学习 *transfer learning*，也是本课程的核心平台或模型模版工具之一. 我们会学习如何分析模型以理解错误发生所在。在此过程中，我们将看到模型犯错的地方，就连宠物种类鉴定专家也会判断出错。

我们还将探讨本课程的授课模式，即自上而下，而非自下而上。也就是说，我们是从实验开始，根据需求，逐步深入学习理论，而非传统方式，讲完理论，才慢慢开始实践。这种方法对老师挑战较大非常耗时，但对学生受益颇丰，例如 [education research at Harvard](https://www.gse.harvard.edu/news/uk/09/01/education-bat-seven-principles-educators) by David Perkins.

我们还将讨论在训练模型时如何设置那些最重要的超参数*hyper-parameter*。我们将采用Leslie Smith's fantastic *learning rate finder* method来设置学习率。最后，我们将研究很少讨论但非常重要的*labeling*数据标记, 并学习fastai 库提供的轻松添加图片标注的功能

如果你想要深入理解pytorch的实际工作，可以参看 [this official PyTorch tutorial](https://pytorch.org/tutorials/beginner/nn_tutorial.html) by Jeremy，但先别急，建议你在学完本课程的几节课后再学

## 




## 链接

### 课程资源

- [Course site](https://course.fast.ai), 课程官网包含了所有平台的GPU设置指南
- [Course repo](https://github.com/fastai/course-v3) 课程的github repo
- [fastai docs](http://docs.fast.ai) library文档
- [fastai datasets](https://course.fast.ai/datasets) 课程用到的所有数据集
- Notebooks:
  - [00_notebook_tutorial.ipynb](https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/00_notebook_tutorial.ipynb)
  - [lesson1-pets.ipynb](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson1-pets.ipynb)
- [第一课 详尽笔记](https://github.com/hiromis/notes/blob/master/Lesson1.md) - 感谢 @hiromi
- [第一课笔记](https://forums.fast.ai/t/deep-learning-lesson-1-notes/27748) - 感谢 @PoonamV (wiki thread - 欢迎大家贡献共建!)
- [课程探讨 thread](https://forums.fast.ai/t/lesson-1-discussion/27332)

### 其他资源

- [Thread on creating your own image dataset](https://forums.fast.ai/t/tips-for-building-large-image-datasets/26688)
- [What you need to do deep learning](http://www.fast.ai/2017/11/16/what-you-need/) (fast.ai 博客讲解了什么是GPU以及它们的必要性)
- [Original Paper for Oxford-IIIT Pet Dataset](http://www.robots.ox.ac.uk/~vgg/publications/2012/parkhi12a/parkhi12a.pdf)
- [The Oxford-IIIT Pet Dataset ](http://www.robots.ox.ac.uk/~vgg/data/pets/)
- [What the Regular Expressions in the notebook meant](https://medium.com/@youknowjamest/parsing-file-names-using-regular-expressions-3e85d64deb69)
- [Understanding Regular Expressions](https://youtu.be/DRR9fOXkfRE) (12 分钟视频)
- [Visualize Regular Expressions](https://regexr.com/)
- [Interactive tutorial to learn Regular Expressions](https://regexone.com)
- [Beginners Tutorial of Regular Expression](https://www.analyticsvidhya.com/blog/2015/06/regular-expression-python/)
- [One-Cycle Policy Fitting paper](https://arxiv.org/abs/1803.09820)
- [Visualizing and Understanding Convolutional Networks (paper)](https://arxiv.org/abs/1311.2901)

### 如何从网页爬取图片

- [官方课程指南](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson2-download.ipynb)
- https://forums.fast.ai/t/tips-for-building-large-image-datasets/26688
- https://forums.fast.ai/t/generating-image-datasets-quickly/19079
- https://forums.fast.ai/t/how-to-scrape-the-web-for-images/7446/8

---

## 编辑此页面

编辑此页面, [点击这里](https://github.com/fastai/course-v3/edit/master/files/dl-2019/notes/notes-1-1.md). 你会进入GitHub一个页面让你上交修改。它们会自动生成 [pull request](https://help.github.com/articles/about-pull-requests/) 然后经由管理员审核后发布。
