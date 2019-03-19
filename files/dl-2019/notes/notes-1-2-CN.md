# 第二课：数据清洗和模型云端应用；手写SGD



## 综述

今天我们要用自己的数据构建属于你的图片分类器，涉及内容包括：
- 图片搜集
- 并行下载
- 创建一个验证集
- 数据清洗，让模型帮助我们找出数据内的瑕疵

我会演示以上步骤，通过一个创建一个模型来区分泰迪熊，棕熊和黑熊。一旦我们的模型训练到位，我们将让这个模型能在云端被调用。

课程录制之后，我们增加很多内容，请关注：
- 云端调用模型平台，例如，[Render.com使用指南](https://course.fast.ai/deployment_render.html)
- Notebook中的新互动界面能帮助我们寻找和修正错误标注的图片

在本节课的后半段，我们将手动创建和训练一个简单模型，并手写我们自己的*梯度下降*循环。在这个过程中，我们将学到很多新名词，请确保做好笔记，因为之后我们会反复使用这些名词。（之后还会学到更多新名词）

## 资源

### 课程资源

- Notebooks:
    - [lesson2-download.ipynb](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson2-download.ipynb)
    - [lesson2-sgd.ipynb](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson2-sgd.ipynb)
- [第二课 详尽笔记](https://github.com/hiromis/notes/blob/master/Lesson2.md) - 感谢 @hiromi
- [第二课 笔记 wiki ](https://forums.fast.ai/t/deep-learning-lesson-2-notes/28772) - 感谢@PoonamV
- [课内讨论](https://forums.fast.ai/t/lesson-2-chat/28722)
- [视频节点列表](https://forums.fast.ai/t/lesson-2-links-to-different-parts-in-video/28777) - 感谢 @melonkernel

### 其他资源

- [ How (and why) to create a good validation set](https://www.fast.ai/2017/11/13/validation-sets/) by @rachel
- [There's no such thing as "not a math person"](https://www.youtube.com/watch?v=q6DGVGJ1WP4) by @rachel
- [Responder](https://github.com/kennethreitz/responder) - a web app framework built on top of Starlette
- Post about an [alternative image downloader/cleaner](https://www.christianwerner.net/tech/Build-your-image-dataset-faster/) by @cwerner
- [A tool for excluding irrelevant images from Google Image Search results](https://forums.fast.ai/t/tool-for-deleting-files-on-the-google-image-search-page-before-downloading/28900) by @melonkernel
- [ Machine Learning is Fun](https://medium.com/@ageitgey/machine-learning-is-fun-part-3-deep-learning-and-convolutional-neural-networks-f40359318721) - source of image/number GIF animation shown in lesson
- [A systematic study of the class imbalance problem in convolutional neural networks](https://arxiv.org/abs/1710.05381), mentioned by Jeremy as a way to solve imbalanced datasets.

---

[编辑此页面](https://github.com/fastai/course-v3/edit/master/files/dl-2019/notes/notes-1-2.md).
