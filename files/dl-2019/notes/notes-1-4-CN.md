# 第四课：自然语言，表格数据，推荐系统算法collab, 嵌入层



## 综述

本课里我们将通过IMDb 电影评论数据集，深入学习自然语言NLP。我们的任务是预测影评的正负面情绪；也就是情绪分析。我们将采用[ULMFiT](https://arxiv.org/abs/1801.06146) 算法，这个算法是我们最初在2018年课程中开发的，随后成为了自然语言中的一个革命性变化的一部分。纽约时报还因此发文称[新系统正在揭秘自然语言](https://www.nytimes.com/2018/11/18/technology/artificial-intelligence-language.html)。如今ULMFiT已经成为最准确的情绪分析算法。

基本步骤：

1. 创建（下载预先训练好的）*language model语言模型*，这个模型是在一个巨大的语言数据集如维基百科上训练而来的。（所谓的"语言模型" 就是能够学习预测句子中下一个词的模型）
2. 用你的目标数据集微调这个语言模型（在我们的案例中，目标数据集是IMDb影评数据）
3. 从这个微调的语言模型中提取encoder, 再给配上一个分类器。然后为最后的分类任务（也就是情绪判断）来微调模型。

完成NLP后，我们还会覆盖表格数据问题如excel和数据库中的表格，以及解决推荐系统问题的collaborative filtering 算法。到此为止，我们覆盖了全课程所有的深度学习应用。

就表格数据而言，我们会学到如何使用类别和连续变量，如何使用`fastai.tabular`模块来设置和训练模型。

随后我们将用表格数据问题所学来构建collaborative filtering模型，但是在使用了特殊技巧后，模型的准确度不仅更高，而且更具解释性。

到此，我们已完成了一半的课程，覆盖了全部的应用领域：

- 机器视觉
- 自然语言
- 表格数据
- 推荐系统的 Collaborate filtering

在课程的后半段，我们讲学习这些模型到底是如何工作的，以及如何手写这些模型。本节课，我们将对以下核心概念做梳理：

- 激活层（值）
- 参数（权重）
- 层（affine线性 和非线性）
- 损失函数

我们会在后续的课时中进一步探索以上概念的相关细节。我们会学到对自然语言，Collaborative filtering, 以及表格数据模型都很重要的一种神经网络层设计：嵌入层 *embedding layer*。 我们会发现，其实“嵌入层”就是一种特殊数组乘法matrix multiplication (基于one-hot encoded 数组乘法）的简化算法。

## 资源

### 课程资源

- Notebooks:
  - [lesson4-collab.ipynb](https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson4-collab.ipynb)
  - [lesson4-tabular.ipynb](https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson4-tabular.ipynb)
- Excel spreadsheets:
  - [collab_filter.xlsx](https://github.com/fastai/course-v3/blob/master/files/xl/collab_filter.xlsx)
- [视频节点清单](https://forums.fast.ai/t/lesson-4-links-to-different-parts-in-the-video/30338) by @melonkernel
- [第四课 笔记](https://forums.fast.ai/t/deep-learning-lesson-4-notes/30983) by @PoonamV
- [第四课 详尽笔记](https://github.com/hiromis/notes/blob/master/Lesson4.md) by @hiromi
- [简介版笔记](https://medium.com/@boy1729/deep-learning-ver3-lesson-4-8f085a1e28ca) from @boy1729
- [课内探讨](https://forums.fast.ai/t/lesson-4-in-class-discussion/30318)
- [课内高阶探讨](https://forums.fast.ai/t/lesson-4-advanced-discussion/30319)

### 其他资源

- [QCon.ai keynote on Analyzing &amp; Preventing Unconscious Bias in Machine Learning](https://www.infoq.com/presentations/unconscious-bias-machine-learning)
- [PyBay keynote with case studies of what can go wrong, and steps toward solutions](https://www.youtube.com/watch?v=WC1kPtG8Iz8&list=PLtmWHNX-gukLQlMvtRJ19s7-8MrnRV6h6)
- [Workshop on Word Embeddings, Bias in ML, Why You Don't Like Math, &amp; Why AI Needs You](https://www.youtube.com/watch?v=25nC0n9ERq4)
- [AI Ethics Resources](https://www.fast.ai/2018/09/24/ai-ethics-resources/) (includes links to experts to follow)
- [What HBR Gets Wrong About Algorithms and Bias](http://www.fast.ai/2018/08/07/hbr-bias-algorithms/)
- [When Data Science Destabilizes Democracy and Facilitates Genocide](http://www.fast.ai/2017/11/02/ethics/)
- [Vim and Ctags for fast function definition lookup](https://andrew.stwrt.ca/posts/vim-ctags/)

---

[编辑此页面](https://github.com/fastai/course-v3/edit/master/files/dl-2019/notes/notes-1-4.md).
