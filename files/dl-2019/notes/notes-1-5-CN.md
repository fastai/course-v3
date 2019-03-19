# 第五课：反向传递，加速版随机梯度下降，手写神经网络

## 综述
在本课中我们会深入训练环节细节来讲解什么是反向传递。在此基础上，我们会手写一个简单的神经网络

我们还将深入观察embedding层的参数，看看模型学到了哪些关于类别变量的知识。这些知识将帮助我们识别那些需要权利回避的电影...

尽管embeddings的知名度在自然语言word embeddings领域里是最高的，但在广义的类别变量问题的背景下，如表格数据问题或者是推荐算法问题里，他们的重要性不容小觑。他们甚至在非神经网络模型里也有杰出表现。

## 资源

### 课程资源

- [第六课 笔记](https://forums.fast.ai/t/deep-learning-lesson-5-notes/31298) - 感谢 @PoonamV
- [第六课 详尽笔记](https://github.com/hiromis/notes/blob/master/Lesson5.md) - 感谢 @hiromi
- Notebooks:
  - [lesson5-sgd-mnist.ipynb](https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson5-sgd-mnist.ipynb)
- Excel spreadsheets:
  - [collab_filter.xlsx](https://github.com/fastai/course-v3/blob/master/files/xl/collab_filter.xlsx);
[Google Sheets full version](https://docs.google.com/spreadsheets/d/1oxY9bxgLPutRidhTrucFeg5Il0Jq7UdMJgR3igTtbPU/edit#gid=1748360111); 在需要运行 solver时，请使用Google Sheets short-cut 版本并按照@Moody的[指南](https://forums.fast.ai/t/google-sheets-versions-of-spreadsheets/10424/7)操作
  - graddesc: [Excel](https://github.com/fastai/course-v3/blob/master/files/xl/graddesc.xlsm) version ; [Google sheets](https://docs.google.com/spreadsheets/d/1uUwjwDgTvsxW7L1uPzpulGlUTaLOm8b-R_v0HIUmAvY/edit?usp=sharing) version
  - [entropy_example.xlsx](https://github.com/fastai/course-v3/blob/master/files/xl/entropy_example.xlsx)
- [Lesson 5 in-class discussion thread](https://forums.fast.ai/t/lesson-5-discussion-thread/30864)
- [Lesson 5 advanced discussion](https://forums.fast.ai/t/lesson-5-further-discussion/30865)
- [Links to different parts in video](https://forums.fast.ai/t/lesson-5-links-to-different-parts-in-video/30891) by @melonkernel

### 其他资源

- [NY Times Article - Finally, a Machine That Can Finish Your Sentence](https://www.nytimes.com/2018/11/18/technology/artificial-intelligence-language.html)
- [Netflix and Chill: Building a Recommendation System in Excel - Latent Factor Visualization in Excel blog post](https://towardsdatascience.com/netflix-and-chill-building-a-recommendation-system-in-excel-c69b33c914f4)
- [An overview of gradient descent optimization algorithms - Sebastian Ruder](http://ruder.io/optimizing-gradient-descent/)

---

[编辑此页面](https://github.com/fastai/course-v3/edit/master/files/dl-2019/notes/notes-1-5.md).
