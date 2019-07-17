# 第六课：正则化，卷积，数据伦理

## 综述

今天我们学习讨论一些帮助我们改进训练和避免过拟合的强大技巧：

- **Dropout**: 随机去除一些激活层上的值，目的是对模型做正则处理
- **Data augmentation数据增强**：对模型输入值做调整，目的是显著增加数据尺寸
- **Batch normalization批量正常化**：规整模型参数值，目的是训练时让损失值变化更平滑

接下来，我们会学习*convolutions卷积*，这个概念可以被理解为一种数组乘法与多个捆绑参数扫描器合作的变体，是当下机器视觉的核心算法（同时也在不断为其他类别的模型所使用）。

我们将基于此创建一个类叫 *activated map*, 其功能是对图片做热力图处理，从而凸显展示图片中对预测最重要的部位。

最后，我们将学习*数据伦理*, 许多学员认为这是一个非常有趣且出乎意料的课程内容。我们会了解到模型在什么情况下会出问题，会着重讲到*feedback loops 反馈循环*， 以及为什么这些会是问题，以及如何规避。我们还会看到数据中的偏差是如何导致算法偏差（歧视）的，还会探讨关于数据科学家能做什么，以及是否应该力争确保他们的工作不会导致意想不到的负面结果。

### 课程资源

- [第六课详细笔记](https://github.com/hiromis/notes/blob/master/Lesson6.md) - 感谢 @hiromi
- Notebooks:
  - [lesson6-rossmann.ipynb](https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson6-rossmann.ipynb)
  - [rossman_data_clean.ipynb](https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/rossman_data_clean.ipynb)
  - [lesson6-pets-more.ipynb](https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson6-pets-more.ipynb)
- [第六课 课内探讨 thread](https://forums.fast.ai/t/lesson-6-in-class-discussion/31440)
- [第六课 深入探讨 (高级)](https://forums.fast.ai/t/lesson-6-advanced-discussion/31442)

### 其他资源

- [platform.ai 平台讨论](https://forums.fast.ai/t/platform-ai-discussion/31445)
- [50 Years of Test (Un)fairness: Lessons for Machine Learning](https://128.84.21.199/pdf/1811.10104.pdf)
- [Convolutions:](http://www.cs.cornell.edu/courses/cs1114/2013sp/sections/S06_convolution.pdf) 什么是卷积
- [Convolution Arithmetic:](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md) 卷积运算可视化解读
- [Normalization:](https://arthurdouillard.com/post/normalization/) 如何理解 normalization
- [Cross entropy loss:](https://gombru.github.io/2018/05/23/cross_entropy_loss/) 如何理解entropy loss
- [How CNNs work:](https://brohrer.github.io/how_convolutional_neural_networks_work.html) CNN 如何工作
- [Image processing and computer vision:](https://openframeworks.cc/ofBook/chapters/image_processing_computer_vision.html) 图片处理与机器视觉
- ["Yes you should understand backprop":](https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b) 如何理解反向传递
- [BERT state-of-the-art language model for NLP:](https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270) 理解当下最先进的语言模型结构
- [Hubel and Wiesel:](https://knowingneurons.com/2014/10/29/hubel-and-wiesel-the-neural-basis-of-visual-perception/) 从脑神经角度理解视觉
- [Perception:](https://grey.colorado.edu/CompCogNeuro/index.php/CCNBook/Perception) CNNbook中perception的解读

---

[编辑此页面](https://github.com/fastai/course-v3/edit/master/files/dl-2019/notes/notes-1-6.md).