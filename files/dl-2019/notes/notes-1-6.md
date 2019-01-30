# Lesson 6: Regularization; Convolutions; Data ethics

## Overview

Today we discuss some powerful techniques for improving training and avoiding over-fitting:

- **Dropout**: remove activations at random during training in order to regularize the model
- **Data augmentation**: modify model inputs during training in order to effectively increase data size
- **Batch normalization**: adjust the parameterization of a model in order to make the loss surface smoother.

Next up, we'll learn all about *convolutions*, which can be thought of as a variant of matrix multiplication with tied weights, and are the operation at the heart of modern computer vision models (and, increasingly, other types of models too).

We'll use this knowledge to create a *class activated map*, which is a heat-map that shows which parts of an image were most important in making a prediction.

Finally, we'll cover a topic that many students have told us is the most interesting and surprising part of the course: data ethics. We'll learn about some of the ways in which models can go wrong, with a particular focus on *feedback loops*, why they cause problems, and how to avoid them. We'll also look at ways in which bias in data can lead to biased algorithms, and discuss questions that data scientists can and should be asking to help ensure that their work doesn't lead to unexpected negative outcomes.

### Lesson Resources

- [Detailed lesson notes](https://github.com/hiromis/notes/blob/master/Lesson6.md) - thanks to @hiromi
- Notebooks:
  - [lesson6-rossmann.ipynb](https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson6-rossmann.ipynb)
  - [rossman_data_clean.ipynb](https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/rossman_data_clean.ipynb)
  - [lesson6-pets-more.ipynb](https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson6-pets-more.ipynb)
- [Lesson 6 in-class discussion thread](https://forums.fast.ai/t/lesson-6-in-class-discussion/31440)
- [Lesson 6 advanced discussion](https://forums.fast.ai/t/lesson-6-advanced-discussion/31442)

### Other Resources

- [platform.ai discussion](https://forums.fast.ai/t/platform-ai-discussion/31445)
- [50 Years of Test (Un)fairness: Lessons for Machine Learning](https://128.84.21.199/pdf/1811.10104.pdf)
- [Convolutions:](http://www.cs.cornell.edu/courses/cs1114/2013sp/sections/S06_convolution.pdf)
- [Convolution Arithmetic:](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md)
- [Normalization:](https://arthurdouillard.com/post/normalization/)
- [Cross entropy loss:](https://gombru.github.io/2018/05/23/cross_entropy_loss/)
- [How CNNs work:](https://brohrer.github.io/how_convolutional_neural_networks_work.html)
- [Image processing and computer vision:](https://openframeworks.cc/ofBook/chapters/image_processing_computer_vision.html)
- ["Yes you should understand backprop":](https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b)
- [BERT state-of-the-art language model for NLP:](https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270)
- [Hubel and Wiesel:](https://knowingneurons.com/2014/10/29/hubel-and-wiesel-the-neural-basis-of-visual-perception/)
- [Perception:](https://grey.colorado.edu/CompCogNeuro/index.php/CCNBook/Perception)

---

[Edit this page](https://github.com/fastai/course-v3/edit/master/files/dl-2019/notes/notes-1-6.md).
