# Lesson 10: Looking inside the model

In lesson 10 we start with a deeper dive into the underlying idea of callbacks and event handlers. We look at many different ways to implement callbacks in Python, and discuss their pros and cons. Then we do a quick review of some other important foundations:

- `__dunder__` special symbols in Python
- How to navigate source code using your editor
- Variance, standard deviation, covariance, and correlation
- Softmax
- Exceptions as control flow

Next up, we use the callback system we've created to set up CNN training on the GPU.

Then we move on to the main topic of this lesson: looking inside the model to see how it behaves during training. To do so, we first need to learn about *hooks* in PyTorch, which allow us to add callbacks to the forward and backward passes. We will use hooks to track the changing distribution of our activations in each layer during training. By plotting this distributions, we can try to identify  problems with our training.

In order to fix the problems we see, we try changing our activation function, and introducing batchnorm. We study the pros and cons of batchnorm, and note some areas where it performs poorly. Finally, we develop a new kind of normalization layer to overcome these problems, and compare it to previously published approaches, and see some very encouraging results.

## Lesson resources

- [Lesson notebooks](https://github.com/fastai/course-v3/tree/master/nbs/dl2)
- [Lesson notes](https://medium.com/@lankinen/fast-ai-lesson-10-notes-part-2-v3-aa733216b70d) by @Lankinen
- [Interpreting the colorful histograms used in this lesson](https://forums.fast.ai/t/the-colorful-dimension/42908)

## Papers to read

- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
- [Layer Normalization](https://arxiv.org/abs/1607.06450)
- [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022)
- [Group Normalization](https://arxiv.org/abs/1803.08494)
- [Revisiting Small Batch Training for Deep Neural Networks](https://arxiv.org/abs/1804.07612)

## Errata

- The layer and instance norm code in the video use `std` instead of `var`. This is fixed in the notebook
- Jeremy said `binomial` when he meant `binary`.
- Jeremy said "Variance of Batch of 1 is infinite," when he meant zero. The normalized output value will become infinite if the batch 
  size is 1.

## Edit this page

To edit this page, [click here](https://github.com/fastai/course-v3/edit/master/files/dl-2019/notes/notes-2-10.md). This will take you to a edit window at GitHub where you can submit your suggested changes. They will automatically be turned in to a [pull request](https://help.github.com/articles/about-pull-requests/) which will be reviewed by an admin prior to publication.
