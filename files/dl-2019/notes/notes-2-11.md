# Lesson 11: Data Block API, and generic optimizer

We start lesson 11 with a brief look at a smart and simple initialization technique called Layer-wise Sequential Unit Variance (LSUV). We implement it from scratch, and then use the methods introduced in the previous lesson to investigate the impact of this technique on our model training. It looks pretty good!

Then we look at one of the jewels of fastai: the Data Block API. We already saw how to use this API in part 1 of the course; but now we learn how to create it from scratch, and in the process we also will learn a lot about how to better use it and customize it. We'll look closely at each step:

- Get files: we'll learn how `os.scandir` provides a highly optimized way to access the filesystem, and `os.walk` provides a powerful recursive tree walking abstraction on top of that
- Transformations: we create a simple but powerful `list` and function composition to transform data on-the-fly
- Split and label: we create flexible functions for each
- DataBunch: we'll see that `DataBunch` is a very simple container for our `DataLoader`s

Next up, we build a new `StatefulOptimizer` class, and show that nearly all optimizers used in modern deep learning training are just special cases of this one class. We use it to add weight decay, momentum, Adam, and LAMB optimizers, and take a look a detailed look at how momentum changes training.

Finally, we look at data augmentation, and benchmark various data augmentation techniques. We develop a new GPU-based data augmentation approach which we find speeds things up quite dramatically, and allows us to then add more sophisticated warp-based transformations.

## Lesson resources

- [Lesson video](https://youtu.be/hPQKzsjTyyQ)
- [Lesson notes](https://medium.com/@lankinen/fast-ai-lesson-11-notes-part-2-v3-6d28e17509f4) by @Lankinen

## Papers

- [L2 Regularization versus Batch and Weight Normalization](https://arxiv.org/abs/1706.05350)
- [Norm matters: efficient and accurate normalization schemes in deep networks](https://arxiv.org/abs/1803.01814)
- [Three Mechanisms of Weight Decay Regularization](https://arxiv.org/abs/1810.12281)
- [Nesterov's Accelerated Gradient and Momentum as approximations to Regularised Update Descent](https://arxiv.org/abs/1607.01981)
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
- [Reducing BERT Pre-Training Time from 3 Days to 76 Minutes](https://arxiv.org/abs/1904.00962)

## Resources
- [Blog post on the interaction between L2 Regularization and Batchnorm, including experiments](https://blog.janestreet.com/l2-regularization-and-batch-norm/)

## Edit this page

To edit this page, [click here](https://github.com/fastai/course-v3/edit/master/files/dl-2019/notes/notes-2-11.md). This will take you to a edit window at GitHub where you can submit your suggested changes. They will automatically be turned in to a [pull request](https://help.github.com/articles/about-pull-requests/) which will be reviewed by an admin prior to publication.
