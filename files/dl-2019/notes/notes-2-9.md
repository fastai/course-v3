# Lesson 9: Loss functions, optimizers, and the training loop

In the last lesson we had an outstanding question about PyTorch's CNN default initialization. In order to answer it, Jeremy did a bit of research, and we start today's lesson seeing how he went about that research, and what he learned.

Then we do a deep dive into the training loop, and show how to make it concise and flexible. First we look briefly at loss functions and optimizers, including implementing softmax and cross-entropy loss (and the *logsumexp* trick). Then we create a simple training loop, and refactor it step by step to make it more concise and more flexible. In the process we'll learn about `nn.Parameter` and `nn.Module`, and see how they work with `nn.optim` classes. We'll also see how `Dataset` and `DataLoader` really work.

Once we have those basic pieces in place, we'll look closely at some key building blocks of fastai: *callbacks*, *DataBunch*, and *Learner*. We'll see how they help, and how they're implemented. Then we'll start writing lots of callbacks to implement lots of new functionality and best practices!

## Lesson resources

- [fastai docs on Callbacks](https://docs.fast.ai/callbacks.html)
- [Lesson notebooks](https://github.com/fastai/course-v3/tree/master/nbs/dl2)
- [Lesson notes](https://medium.com/@lankinen/fast-ai-lesson-9-notes-part-2-v3-ca046a1a62ef) by @Lankinen
- [Discussion thread](https://forums.fast.ai/t/lesson-9-discussion-wiki-2019/41969/16)

## Papers
- [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515) (SELU)
- [Exact solutions to the nonlinear dynamics of learning in deep linear neural networks](https://arxiv.org/abs/1312.6120) (orthogonal initialization)
- [All you need is a good init](https://arxiv.org/abs/1511.06422)
- [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852)-- 2015 paper that won ImageNet, and introduced ResNet and Kaiming Initialization.
- [Fixup Initialization: Residual Learning Without Normalization](https://arxiv.org/abs/1901.09321) -- paper highlighting importance of normalisation - training 10,000 layer network without regularisation

## Other helpful resources

- [Sylvain's talk, An Infinitely Customizable Training Loop](https://www.youtube.com/watch?v=roc-dOSeehM) (from the NYC PyTorch meetup) and the [slides](https://drive.google.com/open?id=1eWWpyHeENyNNCVTtblX2Jm02WZWw-Kes) that go with it
- [Why do we need to set the gradients manually to zero in pytorch?](https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903)
- [What is torch.nn really?](https://pytorch.org/tutorials/beginner/nn_tutorial.html)
- [Blog post explaining decorators](https://pouannes.github.io/blog/decorators/)
- [Primer on Python Decorators](https://realpython.com/primer-on-python-decorators/)
- [Introduction to Mixed Precision Training, Benchmarks using fastai](https://hackernoon.com/rtx-2080ti-vs-gtx-1080ti-fastai-mixed-precision-training-comparisons-on-cifar-100-761d8f615d7f)
- [Explanation and derivation of LogSumExp](https://blog.feedly.com/tricks-of-the-trade-logsumexp/)
- [Blog post about callbacks in fastai #1](https://pouannes.github.io/blog/callbacks-fastai/)
- [Blog post about callbacks in fastai #2](https://medium.com/@edwardeasling/implementing-callbacks-in-fast-ai-1c23de25b6eb)
- [Blog post about weight initialization](https://madaan.github.io/init/)

## Errata

* in nb `02b_initializing.ipynb` a few minor corrections were made where a variance and std were mixed up. Thanks to [Aman Madaan](https://madaan.github.io/init) for pointing those out.

## Edit this page

To edit this page, [click here](https://github.com/fastai/course-v3/edit/master/files/dl-2019/notes/notes-2-9.md). This will take you to a edit window at GitHub where you can submit your suggested changes. They will automatically be turned in to a [pull request](https://help.github.com/articles/about-pull-requests/) which will be reviewed by an admin prior to publication.
