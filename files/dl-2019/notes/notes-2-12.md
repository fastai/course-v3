# Lesson 12: Advanced training techniques; ULMFiT from scratch

We implement some really important training techniques today, all using callbacks:

- MixUp, a data augmentation technique that dramatically improves results, particularly when you have less data, or can train for a longer time
- Label smoothing, which works particularly well with MixUp, and significantly improves results when you have noisy labels
- Mixed precision training, which trains models around 3x faster in many situations.

We also implement *xresnet*, which is a tweaked version of the classic resnet architecture that provides substantial improvements. And, even more important, the development of it provides great insights into what makes an architecture work well.

Finally, we show how to implement ULMFiT from scratch, including building an LSTM RNN, and looking at the various steps necessary to process natural language data to allow it to be passed to a neural network.

## Lesson resources

- [Lesson Notes](https://medium.com/@lankinen/fast-ai-lesson-12-notes-part-2-v3-dd53bec89c0b) @Lankinen
- Notebook 10c (and subsequent) requires the NVIDIA [apex](https://github.com/NVIDIA/apex#linux) python library. `pip install git+https://github.com/NVIDIA/apex`

## Papers

- [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412)
- [Rethinking the Inception Architecture for Computer Vision ](https://arxiv.org/abs/1512.00567) (label smoothing is in part 7)
- [ Bag of Tricks for Image Classification with Convolutional Neural Networks](https://arxiv.org/abs/1812.01187)

## Edit this page

To edit this page, [click here](https://github.com/fastai/course-v3/edit/master/files/dl-2019/notes/notes-2-12.md). This will take you to a edit window at GitHub where you can submit your suggested changes. They will automatically be turned in to a [pull request](https://help.github.com/articles/about-pull-requests/) which will be reviewed by an admin prior to publication.
