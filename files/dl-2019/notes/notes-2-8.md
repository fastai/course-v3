# Lesson 8: Matrix multiplication; forward and backward passes

In this course, we will learn to implement many of the capabilities of Fastai and PyTorch that we could use to build our own deep learning libraries. Along the way, we will learn to implement research papers, which is an important skill to master when making state of the art models.

## Lesson resources

- [Course notebooks](https://github.com/fastai/course-v3/tree/master/nbs/dl2)
- [Slides](https://drive.google.com/file/d/18QwDI25Lf0ld0-cEugu7LxjwTc2NRkha/view?usp=sharing)
- [Excel spreadsheets](https://github.com/fastai/course-v3/tree/master/files/xl) (today's is called `broadcasting.xlsx`). There's also a [Google Sheet version](https://docs.google.com/spreadsheets/d/1bIPBcf-p9iqNG8BGmIVlJCFa4jEsbOZvcPXGTYe5pjI/edit?usp=sharing) thanks to @Moody

## Lesson overview

In today's lesson we'll discuss the purpose of this course, which is, in some ways, the opposite of part 1. This time, we're not learning practical things that we will use right away, but foundations that we can build on. This is particularly important nowadays because this field is moving so fast. We'll also talk about why the last two lessons of this course are about Swift, not Python (Chris Lattner, the original creator of Swift, and now lead of Swift for TensorFlow, will be joining Jeremy to co-teach these lessons).

We'll also discuss the structure of this course, which is extremely "bottom up" (whereas part 1 was extremely "top down"). We'll start from the lowest level foundations (matrix multiplication) and gradually build back up to state of the art models.

We'll gradually refactor and accelerate our first, pure python, matrix multiplication, and in the process will learn about broadcasting and einstein summation. We'll then use this to create a basic neural net forward pass, and in the process will start looking at how neural networks are initialized (a topic we'll be going into in great depth in the coming lessons).

Then we will implement the backwards pass, including a brief refresher of the chain rule (which is really all the backwards pass is). We'll then refactor the backwards pass to make it more flexible and concise, and finally we'll see how this translates to how PyTorch actually works.

## Lesson notes
* [1](https://medium.com/@lankinen/fast-ai-lesson-8-notes-part-2-v3-8965a6532f51) by  @Lankinen
* [2](https://github.com/WittmannF/fastai-dlpt2-notes/blob/master/lesson-08.md) by @wittmannf
* [3](https://forums.fast.ai/t/lesson-8-notes/41442/22) by @timlee

## Things mentioned in the lesson

- Ensure your fastai and pytorch libs are up to date (`conda install -c fastai -c pytorch pytorch fastai`)
- You'll also need to: `conda install nbconvert` and `conda install fire -c conda-forge`
- Jeremy's blog posts about Swift: [fast.ai Embracing Swift for Deep Learning](https://www.fast.ai/2019/03/06/fastai-swift/) and [High Performance Numeric Programming with Swift: Explorations and Reflections](https://www.fast.ai/2019/01/10/swift-numerics/)
- Rachel's post on starting to blog: [Why you (yes, you) should blog](https://medium.com/@racheltho/why-you-yes-you-should-blog-7d2544ac1045)
- Numpy docs on [broadcasting](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)and [einsum](https://docs.scipy.org/doc/numpy/reference/generated/numpy.einsum.html) (has lots of great examples)
- [The Matrix Calculus You Need For Deep Learning](https://explained.ai/matrix-calculus/index.html) by Terence Parr and Jeremy
- [Detexify](http://detexify.kirelabs.org/classify.html) (for finding math symbols) and [Wikipedia list of symbols](https://en.wikipedia.org/wiki/List_of_mathematical_symbols)
- The [matrix multiplication song](https://forums.fast.ai/uploads/default/original/3X/3/c/3cf0495ab3abefe0ad89fe6fbd9f101f7c507c4d.jpeg)

## Papers
- [Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a.html) -- paper that introduced Xavier initialization
- [Fixup Initialization: Residual Learning Without Normalization](https://arxiv.org/abs/1901.09321) -- paper highlighting importance of normalisation - training 10,000 layer network without normalization

## Other helpful resources

- [Mathpix](https://mathpix.com/) - turns images into LaTeX
- [Tutorial](https://towardsdatascience.com/learn-enough-python-to-be-useful-part-2-34f0e9e3fc9d) by @jeffhale on "How to use `if __name__=='__main__'`"
- [Khan academy lesson on the chain rule](https://www.khanacademy.org/math/ap-calculus-ab/ab-differentiation-2-new#ab-3-1a)
- [Basic PyTorch Tensor Tutorial (Includes Jupyter Notebook)](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html)
- [Xavier Initialisation (why divide over sqrt(M))](https://prateekvjoshi.com/2016/03/29/understanding-xavier-initialization-in-deep-neural-networks/) - Link to a blog post that explains it nicely
- [Xavier and Kaiming initialisation](https://pouannes.github.io/blog/initialization/) : Link to a blog post that explains the two papers, and in particular the math in detail

## "Assigned" Homework
* Review concepts from Course 1 (lessons 1 - 7): Affine Functions & non-linearities; Parameters & activations; Random initialization & transfer learning; SGD Momentum Adam (not sure what this one is, Jeremy's head covers it in the video); Convolutions; Batch-norm; Dropout; Data augmentation; Weight decay; Res/dense blocks; Image classification and regression; Embeddings; Continuous & Categorical variables; Collaborative filtering; Language models; NLP classification; Segmentation; U-net; GANS
* Make sure you understand broadcasting (a useful resource is [Computation on Arrays: Broadcasting by Jake VanderPlas](https://jakevdp.github.io/PythonDataScienceHandbook/02.05-computation-on-arrays-broadcasting.html))
* Read section 2.2 in [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852)
* Try to replicate as much of the notebooks as you can without peeking; when you get stuck, peek at the lesson notebook, but then close it and try to do it yourself

## Errata

Sometimes, occasionally, shockingly, Jeremy makes mistakes. It is rumored that these mistakes were made in this lesson:

1. Jeremy claimed that these are equivalent:

```python
for i in range(ar):
    c[i] = (a[i].unsqueeze(-1)*b).sum(dim=0)
    c[i] = (a[i,None]*b).sum(dim=0)
```

But they're not. The 2nd one isn't indexing the second axis. So it should be:

```python
for i in range(ar):
    c[i] = (a[i,:,None]*b).sum(dim=0)
```

---

## Edit this page

To edit this page, [click here](https://github.com/fastai/course-v3/edit/master/files/dl-2019/notes/notes-2-8.md). This will take you to a edit window at GitHub where you can submit your suggested changes. They will automatically be turned in to a [pull request](https://help.github.com/articles/about-pull-requests/) which will be reviewed by an admin prior to publication.
