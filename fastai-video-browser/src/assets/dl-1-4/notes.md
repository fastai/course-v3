## Lesson 4: NLP; Tabular data; Collaborative filtering; Embeddings

In lesson 4 we'll dive in to *natural language processing* (NLP), using the IMDb movie review dataset. In this task, our goal is to predict whether a movie review is positive or negative; this is called *sentiment analysis*. We'll be using the [ULMFiT](https://arxiv.org/abs/1801.06146) algorithm, which was originally developed during the fast.ai 2018 course, and became part of a revolution in NLP during 2018 which led the New York Times to declare that [new systems are starting to crack the code of natural language](https://www.nytimes.com/2018/11/18/technology/artificial-intelligence-language.html). ULMFiT is today the most accurate known sentiment analysis algorithm.

The basic steps are:

1. Create (or, preferred, download a pre-trained) *language model* trained on a large corpus such as Wikipedia (a "language model" is any model that learns to predict the next word of a sentence)
1. Fine-tune this language model using your *target corpus* (in this case, IMDb movie reviews)
1. Remove the *encoder* in this fine tuned language model, and replace it with a *classifier*. Then fine-tune this model for the final classification task (in this case, sentiment analysis).

After our journey into NLP, we'll complete our practical applications for Practical Deep Learning for Coders by covering tabular data (such as spreadsheets and database tables), and collaborative filtering (recommendation systems).

For tabular data, we'll see how to use *categorical* and *continuous* variables, and how to work with the *fastai.tabular* module to set up and train a model.

Then we'll see how collaborative filtering models can be built using similar ideas to those for tabular data, but with some special tricks to get both higher accuracy and more informative model interpretation. 

This brings us to the half-way point of the course, where we have looked at how to build and interpret models in each of these key application areas:

- Computer vision
- NLP
- Tabular
- Collaborative filtering

For the second half of the course, we'll learn about *how* these models really work, and how to create them ourselves from scratch. For this lesson, we'll put together some of the key pieces we've touched on so far:

- Activations
- Parameters
- Layers (affine and non-linear)
- Loss function.

We'll be coming back to each of these in lots more detail during the remaining lessons. We'll also learn about a type of layer that is important for NLP, collaborative filtering, and tabular models: the *embedding layer*. As we'll discover, an "embedding" is simply a computational shortcut for a particular type of matrix multiplication (a multiplication by a *one-hot encoded* matrix).


