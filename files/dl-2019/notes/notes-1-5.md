# Lesson 5: Back propagation; Accelerated SGD; Neural net from scratch

## Overview
In lesson 5 we put all the pieces of training together to understand exactly what is going on when we talk about *back propagation*. We'll use this knowledge to create and train a simple neural network from scratch.

We'll also see how we can look inside the weights of an embedding layer, to find out what our model has learned about our categorical variables. This will let us get some insights into which movies we should probably avoid at all costs&hellip;

Although embeddings are most widely known in the context of word embeddings for NLP, they are at least as important for categorical variables in general, such as for tabular data or collaborative filtering. They can even be used with non-neural models with great success.

## Resources

### Lesson resources

- [Lesson notes](https://forums.fast.ai/t/deep-learning-lesson-5-notes/31298) - thanks to @PoonamV
- [Detailed lesson notes](https://github.com/hiromis/notes/blob/master/Lesson5.md) - thanks to @hiromi
- Notebooks:
  - [lesson5-sgd-mnist.ipynb](https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson5-sgd-mnist.ipynb)
- Excel spreadsheets:
  - [collab_filter.xlsx](https://github.com/fastai/course-v3/blob/master/files/xl/collab_filter.xlsx);
[Google Sheets full version](https://docs.google.com/spreadsheets/d/1oxY9bxgLPutRidhTrucFeg5Il0Jq7UdMJgR3igTtbPU/edit#gid=1748360111); To run solver, please use Google Sheets short-cut version and follow [instruction](https://forums.fast.ai/t/google-sheets-versions-of-spreadsheets/10424/7) by @Moody
  - graddesc: [Excel](https://github.com/fastai/course-v3/blob/master/files/xl/graddesc.xlsm) version ; [Google sheets](https://docs.google.com/spreadsheets/d/1uUwjwDgTvsxW7L1uPzpulGlUTaLOm8b-R_v0HIUmAvY/edit?usp=sharing) version
  - [entropy_example.xlsx](https://github.com/fastai/course-v3/blob/master/files/xl/entropy_example.xlsx)
- [Lesson 5 in-class discussion thread](https://forums.fast.ai/t/lesson-5-discussion-thread/30864)
- [Lesson 5 advanced discussion](https://forums.fast.ai/t/lesson-5-further-discussion/30865)
- [Links to different parts in video](https://forums.fast.ai/t/lesson-5-links-to-different-parts-in-video/30891) by @melonkernel

### Other resources

- [NY Times Article - Finally, a Machine That Can Finish Your Sentence](https://www.nytimes.com/2018/11/18/technology/artificial-intelligence-language.html)
- [Netflix and Chill: Building a Recommendation System in Excel - Latent Factor Visualization in Excel blog post](https://towardsdatascience.com/netflix-and-chill-building-a-recommendation-system-in-excel-c69b33c914f4)
- [An overview of gradient descent optimization algorithms - Sebastian Ruder](http://ruder.io/optimizing-gradient-descent/)

---

[Edit this page](https://github.com/fastai/course-v3/edit/master/files/dl-2019/notes/notes-1-5.md).
