# Lesson 6: Regularization; Convolutions; Data ethics

## Overview

Today we discuss some powerful techniques for improving training and avoiding over-fitting:

- **Dropout**: remove activations at random during training in order to regularize the model
- **Data augmentation**: modify model inputs during training in order to effectively increase data size
- **Batch normalization**: adjust the parameterization of a model in order to make the loss surface smoother.

Next up, we'll learn all about *convolutions*, which can be thought of as a variant of matrix multiplication with tied weights, and are the operation at the heart of modern computer vision models (and, increasingly, other types of models too).

We'll use this knowledge to create a *class activated map*, which is a heat-map that shows which parts of an image were most important in making a prediction.

Finally, we'll cover a topic that many students have told us is the most interesting and surprising part of the course: data ethics. We'll learn about some of the ways in which models can go wrong, with a particular focus on *feedback loops*, why they cause problems, and how to avoid them. We'll also look at ways in which bias in data can lead to biased algorithms, and discuss questions that data scientists can and should be asking to help ensure that their work doesn't lead to unexpected negative outcomes.

## Resources

- [platform.ai discussion](https://forums.fast.ai/t/platform-ai-discussion/31445)
- [50 Years of Test (Un)fairness: Lessons for Machine Learning](https://128.84.21.199/pdf/1811.10104.pdf)
