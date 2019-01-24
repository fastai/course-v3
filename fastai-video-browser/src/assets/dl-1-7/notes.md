## Lesson 7: Resnets from scratch; U-net; Generative (adversarial) networks

In the final lesson of Practical Deep Learning for Coders we'll study one of the most important techniques in modern architectures: the *skip connection*. This is most famously used in the *resnet*, which is the architecture we've used throughout this course for image classification, and appears in many cutting edge results. We'll also look at the *U-net* architecture, which uses a different type of skip connection to greatly improve segmentation results (and also for similar tasks where the output structure is similar to the input).

We'll then use the U-net architecture to train a *super-resolution* model. This is a model which can increase the resolution of a low-quality image. Our model won't only increase resolution&mdash;it will also remove jpeg artifacts, and remove unwanted text watermarks.

In order to make our model produce high quality results, we will need to create a custom loss function which incorporates *feature loss* (also known as *perceptual loss*), along with *gram loss*. These techniques can be used for many other types of image generation task, such as image colorization.

Finally, we'll learn about a recent loss function known as *generative adversarial* loss (used in generative adversarial networks, or *GANs*), which can improve the quality of generative models in some contexts, at the cost of speed.

The techniques we show in this lesson include some unpublished research that:

- Let us train GANs more quickly and reliably than standard approaches, by leveraging transfer learning
- Combines architectural innovations and loss function approaches that haven't been used in this way before.

The results are stunning, and train in just a couple of hours (compared to previous approaches that take a couple of days).

