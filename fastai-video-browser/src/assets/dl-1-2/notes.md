## Lesson 2: Data cleaning and production; SGD from scratch

We start today's lesson learning how to build your own image classification model using your own data, including topics such as:

- Image collection
- Parallel downloading
- Creating a validation set, and
- Data cleaning, using the model to help us find data problems.

I'll demonstrate all these steps as I create a model that can take on the vital task of differentiating teddy bears from grizzly bears. Once we've got our data set in order, we'll then learn how to productionize our teddy-finder, and make it available online.

We've had some great additions since this lesson was recorded, so be sure to check out:

- The *production starter kits* on the course web site, such as [this one](https://course-v3.fast.ai/deployment_render.html) for deploying to Render.com
- The new interactive GUI in the lesson notebook for using the model to find and fix mislabeled or incorrectly-collected images.

In the second half of the lesson we'll train a simple model from scratch, creating our own *gradient descent* loop. In the process, we'll be learning lots of new jargon, so be sure you've got a good place to take notes, since we'll be referring to this new terminology throughout the course (and there will be lots more introduced in every lesson from here on).

