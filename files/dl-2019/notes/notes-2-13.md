# Lesson 13: Basics of Swift for Deep Learning

- [Lesson notebooks](https://github.com/fastai/course-v3/tree/master/nbs/swift) **NB: this is a different folder to the Python lesson notebooks**
- [Slides for the lesson](https://docs.google.com/presentation/d/1dc6o2o-uYGnJeCeyvgsgyk05dBMneArxdICW5vF75oU/edit#slide=id.p)
- [Lesson Notes](https://medium.com/@lankinen/fast-ai-lesson-13-notes-part-2-v3-2d62ef11d2db) @Lankinen

## Overview

We've just completed building much of the fastai library for Python from scratch. Now we're going to try to repeat the process for Swift! These next two lessons are co-taught by Jeremy along with Chris Lattner, the original developer of Swift, and the lead of the Swift for TensorFlow project at Google Brain.

In today's lesson, Chris explains what Swift is, and what it's designed to do. He shares insights on its development history, and why he thinks it's a great fit for deep learning and numeric programming more generally. He also provides some background on how Swift and TensorFlow fit together, both now and in the future. Next up, Chris shows a bit about using types to ensure your code has less errors, whilst letting Swift figure out most of your types for you. And he explains some of the key pieces of syntax we'll need to get started.

Chris also explains what a compiler is, and how LLVM makes compiler development easier. Then he shows how we can actually access and change LLVM builtin types directly from Swift! Thanks to the compilation and language design, basic code runs very fast indeed - about 8000 times faster than Python in the simple example Chris showed in class.

Finally, we looked at different ways of calculating matrix products in Swift, including using Swift for TensorFlow's `Tensor` class.

## Software requirements

- You need to install swift for TensorFlow and swift-jupyter, see Jeremy's [install guide](https://forums.fast.ai/t/jeremys-harebrained-install-guide/43814).
- [s4tf download](https://github.com/tensorflow/swift/blob/master/Installation.md)
- For help installing s4tf, please ask on the above thread. *Don't* ask install questions in this lesson discussion thread please!

## Swift resources
- The [swift book](https://docs.swift.org/swift-book/)
- [A swift tour](https://docs.swift.org/swift-book/GuidedTour/GuidedTour.html) (download in playground on an iPad or a Mac if you can).
- The [harebrain](https://forums.fast.ai/c/harebrain)  forum category. This is where to ask your S4TF questions.

## Other resources

- S4TF [notebook tutorials](https://github.com/tensorflow/swift#tutorials-) (rather out of date at the moment)
- Why [fastai is embracing S4TF](https://www.fast.ai/2019/03/06/fastai-swift/)?

## Edit this page

To edit this page, [click here](https://github.com/fastai/course-v3/edit/master/files/dl-2019/notes/notes-2-13.md). This will take you to a edit window at GitHub where you can submit your suggested changes. They will automatically be turned in to a [pull request](https://help.github.com/articles/about-pull-requests/) which will be reviewed by an admin prior to publication.
