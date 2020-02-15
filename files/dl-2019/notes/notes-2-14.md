# Lesson 14: C interop; Protocols; Putting it all together

Today's lesson starts with a discussion of the ways that Swift programmers will be able to write high performance GPU code in plain Swift. Chris Lattner discusses kernel fusion, XLA, and MLIR, which are exciting technologies coming soon to Swift programmers.

Then Jeremy talks about something that's available right now: amazingly great C interop. He shows how to use this to quickly and easily get high performance code by interfacing with existing C libraries, using Sox audio processing, and VIPS and OpenCV image processing as complete working examples.

Next up, we implement the Data Block API in Swift! Well... actually in some ways it's even *better* than the original Python version. We take advantage of an enormously powerful Swift feature: *protocols* (aka *type classes*).

We now have enough Swift knowledge to implement a complete fully connect network forward pass in Swift&mdash;so that's what we do! Then we start looking at the backward pass, and use Swift's optional *reference semantics* to replicate the PyTorch approach. But then we learn how to do the same thing in a more "Swifty" way, using *value semantics* to do the backward pass in a really concise and flexible manner.

Finally, we put it all together, implementing our generic optimizer, Learner, callbacks, etc, to train Imagenette from scratch! The final notebooks in Swift show how to build and use much of the fastai.vision library in Swift, even although in these two lessons there wasn't time to cover everything. So be sure to study the notebooks to see lots more Swift tricks...

## Lesson resources

- [Slides for the lesson - from slide 41 ](https://docs.google.com/presentation/d/1dc6o2o-uYGnJeCeyvgsgyk05dBMneArxdICW5vF75oU/edit#slide=id.p)
- [Lesson Notebooks](https://github.com/fastai/course-v3/tree/master/nbs/swift)
- [Lesson Notes](https://medium.com/@lankinen/fast-ai-lesson-14-notes-part-2-v3-be4667394295) @Lankinen

## Links

- [Skip the FFI: Embedding Clang for C Interoperability](http://llvm.org/devmtg/2014-10/#talk18)
- [Value Semantics](https://academy.realm.io/posts/swift-gallagher-value-semantics/) talk by @AlexisGallagher
- [Tensor Comprehensions: Framework-Agnostic
High-Performance Machine Learning Abstractions](https://arxiv.org/pdf/1802.04730.pdf)

## Edit this page

To edit this page, [click here](https://github.com/fastai/course-v3/edit/master/files/dl-2019/notes/notes-2-14.md). This will take you to a edit window at GitHub where you can submit your suggested changes. They will automatically be turned in to a [pull request](https://help.github.com/articles/about-pull-requests/) which will be reviewed by an admin prior to publication.
