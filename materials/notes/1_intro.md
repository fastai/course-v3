Lecture 01 note

- Pandemic, Corona Virus
- [Deep Learning for Coders with fastai and PyTorch: AI Applications without a PhD](https://github.com/fastai/fastbook)
- About Deep Learning Myth
	- 	Math: Just hight school math is sufficient
	-  Data: We've seen record breaking results with < 50 items of data
	-  Computing resource: You can get what you need for state of the art work for free

- NLP: Answering Questions, Speech Recognition, Summarising Documents, Classifying Documents, Finding names, dates, etc. in documents, searching for articles mentioning a concept	
- Neural Networks: a brief history
	- Minsky and AI winter.
- Parallel Distributed Processing(PDP)
- The age of deep learning is here
	- In theory, adding just one extra layer of neurons was enough, but in practice such nets were often too big and slow.
- For most people 1. play the whole game 2. make the game worth playing 3. Work on the hard parts
	- 	David Perkins
	-  Make people what's going on this game and try to understand them

- The software : Pytorch, fastai, jupyter notebook
	- PyTorch is super flexible.
	- Fastai is On the top of PyTorch not only for beginners, but also researches.
- Getting a GPU deep learning server
	- recommended to use NVIDIA GPU
	- rent GPU do not buy ([referred site](https://course.fast.ai/) - server setup)
	- Please ask of your server setup before ask..(someone would have asked it)

- Jupyter notebook explaination

- Because there is random variations, don't expect to get the exact same value.
- Don't use mac's GPU(sticking with NVIDIA will make your life easier)

- What is ML?
	- Let the program(model) solve the problem itself, instead of order computer the each specific steps.

- Terms
	- model : architecture,
	- weights : parameters
	- predictions : value which was calculated from independent values
	- results of models are called predictions
	- and so on

- 	Limitations of DNN
	-  we need data
	-  can only learn to operate on the patterns seen in the input data used to train it
	-  only creates predictions, not recommended actions
	-  we need not only data, but also label for the predictions

- Consider how a model interacts with its environments
	- can creates feedback loops
	- Think about how the model used and how it affects to the train data(and it can magnify bias)

- about import *
	- If you feel inconvenient with import *.. it's okay!
	- Because fast.ai is designed to avoid the problem from importing star!
	- But if you can't track where the method comes from, just type it! and it shows the source
	- and also use doc(method)
	- plus, all the document was written using jupyter notebook. further information: [nbdev](https://www.fast.ai/2019/12/02/nbdev/)

- Proper fitting vs Overfitting
	- The only way to see if it's overfitting or not is to watch on your model works well with unseen data
- fastai incorporates ideas from Lisp, APL, math and many other languages and notations, and uses approached customised to data science. 
- What we should do?
	- Install GPU
	- Don't move on
	- read the chapter1 of the book
	- (recommended) read the document of fastai
- Next time
	- Deploy the model