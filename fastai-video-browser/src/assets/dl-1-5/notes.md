## Lesson 5: Back propagation; Accelerated SGD; Neural net from scratch

In lesson 5 we put all the pieces of training together to understand exactly what is going on when we talk about *back propagation*. We'll use this knowledge to create and train a simple neural network from scratch.

We'll also see how we can look inside the weights of an embedding layer, to find out what our model has learned about our categorical variables. This will let us get some insights into which movies we should probably avoid at all costs&hellip;

Although embeddings are most widely known in the context of word embeddings for NLP, they are at least as important for categorical variables in general, such as for tabular data or collaborative filtering. They can even be used with non-neural models with great success.

