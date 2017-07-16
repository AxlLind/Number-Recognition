# Number-Recognition
Artificial Neural Network created from scratch to classify hand-written
numbers.

It's a three layer artificial neural network using gradient descent to train
the network and forward propagation to classify data. It's trained using the
[MNIST database](http://yann.lecun.com/exdb/mnist/) of hand written numbers. The dataset contains 60000 hand 
written numbers used for training and 10000 used for testing after training.

The network classifies about 84% of the testing set correctly with a
threshold of 0.7!

Instead of using a neural network library like tensorflow I opted to create
everything from scratch, including the Matrix class for linear algebra, to
learn more about how neural networks work.

