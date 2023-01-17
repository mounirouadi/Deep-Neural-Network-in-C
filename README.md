# A Deep Neural Network Implementation in C
This is a simple neural network implemented in C language, trained and tested on the MNIST dataset of handwritten digits. The network is trained to recognize digits from 0 to 9. The implemented network is a multi-layer perceptron with one hidden layer. The code includes functions for training, testing and evaluating the accuracy of the model. It also includes functions to save and load the weights and biases of the model.
It includes forward_propagation, back_propagation, activation functions and cost functions.

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## Prerequisites
* A C compiler (e.g. GCC)
* The MNIST dataset in binary format (can be obtained from http://yann.lecun.com/exdb/mnist/)
## Installing
* Clone the repository to your local machine
* Place the MNIST dataset files (mnist_train_images.bin, mnist_train_labels.bin, mnist_test_images.bin, and mnist_test_labels.bin) in the project's root directory
* Compile the code using the following command:
' gcc -o main main.c '

