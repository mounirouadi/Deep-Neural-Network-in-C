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

* Run the program using the following command:
Copy code
`./main`
## Saving and Loading the Weights and Biases
The program includes functions to save and load the weights and biases of the trained network.

* To save the weights and biases, you can call the function save_weights_biases(char* file_name) and pass the desired file name as a string to the function.
For example:
`save_weights_biases("weights_biases.bin");`
This will save the weights and biases to the file "weights_biases.bin" in the same directory as the program.

* To load the weights and biases, you can call the function load_weights_biases(char* file_name) and pass the file name as a string to the function.
For example:
`load_weights_biases("weights_biases.bin");`

## Running the tests
The program will train the neural network on the MNIST training dataset and test it on the MNIST test dataset. The program will print the prediction for each test image.

## To Do
* Use vectorization techniques for matrice operations.
* Use multithreading to optimize the runtime.
* Migrate to C++ for OOP support.

## Authors
OUADI Mounir - [@mounirouadi](https://github.com/mounirouadi/)
## License
This project is licensed under the MIT License - see the LICENSE.md file for details.

