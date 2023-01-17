#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define INPUT_NODES 784  // 28*28 pixels
#define HIDDEN_NODES 256 // Number of hidden nodes
#define OUTPUT_NODES 10  // 10 digits (0-9)

#define NUM_TRAINING_IMAGES 60000
#define NUM_TEST_IMAGES 10000

#define NUMBER_OF_EPOCHS 10

double training_images[NUM_TRAINING_IMAGES][INPUT_NODES];
double training_labels[NUM_TRAINING_IMAGES][OUTPUT_NODES];
double test_images[NUM_TEST_IMAGES][INPUT_NODES];
double test_labels[NUM_TEST_IMAGES][OUTPUT_NODES];

// Initialize weights and biases
double weight1[INPUT_NODES][HIDDEN_NODES];
double weight2[HIDDEN_NODES][OUTPUT_NODES];
double bias1[HIDDEN_NODES];
double bias2[OUTPUT_NODES];

 int correct_predictions;
 int forward_prob_output;

void load_mnist()
{
    // Open the training images file
    FILE *training_images_file = fopen("mnist_train_images.bin", "rb");
    if (training_images_file == NULL)
    {
        printf("Error opening training images file\n");
        exit(1);
    }

    // Open the training labels file
    FILE *training_labels_file = fopen("mnist_train_labels.bin", "rb");
    if (training_labels_file == NULL)
    {
        printf("Error opening training labels file\n");
        exit(1);
    }

    // Open the test images file
    FILE *test_images_file = fopen("mnist_test_images.bin", "rb");
    if (test_images_file == NULL)
    {
        printf("Error opening test images file\n");
        exit(1);
    }

    // Open the test labels file
    FILE *test_labels_file = fopen("mnist_test_labels.bin", "rb");
    if (test_labels_file == NULL)
    {
        printf("Error opening test labels file\n");
        exit(1);
    }

    // Read the training images
    for (int i = 0; i < NUM_TRAINING_IMAGES; i++)
    {
        for (int j = 0; j < INPUT_NODES; j++)
        {
            unsigned char pixel;
            fread(&pixel, sizeof(unsigned char), 1, training_images_file);
            training_images[i][j] = (double)pixel / 255.0;
        }
    }

    // Read the training labels
    for (int i = 0; i < NUM_TRAINING_IMAGES; i++)
    {
        unsigned char label;
        fread(&label, sizeof(unsigned char), 1, training_labels_file);
        for (int j = 0; j < OUTPUT_NODES; j++)
        {
            if (j == label)
            {
                training_labels[i][j] = 1;
            }
            else
            {
                training_labels[i][j] = 0;
            }
        }
    }

    // Read the test images
    for (int i = 0; i < NUM_TEST_IMAGES; i++)
    {
        for (int j = 0; j < INPUT_NODES; j++)
        {
            unsigned char pixel;
            fread(&pixel, sizeof(unsigned char), 1, test_images_file);
            test_images[i][j] = (double)pixel / 255.0;
        }
    }

    // Read the test labels
    for (int i = 0; i < NUM_TEST_IMAGES; i++)
    {
        unsigned char label;
        fread(&label, sizeof(unsigned char), 1, test_labels_file);
        for (int j = 0; j < OUTPUT_NODES; j++)
        {
            if (j == label)
            {
                test_labels[i][j] = 1;
            }
            else
            {
                test_labels[i][j] = 0;
            }
        }
    }

    // Close the files
    fclose(training_images_file);
    fclose(training_labels_file);
    fclose(test_images_file);
    fclose(test_labels_file);
}

double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}
int max_index(double arr[], int size) {
    int max_i = 0;
    for (int i = 1; i < size; i++) {
        if (arr[i] > arr[max_i]) {
            max_i = i;
        }
    }
    return max_i;
}

void train(double input[INPUT_NODES], double output[OUTPUT_NODES], double weight1[INPUT_NODES][HIDDEN_NODES], double weight2[HIDDEN_NODES][OUTPUT_NODES], double bias1[HIDDEN_NODES], double bias2[OUTPUT_NODES], int correct_label)
{
    double hidden[HIDDEN_NODES];
    double output_layer[OUTPUT_NODES];

    // Feedforward
    for (int i = 0; i < HIDDEN_NODES; i++)
    {
        double sum = 0;
        for (int j = 0; j < INPUT_NODES; j++)
        {
            sum += input[j] * weight1[j][i];
        }
        sum += bias1[i];
        hidden[i] = sigmoid(sum);
    }
    for (int i = 0; i < OUTPUT_NODES; i++)
    {
        double sum = 0;
        for (int j = 0; j < HIDDEN_NODES; j++)
        {
            sum += hidden[j] * weight2[j][i];
        }
        sum += bias2[i];
        output_layer[i] = sigmoid(sum);
    }

    int index = max_index(output_layer, OUTPUT_NODES);

    if (index == correct_label) {
        forward_prob_output++;
    }
    

    // Backpropagation
    double error[OUTPUT_NODES];
    for (int i = 0; i < OUTPUT_NODES; i++)
    {
        error[i] = output[i] - output_layer[i];
    }
    double delta2[HIDDEN_NODES];
    for (int i = 0; i < HIDDEN_NODES; i++)
    {
        delta2[i] = 0;
        for (int j = 0; j < OUTPUT_NODES; j++)
        {
            delta2[i] += error[j] * output_layer[j] * (1 - output_layer[j]) * weight2[i][j];
        }
    }
    double delta1[INPUT_NODES];
    for (int i = 0; i < INPUT_NODES; i++)
    {
        delta1[i] = 0;
        for (int j = 0; j < HIDDEN_NODES; j++)
        {
            delta1[i] += delta2[j] * hidden[j] * (1 - hidden[j]) * weight1[i][j];
        }
    }

    // Update weights and biases
    double learning_rate = 0.1;
    for (int i = 0; i < INPUT_NODES; i++)
    {
        for (int j = 0; j < HIDDEN_NODES; j++)
        {
            weight1[i][j] += learning_rate * delta1[i] * input[j];
        }
    }
    for (int i = 0; i < HIDDEN_NODES; i++)
    {
        bias1[i] += learning_rate * delta2[i];
        for (int j = 0; j < OUTPUT_NODES; j++)
        {
            weight2[i][j] += learning_rate * error[j] * output_layer[j] * (1 - output_layer[j]) * hidden[i];
        }
    }
    for (int i = 0; i < OUTPUT_NODES; i++)
    {
        bias2[i] += learning_rate * error[i] * output_layer[i] * (1 - output_layer[i]);
    }
}

void test(double input[INPUT_NODES], double weight1[INPUT_NODES][HIDDEN_NODES], double weight2[HIDDEN_NODES][OUTPUT_NODES], double bias1[HIDDEN_NODES], double bias2[OUTPUT_NODES], int correct_label)
{
    double hidden[HIDDEN_NODES];
    double output_layer[OUTPUT_NODES];

    // Feedforward
    for (int i = 0; i < HIDDEN_NODES; i++)
    {
        double sum = 0;
        for (int j = 0; j < INPUT_NODES; j++)
        {
            sum += input[j] * weight1[j][i];
        }
        sum += bias1[i];
        hidden[i] = sigmoid(sum);
    }
    for (int i = 0; i < OUTPUT_NODES; i++)
    {
        double sum = 0;
        for (int j = 0; j < HIDDEN_NODES; j++)
        {
            sum += hidden[j] * weight2[j][i];
        }
        sum += bias2[i];
        output_layer[i] = sigmoid(sum);
    }
    int index = max_index(output_layer, OUTPUT_NODES);

    printf("Prediction: %d\n", index);
    if (index == correct_label) {
        correct_predictions++;
    }
}




void save_weights_biases(char* file_name) {
    FILE* file = fopen(file_name, "wb");
    if (file == NULL) {
        printf("Error opening file\n");
        exit(1);
    }
    fwrite(weight1, sizeof(double), HIDDEN_NODES * INPUT_NODES, file);
    fwrite(weight2, sizeof(double), HIDDEN_NODES * OUTPUT_NODES, file);
    fwrite(bias1, sizeof(double), HIDDEN_NODES, file);
    fwrite(bias2, sizeof(double), OUTPUT_NODES, file);
    fclose(file);
}

void load_weights_biases(char* file_name) {
    FILE* file = fopen(file_name, "rb");
    if (file == NULL) {
        printf("Error opening file\n");
        exit(1);
    }
    fread(weight1, sizeof(double), HIDDEN_NODES * INPUT_NODES, file);
    fread(weight2, sizeof(double), HIDDEN_NODES * OUTPUT_NODES, file);
    fread(bias1, sizeof(double), HIDDEN_NODES, file);
    fread(bias2, sizeof(double), OUTPUT_NODES, file);
    fclose(file);
}


int main()
{
    for (int i = 0; i < INPUT_NODES; i++)
    {
        for (int j = 0; j < HIDDEN_NODES; j++)
        {
            weight1[i][j] = (double)rand() / RAND_MAX;
        }
    }
    for (int i = 0; i < HIDDEN_NODES; i++)
    {
        bias1[i] = (double)rand() / RAND_MAX;
        for (int j = 0; j < OUTPUT_NODES; j++)
        {
            weight2[i][j] = (double)rand() / RAND_MAX;
        }
    }
    for (int i = 0; i < OUTPUT_NODES; i++)
    {
        bias2[i] = (double)rand() / RAND_MAX;
    }

    // Load MNIST dataset
    load_mnist();

    // Train the network
    for(int epoch=0;epoch<NUMBER_OF_EPOCHS;epoch++)
    {
        int forward_prob_output = 0;
        for (int i = 0; i < NUM_TRAINING_IMAGES; i++)
        {
            int correct_label = max_index(training_labels[i], OUTPUT_NODES);
            train(training_images[i], training_labels[i], weight1, weight2, bias1, bias2, correct_label);
        }
        printf("Epoch %d : Training Accuracy: %f\n",epoch , (double) forward_prob_output / NUM_TRAINING_IMAGES);
    }
    save_weights_biases("model.bin");

    // Test the network
    int correct_predictions = 0;
    for (int i = 0; i < NUM_TEST_IMAGES; i++)
    {
        int correct_label = max_index(test_labels[i], OUTPUT_NODES);
        test(test_images[i], weight1, weight2, bias1, bias2, correct_label);
    }
    printf("Testing Accuracy: %f\n", (double) correct_predictions / NUM_TEST_IMAGES);

    return 0;
}
