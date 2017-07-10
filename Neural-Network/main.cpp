
#include <iostream>
#include "Matrix.h"
#include "NeuralNetwork.h"
#include "MNISTParser.h"

int main()
{
    clock_t before = clock();
    Matrix training_labels = MNISTParser::Parse(MNISTParser::TrainingLabels);
    Matrix training_data   = MNISTParser::Parse(MNISTParser::TrainingData);
    Matrix test_labels     = MNISTParser::Parse(MNISTParser::TestLabels);
    Matrix test_data       = MNISTParser::Parse(MNISTParser::TestData);
    std::cout << "Time to parse MNIST data: " << (double)(clock() - before) / CLOCKS_PER_SEC << "s\n" << std::endl;

    NeuralNetwork brain(784, 30, 10, 0.2);
    int steps = 50;
    for (int i = 0; i < steps; ++i) {
        brain.train(training_data, training_labels);
        std::cout << "Training: " << 100 * (i+1)/steps << "%\r" << std::flush;
    }

    std::cout << "\nCost: " << brain.cost(test_data, test_labels).vectorLength() << "\n\n";
    std::cout << "Training: " << (double)(clock() - before) / CLOCKS_PER_SEC << "s\n";
}