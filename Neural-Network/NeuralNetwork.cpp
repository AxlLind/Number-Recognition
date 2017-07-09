//
// Created by Axel Lindeberg on 2017-07-01.
//

#include <iostream>
#include "Matrix.h"
#include "NeuralNetwork.h"
#include "MNISTReader.h"

int main()
{
    clock_t before = clock();
    Matrix training_labels = MNISTReader::Parse(MNISTReader::TrainingLabels);
    Matrix training_data   = MNISTReader::Parse(MNISTReader::TrainingData);
    Matrix test_labels     = MNISTReader::Parse(MNISTReader::TestLabels);
    Matrix test_data       = MNISTReader::Parse(MNISTReader::TestData);
    std::cout << "Time to read MNIST data: " << (double)(clock() - before) / CLOCKS_PER_SEC << "s\n";

    NeuralNetwork brain(784, 30, 10, 0.2);
    for (int i = 0; i < 50; ++i) {
        clock_t begin = clock();
        brain.train(training_data, training_labels);
        std::cout << (double)(clock() - before) / CLOCKS_PER_SEC << "s ";
    }

    std::cout << "\n\n" << brain.cost(test_data, test_labels);
}