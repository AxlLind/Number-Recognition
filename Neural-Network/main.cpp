
#include <iostream>
#include "Matrix.h"
#include "NeuralNetwork.h"
#include "MNIST.h"

int main() {
    int batch_size = 20, num_batches = 1000;
    clock_t before = clock();

    auto training_labels = MNIST::Parse(MNIST::TrainingLabels, batch_size, num_batches);
    auto training_data   = MNIST::Parse(MNIST::TrainingData, batch_size, num_batches);
    auto test_labels     = MNIST::Parse(MNIST::TestLabels, 100, 1)[0];
    auto test_data       = MNIST::Parse(MNIST::TestData, 100, 1)[0];

    std::cout << "Time to parse MNIST data: " << (double)(clock() - before) / CLOCKS_PER_SEC << "s\n" << std::endl;

    NeuralNetwork brain(784, 30, 10, 0.5);

    int last_percent = 0, percent = 0;
    for (int i = 0; i < training_data.size(); ++i) {
        brain.train(training_data[i], training_labels[i]);
        percent = 100 * (i + 1) / training_data.size();
        if (percent > last_percent) {
            std::cout << "Training: " << (last_percent = percent) << "%\r" << std::flush;
        }
    }

    std::cout << "\nCost: " << brain.cost(test_data, test_labels).vectorLength() << "\n";
    std::cout << "\nTraining: " << (double)(clock() - before) / CLOCKS_PER_SEC << "s\n";
}
