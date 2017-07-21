
#include <iostream>
#include "Matrix.h"
#include "NeuralNetwork.h"
#include "MNIST.h"

int main() {
    srand(time(NULL));
    int num_batches = 500, batch_size = 60000 / num_batches;

    clock_t before = clock();

    auto training_labels = MNIST::Parse(MNIST::TrainingLabels, batch_size, num_batches);
    auto training_data   = MNIST::Parse(MNIST::TrainingData,   batch_size, num_batches);
    auto test_labels     = MNIST::ParseAll(MNIST::TestLabels);
    auto test_data       = MNIST::ParseAll(MNIST::TestData);

    std::cout << "Time to parse MNIST data: " << (double)(clock() - before) / CLOCKS_PER_SEC << "s\n" << std::endl;

    NeuralNetwork NN(test_data.cols, 20, 10, 0.2);
    std::vector<int> indexes(training_data.size());
    for (int i = 0; i < indexes.size(); ++i) indexes[i] = i;

    int percent_size = training_data.size() / 100, epochs = 1;
    for (int e = 0; e < epochs; ++e) {
        before = clock();
        std::random_shuffle(indexes.begin(), indexes.end());
        for (int i = 0; i < indexes.size(); ++i) {
            NN.train(training_data[ indexes[i] ], training_labels[ indexes[i] ]);
            if (i % percent_size == 0)
                std::cout << "Training " << e+1 << "/" << epochs << ": "<< 100 * i / training_data.size() << "%\r" << std::flush;
        }
        std::cout << "Training Time: " << (double)(clock() - before) / CLOCKS_PER_SEC << "s\n";
    }

    std::cout << "\nCost: " << NN.cost(test_data, test_labels).vectorLength() << "\n";
    std::cout << "Percent correct: " << NN.percentCorrect(test_data, test_labels, 0.7) << "%\n";
}
