
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
    auto test_labels     = MNIST::ParseFull(MNIST::TestLabels);
    auto test_data       = MNIST::ParseFull(MNIST::TestData);

    std::cout << "Time to parse MNIST data: " << (double)(clock() - before) / CLOCKS_PER_SEC << "s\n" << std::endl;

    before = clock();

    NeuralNetwork NN(784, 30, 10, 0.2);
    long percent_size = training_data.size() / 100, offset = training_data.size() % percent_size;
    std::vector<int> indexes(training_data.size());
    for (int i = 0; i < indexes.size(); ++i) indexes[i] = i;

    int num_sets = 3;
    for (int s = 0; s < num_sets; ++s) {
        before = clock();
        std::random_shuffle(indexes.begin(), indexes.end());
        for (int i = 0; i < indexes.size(); ++i) {
            NN.train(training_data[ indexes[i] ], training_labels[ indexes[i] ]);
            if (i % percent_size == offset)
                std::cout << "Training: " << 100 * (i + 1) / training_data.size() << "%\r" << std::flush;
        }
        std::cout << "Training Time: " << (double)(clock() - before) / CLOCKS_PER_SEC << "s\n";
    }


    std::cout << "\nCost: " << NN.cost(test_data, test_labels).vectorLength() << "\n";
    std::cout << "Percent correct: " << NN.percentCorrect(test_data, test_labels, 0.7) << "%\n";
}

