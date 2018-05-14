#include <iostream>
#include "Matrix.h"
#include "NeuralNetwork.h"
#include "MNIST.h"

/* Program Parameters */
int num_batches = 500, batch_size = 60000 / num_batches, hidden_neurons = 20;
double learn_rate = 0.2, correct_threshold = 0.5;
std::string file_path = "../Data/network.state";
/* Program Parameters */

void example(const NeuralNetwork &NN, const Matrix<double> &data, const Matrix<double> &labels) {
    Matrix<double> example(1, data.cols());
    int index = rand() % data.rows();
    for (int i = 0; i < example.cols(); ++i) {
        example(0, i) = data(index, i);
        std::cout << (example(0, i) == 0 ? "--" : "##");
        if ((i + 1) % 28 == 0) std::cout << "\n";
    }

    Matrix<double> result = NN.evaluate(example);
    double max = result(0,0);
    int maxI = 0;
    for (int i = 1; i < result.cols(); ++i) {
        double tmp = result(0,i);
        if (max < tmp) {
            max = tmp;
            maxI = i;
        }
    }
    std::cout << "> Neural Network classification: " << maxI;
    std::cout << (labels(index, maxI) == 1 ? ", Correct! " : ", Incorrect! ") << "\n";
}

void test(const NeuralNetwork &NN, const Matrix<double> &data, const Matrix<double> &labels) {
    std::cout << "> Percent of test set correctly identified: ";
    std::cout << NN.percentCorrect(data, labels, correct_threshold) << "%\n";
}

void train(NeuralNetwork &NN, const std::vector<Matrix<double>> &data, const std::vector<Matrix<double>> &labels) {
    std::vector<int> indexes(num_batches);
    for (int i = 0; i < indexes.size(); ++i) indexes[i] = i;
    std::random_shuffle(indexes.begin(), indexes.end());

    int percent_size = data.size() / 100, percent = 0;
    for (int i = 0; i < data.size(); ++i) {
        NN.train(data[ indexes[i] ], labels[ indexes[i] ]);
        if (i % percent_size == 0)
            std::cout << "> Training: " << ++percent << "%\r" << std::flush;
    }
    std::cout << "> Training: 100%\n";
}

void read(NeuralNetwork &NN) {
    NN.readState(file_path);
    std::cout << "> Neural Network state successfully read from file!\n";
}

void save(const NeuralNetwork &NN) {
    NN.saveState(file_path);
    std::cout << "> Neural Network state successfully saved to file!\n";
}

void reset(NeuralNetwork &NN) {
    NN.reset();
    std::cout << "> Neural Network has been reset!\n";
}

int userInput() {
    int input;
    std::cin >> input;
    std::cout << "\n";
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    return std::cin.fail() ? -1 : input;
}

int main() {
    srand(time(NULL));
    clock_t before = clock();
    std::cout << "\n> Parsing MNIST data set.\n" << std::flush;
    auto training_labels = MNIST::Parse(MNIST::TrainingLabels, batch_size, num_batches);
    auto training_data   = MNIST::Parse(MNIST::TrainingData,   batch_size, num_batches);
    auto test_labels     = MNIST::ParseAll(MNIST::TestLabels);
    auto test_data       = MNIST::ParseAll(MNIST::TestData);
    std::cout << "> Time: " << double(clock() - before) / CLOCKS_PER_SEC << "s\n";

    NeuralNetwork NN(test_data.cols(), hidden_neurons, test_labels.cols(), learn_rate);
    while(1) {
        std::cout << "1: Example | 2: Test | 3: Train | 4: Read | 5: Reset | 6: Save | 7: Exit\n";
        std::cout << "Enter a number to choose an action: ";
        int input = userInput();

        before = clock();
        switch(input) {
            case 1: example(NN, test_data, test_labels);       break;
            case 2: test(NN, test_data, test_labels);          break;
            case 3: train(NN, training_data, training_labels); break;
            case 4: read(NN);                                  break;
            case 5: reset(NN);                                 break;
            case 6: save(NN);                                  break;
            case 7: std::cout << "> Neural Network exits.\n";
                return 0;
            default: std::cout << "> Not a valid choice.\n";
        }
        std::cout << "> Time: " << double(clock() - before) / CLOCKS_PER_SEC << "s\n";
    }
}
