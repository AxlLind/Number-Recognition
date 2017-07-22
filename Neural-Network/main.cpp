
#include <iostream>
#include "Matrix.h"
#include "NeuralNetwork.h"
#include "MNIST.h"

/* Program Parameters */
int num_batches = 500, batch_size = 60000 / num_batches, num_hidden_neurons = 20;
double learn_rate = 0.2, correct_threshold = 0.7;
std::string file_path = "../Data/network.state";
/* Program Parameters */

void optionExample(const NeuralNetwork &NN, const Matrix &test_data) {
    auto before = clock();
    Matrix Sample(1, test_data.cols);
    int index = rand() % test_data.rows, maxI = -1;
    for (int j = 0; j < Sample.cols; ++j)
        Sample(0, j, test_data(index, j));

    for (int i = 0; i < Sample.cols; ++i)
        std::cout << (Sample(0,i) == 0 ? "--" : "##") << ((i+1) % 28 == 0 ? "\n" : "");

    Matrix result = NN.evaluate(Sample);
    double max = -1;
    for (int i = 0; i < result.cols; ++i) {
        double tmp = result(0,i);
        if (max < tmp) {
            max = tmp;
            maxI = i;
        }
    }
    std::cout << "> Neural Network classification: " << maxI << "\n";
    std::cout << "> Time: " << (double)(clock() - before) / CLOCKS_PER_SEC << "s";
}

void optionEvaluate(const NeuralNetwork &NN, const Matrix &test_data, const Matrix &test_labels) {
    auto before = clock();
    double percent_correct = NN.percentCorrect(test_data, test_labels, correct_threshold);
    std::cout << "> Percent of test set correctly identified: " << percent_correct << "%\n";
    std::cout << "> Time: " << (double)(clock() - before) / CLOCKS_PER_SEC << "s";
}

void optionTrain(NeuralNetwork &NN, const std::vector<Matrix> &training_data, const std::vector<Matrix> &training_labels) {
    auto before = clock();
    std::vector<int> indexes(num_batches);
    for (int i = 0; i < indexes.size(); ++i) indexes[i] = i;
    std::random_shuffle(indexes.begin(), indexes.end());

    int percent_size = training_data.size() / 100;
    for (int i = 0; i < training_data.size(); ++i) {
        NN.train(training_data[ indexes[i] ], training_labels[ indexes[i] ]);
        if (i % percent_size == 0) std::cout << "> Training: " << 100 * i / training_data.size() << "%\r" << std::flush;
    }
    std::cout << "> Training: 100%\n> Time: " << (double)(clock() - before) / CLOCKS_PER_SEC << "s";
}

void optionRead(NeuralNetwork &NN) {
    auto before = clock();
    NN.readState(file_path);
    std::cout << "> Neural Network state successfully read from file!\n";
    std::cout << "> Time: " << (double)(clock() - before) / CLOCKS_PER_SEC << "s";
}

void optionSave(const NeuralNetwork &NN) {
    auto before = clock();
    NN.saveState(file_path);
    std::cout << "> Neural Network state successfully saved to file!\n";
    std::cout << "> Time: " << (double)(clock() - before) / CLOCKS_PER_SEC << "s";
}

void optionReset(NeuralNetwork &NN) {
    auto before = clock();
    NN.reset();
    std::cout << "> Neural Network has been reset!\n";
    std::cout << "> Time: " << (double)(clock() - before) / CLOCKS_PER_SEC << "s";
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

    std::cout << "\nParsing MNIST data set...\r" << std::flush;
    clock_t before = clock();
    auto training_labels = MNIST::Parse(MNIST::TrainingLabels, batch_size, num_batches);
    auto training_data   = MNIST::Parse(MNIST::TrainingData,   batch_size, num_batches);
    auto test_labels     = MNIST::ParseAll(MNIST::TestLabels);
    auto test_data       = MNIST::ParseAll(MNIST::TestData);
    std::cout << "Time to parse MNIST data: " << (double)(clock() - before) / CLOCKS_PER_SEC << "s";

    NeuralNetwork NN(test_data.cols, num_hidden_neurons, test_labels.cols, learn_rate);
    while(1) {
        std::cout << "\n\n1: Example | 2: Evaluate | 3: Train | 4: Read | 5: Save | 6: Reset | 7: Exit\n";
        std::cout << "Enter a number to choose an action: ";
        switch(userInput()) {
            case 1:
                optionExample(NN, test_data);
                break;
            case 2:
                optionEvaluate(NN, test_data, test_labels);
                break;
            case 3:
                optionTrain(NN, training_data, training_labels);
                break;
            case 4:
                optionRead(NN);
                break;
            case 5:
                optionSave(NN);
                break;
            case 6:
                optionReset(NN);
                break;
            case 7:
                std::cout << "> Neural Network exits.";
                return 0;

            default: std::cout << "> Not a valid choice.";
        }
    }
}
