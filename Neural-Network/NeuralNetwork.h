
#ifndef NUMBER_RECOGNITION_NEURALNETWORK_H
#define NUMBER_RECOGNITION_NEURALNETWORK_H

#include <fstream>
#include "Matrix.h"

/**
 * Artificial neural network with a single hidden layer.
 * Uses forward propagation to classify/apply regression to input data.
 * Uses back-propagation and gradient descent to train the neural network.
 * Supports both single and batch gradient descent.
 * Activation function is the sigmoid function, might want to use ReLU instead.
 * Uses the Matrix class for input, output, and internally.
 *
 * Example:
 *
 *      NeuralNetwork NN(3,5,1)     // creates nn with 3 input-, 5 hidden-, and 1 output-neuron
 *      NN.train( data, labels )    // trains nn with gradient descent
 *      NN.evaluate( testing_data ) // classifies testing_data
 *
 * @author Axel Lindeberg
 * @date 2017-07-01
 */
class NeuralNetwork {
    const int num_in, num_hidden, num_out;
    const double learn_rate;
    Matrix W1, W2;

    /**
     * Activation function, currently the sigmoid function.
     *
     * @param derivative - if true returns derivative of activation function.
     */
    double activation(double x, bool derivative = false) const {
        double d = 1 / (1 + exp(-x));
        return derivative ? d * (1 - d) : d;
    }

    /**
     * Apply activation function element-wise to matrix A.
     *
     * @param derivative - if true applies derivative of activation function.
     */
    Matrix activation(const Matrix &A, bool derivative = false) const {
        Matrix OUT(A.rows, A.cols);
        for (int i = 0; i < OUT.rows; ++i) {
            for (int j = 0; j < OUT.cols; ++j)
                OUT(i, j, activation( A(i, j), derivative ));
        }
        return OUT;
    }

    Matrix normalizeRows(const Matrix &A) const {
        Matrix OUT(A.rows, A.cols);
        for (int i = 0; i < OUT.rows; ++i) {
            double total = 0;
            for (int j = 0; j < OUT.cols; ++j)
                total += A(i, j);

            if (total == 0) continue;

            for (int j = 0; j < OUT.cols; ++j)
                OUT(i, j, A(i, j) / total);
        }
        return OUT;
    }

    std::vector<Matrix> fullForward(const Matrix &A) const {
        if (A.cols != num_in)
            throw std::invalid_argument("NeuralNetwork::fullForward() - Input does not match neural network");

        Matrix Z2 = A * W1;
        Matrix A2 = activation(Z2);
        Matrix Z3 = A2 * W2;
        Matrix yHat = num_out == 1 ? activation(Z3) : normalizeRows(activation(Z3));

        return std::vector<Matrix>({Z2, A2, Z3, yHat});
    }

    std::vector<Matrix> costPrime(const Matrix &data, const Matrix &labels) const {
        if (labels.rows != data.rows)
            throw std::invalid_argument("NeuralNetwork::costPrime() - Input-data and label-data need to be of same size");

        auto matrices = fullForward(data); // {Z2, A2, Z3, yHat}

        Matrix delta3 = -1 * ( labels - matrices[3] ).scalarMulti( activation(matrices[2], true) );
        Matrix d_W2 = matrices[1].T() * delta3;

        Matrix delta2 = ( delta3 * W2.T() ).scalarMulti( activation(matrices[0], true) );
        Matrix d_W1 = data.T() * delta2;

        return std::vector<Matrix>({d_W1, d_W2});
    }

 public:
    NeuralNetwork(int numIn, int numHidden, int numOut, double learnRate) :
            W1(numIn, numHidden), W2(numHidden, numOut),
            num_in(numIn), num_hidden(numHidden), num_out(numOut), learn_rate(learnRate) {
        if (numIn < 1 || numHidden < 1 || numOut < 1 || learnRate <= 0)
            throw std::invalid_argument("NeuralNetwork::Constructor() - Invalid argument(s)");

        reset();
    }

    /** Evaluates the data using forward propagation through the network. */
    Matrix evaluate(const Matrix &data) const {
        if (data.cols != num_in)
            throw std::invalid_argument("NeuralNetwork::evaluate() - Input does not match neural network");

        Matrix OUT = activation(activation(data * W1) * W2);
        return num_out == 1 ? OUT : normalizeRows(OUT);
    }

    /**
     * Returns the percentage of datapoints that after evaluation
     * are classified correctly above the threshold.
     *
     * @param threshold If evaluation is above the threshold it's labeled correct
     * @return percentage of data correctly classified
     */
    double percentCorrect(const Matrix &data, const Matrix &labels, double threshold) const {
        if (data.rows != labels.rows)
            throw std::invalid_argument("NeuralNetwork::numCorrect() - Input-data and label-data need to be of same size");

        Matrix result = evaluate(data);

        double num_correct = 0;
        for (int i = 0; i < result.rows; ++i) {
            for (int j = 0; j < labels.cols; ++j) {
                if (labels(i,j) == 1) {
                    if (result(i, j) >= threshold)
                        ++num_correct;
                    break;
                }
            }
        }
        return num_correct * 100 / data.rows;
    }

    /**
     * Train the network using back propagation.
     *
     * @param data Matrix containing the data
     * @param labels Matrix containing the labels
     */
    void train(const Matrix &data, const Matrix &labels) {
        if (data.rows != labels.rows)
            throw std::invalid_argument("NeuralNetwork::train() - Input-data and label-data need to be of same size");

        auto dW = costPrime(data, labels);
        W1 -= learn_rate * dW[0];
        W2 -= learn_rate * dW[1];
    }

    /**
     * Randomizes the weight-matrices, thereby clearing the network.
     * I.e removes any training.
     */
    void reset() {
        W1.randomize();
        W2.randomize();
    }

    /**
     * Saves the current state of the neural network to a file.
     * The file is of the following format:
     *
     *  1    num_in num_hidden num_out
     *  2    [values of W1 separated by " "]
     *  3    [values of W2 separated by " "]
     *
     * The state is fully represented by W1 and W1, the first line is to
     * check that the file matches the network when reading in the state.
     *
     * @param file_path path where the file that contains the state is saved
     */
    void saveState(const std::string &file_path) const {
        std::remove(file_path.c_str());
        std::ofstream file_out(file_path);
        file_out << num_in << " " << num_hidden << " " << num_out << "\n";
        for (int i = 0; i < W1.rows; ++i) {
            for (int j = 0; j < W1.cols; ++j)
                file_out << W1(i,j) << " ";
        }
        file_out << "\n";
        for (int i = 0; i < W2.rows; ++i) {
            for (int j = 0; j < W2.cols; ++j)
                file_out << W2(i,j) << " ";
        }
        file_out << std::endl;
    }

    /**
     * Reads the state of the neural network from a file at the specified path.
     * Throws error if the file is not found or if the file does not
     * match the neural network.
     *
     * @param file_path path where the file is located
     */
    void readState(const std::string &file_path) {
        std::ifstream file_in(file_path);
        if (!file_in.is_open())
            throw std::invalid_argument("NeuralNetwork::readState() - Error reading file");

        int in, hidden, out;
        file_in >> in >> hidden >> out;
        if (in != num_in || hidden != num_hidden || out != num_out)
            throw std::runtime_error("NeuralNetwork::readState() - File does not match network");

        double d;
        for (int i = 0; i < W1.rows; ++i) {
            for (int j = 0; j < W1.cols; ++j) {
                file_in >> d;
                W1(i,j, d);
            }
        }
        for (int i = 0; i < W2.rows; ++i) {
            for (int j = 0; j < W2.cols; ++j) {
                file_in >> d;
                W2(i,j, d);
            }
        }
    }
};

#endif //NUMBER_RECOGNITION_NEURALNETWORK_H
