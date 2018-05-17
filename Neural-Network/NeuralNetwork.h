#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <fstream>
#include <vector>
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
 */
class NeuralNetwork {
    const int num_in, num_hidden, num_out;
    const double learn_rate;
    Matrix<double> W1, W2;

    /**
     * Activation function, the sigmoid function.
     *
     * @param deri - if true returns derivative of activation function.
     */
    double activation(double x, bool deriv = false) const {
        double d = 1 / (1 + exp(-x));
        return deriv ? d * (1 - d) : d;
    }

    /**
     * Apply activation function element-wise to matrix A.
     *
     * @param deri - if true applies derivative of activation function.
     */
    Matrix<double> activation(const Matrix<double> &a, bool deriv = false) const {
        Matrix<double> res(a.rows(), a.cols());
        std::transform(a.begin(), a.end(), res.begin(), [this, deriv](double d){ return activation(d, deriv); });
        return res;
    }

    Matrix<double> normalizeRows(const Matrix<double> &a) const {
        Matrix<double> m(a.rows(), a.cols());
        for (int i = 0; i < m.rows(); ++i) {
            double sum = std::accumulate(a.begin(i), a.end(i), 0.0);
            if (sum != 0)
                std::transform(a.begin(i), a.end(i), m.begin(i), [sum](double d){ return d/sum; });
        }
        return m;
    }

    std::vector<Matrix<double>> fullForward(const Matrix<double> &A) const {
        if (A.cols() != num_in)
            throw std::invalid_argument("NeuralNetwork::fullForward() - Input does not match neural network");

        Matrix<double> Z2 = A * W1;
        Matrix<double> A2 = activation(Z2);
        Matrix<double> Z3 = A2 * W2;
        Matrix<double> yHat = num_out == 1 ? activation(Z3) : normalizeRows(activation(Z3));

        return {Z2, A2, Z3, yHat};
    }

    std::vector<Matrix<double>> costPrime(const Matrix<double> &data, const Matrix<double> &labels) const {
        if (labels.rows() != data.rows())
            throw std::invalid_argument("NeuralNetwork::costPrime() - Input-data and label-data need to be of same size");

        auto matrices = fullForward(data); // {Z2, A2, Z3, yHat}

        Matrix<double> delta3 = -1.0 * ( labels - matrices[3] ).scalar_multi( activation(matrices[2], true) );
        Matrix<double> d_W2 = matrices[1].transpose() * delta3;

        Matrix<double> delta2 = ( delta3 * W2.transpose() ).scalar_multi( activation(matrices[0], true) );
        Matrix<double> d_W1 = data.transpose() * delta2;

        return {d_W1, d_W2};
    }

public:
    NeuralNetwork(int in, int hidden, int out, double rate) :
    num_in(in), num_hidden(hidden), num_out(out), learn_rate(rate), W1(in, hidden), W2(hidden, out) {
        if (num_in < 1 || num_hidden < 1 || num_out < 1 || learn_rate <= 0)
            throw std::invalid_argument("NeuralNetwork::Constructor() - Invalid argument(s)");
        reset();
    }

    /** Evaluates the data using forward propagation through the network. */
    Matrix<double> evaluate(const Matrix<double> &data) const {
        if (data.cols() != num_in)
            throw std::invalid_argument("NeuralNetwork::evaluate() - Input does not match neural network");

        Matrix<double> res = activation(activation(data * W1) * W2);
        return num_out == 1 ? res : normalizeRows(res);
    }

    /**
     * Returns the percentage of datapoints that after evaluation
     * are classified correctly above the threshold.
     *
     * @param threshold If evaluation is above the threshold it's labeled correct
     * @return percentage of data correctly classified
     */
    double percentCorrect(const Matrix<double> &data, const Matrix<double> &labels, double threshold) const {
        if (data.rows() != labels.rows())
            throw std::invalid_argument("NeuralNetwork::numCorrect() - Input-data and label-data need to be of same size");

        Matrix<double> result = evaluate(data);
        double num_correct = 0;
        for (int i = 0; i < result.rows(); ++i) {
            for (int j = 0; j < labels.cols(); ++j) {
                if (labels(i,j) == 1) {
                    if (result(i, j) >= threshold)
                        ++num_correct;
                    break;
                }
            }
        }
        return 100 * num_correct / data.rows();
    }

    /**
     * Train the network using back propagation.
     *
     * @param data Matrix containing the data
     * @param labels Matrix containing the labels
     */
    void train(const Matrix<double> &data, const Matrix<double> &labels) {
        if (data.rows() != labels.rows())
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
        file_out.precision(15);
        file_out << num_in << " " << num_hidden << " " << num_out << "\n";
        std::for_each(W1.begin(), W1.end(), [&file_out](double d){ file_out << d << ' '; });
        file_out << "\n";
        std::for_each(W2.begin(), W2.end(), [&file_out](double d){ file_out << d << ' '; });
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
        file_in >> W1 >> W2;
    }
};

#endif /* NEURALNETWORK_H */
