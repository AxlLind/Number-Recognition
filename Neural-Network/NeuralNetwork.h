//
// Created by Axel Lindeberg on 2017-07-01.
//

#ifndef NUMBER_RECOGNITION_NEURALNETWORK_H
#define NUMBER_RECOGNITION_NEURALNETWORK_H

#include <cmath>
#include "Matrix.h"

/**
 * Artificial neural network with three layers.
 * Uses forward propagation to classify/apply regression to input data.
 * Uses back-propagation and gradient descent to train the neural network.
 * Supports both single and batch gradient descent.
 *
 * Activation function is the sigmoid function, might want to use ReLU instead.
 *
 * Uses the Matrix class for input, output, and internally.
 */
class NeuralNetwork {
    const int num_in, num_hidden, num_out;
    Matrix W1, W2;
    double learn_rate;

    /**
     * Activation function, currently the sigmoid function.
     *
     * @param derivative - if true returns derivative of activation function.
     */
    double activation(double x, bool derivative = false) {
        double d = 1 / (1 + exp(-x));
        return derivative ? d * (1 - d) : d;
    }

    /**
     * Apply activation function element-wise to matrix A.
     *
     * @param derivative - if true applies derivative of activation function.
     */
    Matrix activation(const Matrix &A, bool derivative = false) {
        Matrix OUT(A.rows, A.cols);
        for (int i = 0; i < OUT.rows; ++i) {
            for (int j = 0; j < OUT.cols; ++j)
                OUT(i, j, activation( A(i, j), derivative ));
        }
        return OUT;
    }

    std::vector<Matrix> fullForward(const Matrix& A) {
        if (A.cols != num_in)
            throw std::invalid_argument("Input does not match neural network");

        Matrix Z2 = A * W1;
        Matrix A2 = activation(Z2);
        Matrix Z3 = A2 * W2;
        Matrix yHat = num_out == 1 ? activation(Z3) : normalizeRows(activation(Z3));

        return std::vector<Matrix>({Z2, A2, Z3, yHat});
    }

    std::vector<Matrix> costPrime(const Matrix &input, const Matrix &y) {
        if (y.rows != input.rows)
            throw std::invalid_argument("Input-data and label-data need to be of same size");

        auto matrices = fullForward(input); // {Z2, A2, Z3, yHat}

        Matrix delta3 = -1 * ( y - matrices[3] ).scalarMulti( activation(matrices[2], true) );
        Matrix d_W2 = matrices[1].T() * delta3;

        Matrix delta2 = ( delta3 * W2.T() ).scalarMulti( activation(matrices[0], true) );
        Matrix d_W1 = input.T() * delta2;

        return std::vector<Matrix>({d_W1, d_W2});
    }

 public:
    NeuralNetwork(int numIn, int numHidden, int numOut, double learnRate) :
            W1(numIn, numHidden), W2(numHidden, numOut),
            num_in(numIn), num_hidden(numHidden), num_out(numOut), learn_rate(learnRate) {
        srand(time(NULL));
        W1.randomizeValues();
        W2.randomizeValues();
    }

    Matrix forward(const Matrix& A) {
        if (A.cols != num_in)
            throw std::invalid_argument("Input does not match neural network");

        Matrix OUT = activation(activation(A * W1) * W2);
        return num_out == 1 ? OUT : normalizeRows(OUT);
    }

    Matrix normalizeRows(const Matrix& A) {
        Matrix OUT(A.rows, A.cols);
        for (int i = 0; i < OUT.rows; ++i) {
            double total = 0;
            for (int j = 0; j < OUT.cols; ++j)
                total += A(i, j);

            for (int j = 0; j < OUT.cols; ++j)
                OUT(i, j, A(i, j) / total);
        }
        return OUT;
    }


    Matrix cost(const Matrix &input, const Matrix &y) {
        if (y.rows != input.rows)
            throw std::invalid_argument("Input-data and label-data need to be of same size");

        Matrix yDiff = y - forward(input);
        Matrix error(yDiff.rows, 1);
        for (int i = 0; i < yDiff.rows; ++i) {
            double sum = 0;
            for (int j = 0; j < yDiff.cols; ++j)
                sum += yDiff(i,j) * yDiff(i,j);

            error(i,0, 0.5 * sum);
        }
        return error;
    }

    void train(const Matrix& trainingData, const Matrix& labels) {
        if (labels.rows != trainingData.rows)
            throw std::invalid_argument("Input-data and label-data need to be of same size");

        auto W_prime = costPrime(trainingData, labels);

        W1 -= learn_rate * W_prime[0];
        W2 -= learn_rate * W_prime[1];
    }
};

#endif NUMBER_RECOGNITION_NEURALNETWORK_H
