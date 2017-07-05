//
// Created by Axel Lindeberg on 2017-07-01.
//

#ifndef NUMBER_RECOGNITION_NEURALNETWORK_H
#define NUMBER_RECOGNITION_NEURALNETWORK_H

#include <cmath>
#include "Matrix.h"

class NeuralNetwork {
    const int numIn, numHidden, numOut;
    Matrix W1, W2;
    double learnRate;

    double sigmoid(double x) { return 1 / (1 + exp(-x)); }

    Matrix sigmoid(const Matrix A) {
        Matrix OUT(A.rows, A.cols);
        for (int i = 0; i < OUT.rows; ++i) {
            for (int j = 0; j < OUT.cols; ++j)
                OUT(i, j, sigmoid(A(i, j)));
        }
        return OUT;
    }

    double sigmoidPrime(double x) { return exp(-x) / pow((exp(-x) + 1), 2); }

    Matrix sigmoidPrime(const Matrix A) {
        Matrix OUT(A.rows, A.cols);
        for (int i = 0; i < OUT.rows; ++i) {
            for (int j = 0; j < OUT.cols; ++j)
                OUT(i, j, sigmoidPrime(A(i, j)));
        }
        return OUT;
    }

    std::vector<Matrix> fullForward(const Matrix A) {
        if (A.cols != numIn)
            throw std::invalid_argument("Input does not match neural network");

        Matrix Z2 = (A * W1);
        Matrix A2 = sigmoid(Z2);
        Matrix Z3 = A2 * W2;
        Matrix yHat = numOut == 1 ? sigmoid(Z3) : normalizeRows( sigmoid(Z3) );

        return std::vector<Matrix>({Z2, A2, Z3, yHat});
    }

    std::vector<Matrix> costFuncPrime(const Matrix input, const Matrix y)
    {
        if (y.rows != input.rows)
            throw std::invalid_argument("Input-data and label-data need to be of same size");

        std::vector<Matrix> matrices = fullForward(input);
        // {Z2, A2, Z3, yHat}

        Matrix delta3 = -1 * (y - matrices[3]).scalarMulti( sigmoidPrime(matrices[2]) );
        Matrix d_W2 = matrices[1].transpose() * delta3;

        Matrix delta2 = delta3 * W2.transpose() * sigmoidPrime(matrices[0]);
        Matrix d_W1 = input.transpose() * delta2;

        return std::vector<Matrix>({d_W1, d_W2});
    }

public:

    NeuralNetwork(const int numIn, const int numHidden, const int numOut, const double learnRate) :
            W1(numIn, numHidden), W2(numHidden, numOut),
            numIn(numIn), numHidden(numHidden), numOut(numOut), learnRate(learnRate) 
    {
        srand((unsigned int) time(NULL));
        W1.randomizeValues();
        W2.randomizeValues();
    }

    Matrix forward(const Matrix A) {
        if (A.cols != numIn)
            throw std::invalid_argument("Input does not match neural network");

        Matrix OUT = sigmoid( sigmoid(A * W1) * W2 );
        return numOut == 1 ? OUT : normalizeRows( OUT );
    }

    Matrix normalizeRows(const Matrix A)
    {
        Matrix OUT(A.rows, A.cols);
        for (int i = 0; i < OUT.rows; ++i) {
            double total = 0;
            for (int j = 0; j < OUT.cols; ++j) total += A(i, j);

            for (int j = 0; j < OUT.cols; ++j)
                OUT(i, j, A(i, j) / total);
        }
        return OUT;
    }


    Matrix costFunc(const Matrix input, const Matrix y)
    {
        if (y.rows != input.rows)
            throw std::invalid_argument("Input-data and label-data need to be of same size");

        Matrix yDiff = y - forward(input);
        Matrix error(yDiff.rows, 1);
        for (int i = 0; i < yDiff.rows; ++i) {
            double sum = 0;
            for (int j = 0; j < yDiff.cols; ++j)
                sum += yDiff(i,j) * yDiff(i,j);

            error(i,0, sum/2);
        }
        return error;
    }

    void train(const Matrix trainingData, const Matrix labels)
    {
        if (labels.rows != trainingData.rows)
            throw std::invalid_argument("Input-data and label-data need to be of same size");

        std::vector<Matrix> derivatives = costFuncPrime(trainingData, labels);

        W1 -= learnRate * derivatives[0];
        W2 -= learnRate * derivatives[1];
    }
};

#endif NUMBER_RECOGNITION_NEURALNETWORK_H
