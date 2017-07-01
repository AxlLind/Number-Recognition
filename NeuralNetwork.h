//
// Created by Axel Lindeberg on 2017-07-01.
//

#ifndef NUMBER_RECOGNITION_NEURALNETWORK_H
#define NUMBER_RECOGNITION_NEURALNETWORK_H

#include <iostream>
#include <cmath>
#include "Matrix.h"

class NeuralNetwork
{
    Matrix wIN;
    Matrix wOUT;
    int numIn;
    int numHidden;
    int numOut;

    void randomizeWeights(Matrix A)
    {
        for (int row = 0; row < A.rows; ++row)
        {
            for (int col = 0; col < A.cols; ++col)
                A.set(row, col, 2 * ((double)rand() / (double) RAND_MAX) - 1);
        }
        std::cout << A << "\n\n";
    }

    double sigmoid (double x) { return 1 / (1 + exp(-x)); }
    double sidmoidPrime (double x) { return exp(x) / pow((exp(x) + 1),2);  }

public:
    NeuralNetwork(const int input, const int hidden, const int output) :
            wIN(input, hidden), wOUT(hidden, output),
            numIn(input), numHidden(hidden), numOut(output)
    {
        randomizeWeights(wIN);
        randomizeWeights(wOUT);
    }

    Matrix forward(Matrix &input)
    {
        if (input.cols != 1 || input.rows != numIn)
            throw std::invalid_argument("Input does not match neural network");

        Matrix hidden = input * wIN;
        return Matrix(1,1);
    }
};

#endif //NUMBER_RECOGNITION_NEURALNETWORK_H
