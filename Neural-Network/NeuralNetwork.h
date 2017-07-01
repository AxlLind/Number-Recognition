//
// Created by Axel Lindeberg on 2017-07-01.
//

#ifndef NUMBER_RECOGNITION_NEURALNETWORK_H
#define NUMBER_RECOGNITION_NEURALNETWORK_H

#include <cmath>
#include "Matrix.h"

class NeuralNetwork
{
    Matrix wIN, wOUT;
    int numIn, numHidden, numOut;

    double sigmoidPrime (double x) { return exp(x) / pow((exp(x) + 1), 2);  }
    double sigmoid (double x) { return 1 / (1 + exp(-x)); }

    Matrix sigmoid (const Matrix &A)
    {
        Matrix OUT(A.rows, A.cols);
        for (int i = 0; i < OUT.rows; ++i) {
            for (int j = 0; j < OUT.cols; ++j)
                OUT(i,j, sigmoid( A(i,j) ) );
        }
        return OUT;
    }

public:

    NeuralNetwork(const int input, const int hidden, const int output) :
            wIN(input, hidden), wOUT(hidden, output),
            numIn(input), numHidden(hidden), numOut(output)
    {
        srand((unsigned int) time(NULL));
        wIN.randomizeValues();
        wOUT.randomizeValues();
    }

    Matrix forward(const Matrix A)
    {
        if (A.cols != numIn)
            throw std::invalid_argument("Input does not match neural network");

        return sigmoid( sigmoid(A * wIN) * wOUT );
    }
};

#endif //NUMBER_RECOGNITION_NEURALNETWORK_H
