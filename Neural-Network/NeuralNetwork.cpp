//
// Created by Axel Lindeberg on 2017-07-01.
//

#include <iostream>
#include "Matrix.h"
#include "NeuralNetwork.h"

int main()
{
    NeuralNetwork brain(2, 3, 1, 0.5);
    Matrix A(3,2);
    A(0,0, 0.3); A(0,1, 0.4);
    A(1,0, 0.5); A(1,1, 0.1);
    A(2,0, 1.0); A(2,1, 0.2);

    Matrix Y(3,1);
    Y.setColumn(0, std::vector<double>({0.4, 0.5, 0.6}));
    for (int i = 0; i < 10; i++) {
        brain.train(A, Y);
        std::cout << brain.costFunc(A, Y).vectorLength() << "\n";
    }

    std::cout << brain.forward(A);
}