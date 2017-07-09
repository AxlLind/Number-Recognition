//
// Created by Axel Lindeberg on 2017-07-01.
//

#include <iostream>
#include "Matrix.h"
#include "NeuralNetwork.h"

int main()
{
    NeuralNetwork brain(2, 3, 1, 0.2);
    Matrix A(3,2);
    A(0,0, 3 ); A(0,1, 4);
    A(1,0, 5 ); A(1,1, 1);
    A(2,0, 10); A(2,1, 2);

    Matrix Y(3,1);
    Y.setColumn(0, std::vector<double>({0.2, 0.5, 0.8}));
    for (int i = 0; i < 20000; i++) {
        brain.train(A, Y);
    }

    std::cout << brain.forward(A) << "\n\n";
    std::cout << brain.cost(A, Y) << "\n\n";
}