//
// Created by Axel Lindeberg on 2017-07-01.
//

#include <iostream>
#include "Matrix.h"
#include "NeuralNetwork.h"

int main()
{
    NeuralNetwork brain(2, 3, 1);
    Matrix A(3,2);
    A(0,0, 0.3); A(0,1, 0.5);
    A(1,0, 0.5); A(1,1, 0.1);
    A(2,0, 1.0); A(2,1, 0.2);
    std::cout << brain.forward(A) << "\n\n";
}