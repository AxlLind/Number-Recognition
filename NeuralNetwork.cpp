//
// Created by Axel Lindeberg on 2017-07-01.
//

#include <iostream>
#include "Matrix.h"
#include "NeuralNetwork.h"

int main() {
    Matrix A(3,2);
    A.set(0,0,5);
    A.set(0,1,6);
    A.set(1,0,7);
    A.set(1,1,8);
    A.set(2,0,9);
    A.set(2,1,10);
    std::cout << A << "\n\n";

    Matrix B(2,2);
    B.set(0,0,1);
    B.set(0,1,2);
    B.set(1,0,3);
    B.set(1,1,4);
    std::cout << B << "\n\n";

    std::cout << A * B << "\n\n";
    NeuralNetwork brain(3, 5, 2);
}