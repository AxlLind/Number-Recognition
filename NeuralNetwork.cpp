//
// Created by Axel Lindeberg on 2017-07-01.
//

#include <iostream>
#include "Matrix.h"

int main() {
    Matrix a(3,2);
    a.set(0,0,5);
    a.set(0,1,6);
    a.set(1,0,7);
    a.set(1,1,8);
    a.set(2,0,9);
    a.set(2,1,10);
    std::cout << a << "\n\n";

    Matrix b(2,2);
    b.set(0,0,1);
    b.set(0,1,2);
    b.set(1,0,3);
    b.set(1,1,4);
    std::cout << b << "\n\n";

    Matrix c = Matrix::multiply(a,b);
    std::cout << c << "\n\n";
}