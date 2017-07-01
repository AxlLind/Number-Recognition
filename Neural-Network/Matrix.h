//
// Created by Axel Lindeberg on 2017-07-01.
//

#ifndef NUMBER_RECOGNITION_MATRIX_H
#define NUMBER_RECOGNITION_MATRIX_H


#include <vector>
#include <cstdlib>
#include <string>

class Matrix {

    std::vector<double> arr;
    double get(int row, int col) const { return arr[row * cols + col]; };
    void set(int row, int col, double value) { arr[row * cols +  col] = value; };

public:
    const int rows;
    const int cols;

    Matrix(int r, int c) : arr(r * c), rows(r), cols(c) { clear(); };

    Matrix(double v[], int num) : rows(num), cols(1), arr (num) {
        for (int i = 0; i < arr.size(); ++i) {
            arr[i] = v[i];
        }
    };


    void clear()
    {
        for (int row = 0; row < rows; ++row)
        {
            for (int col = 0; col < cols; ++col)
                set(row, col, 0);
        }
    }

    void randomizeValues()
    {
        for (int row = 0; row < rows; ++row)
        {
            for (int col = 0; col < cols; ++col)
                set(row, col, 2 * ((double)rand() / (double) RAND_MAX) - 1);
        }
    }

    std::string toString() const
    {
        std::string s = "[";
        for (int row = 0; row < rows; ++row)
        {
            for (int col = 0; col < cols; ++col) {
                if (get(row, col) > 0) s += " ";
                s += std::to_string(get(row, col)) + " ";
            }
            s += "\n ";
        }
        int i = s.size() > 3 ? 3 : s.size(); // fixes 1x1 matrix
        s.replace(s.length()-i, i,  " ]");
        return s;
    }

    double operator()(int row, int col) const { return get(row, col); }
    void operator()(int row, int col, double value) { set(row, col, value); }
    friend std::ostream& operator<< (std::ostream &stream, const Matrix &A);
    friend Matrix operator* (const Matrix &A, const Matrix &B);
    friend Matrix operator* (const Matrix &A, double b);
    friend Matrix operator* (double b, const Matrix &A);
};


// ### Operator overloading

Matrix operator*(const Matrix &A,const Matrix &B) {
    if (A.cols != B.rows)
        throw std::invalid_argument("Invalid matrix multiplication: rows and columns don't match correctly");

    Matrix out(A.rows, B.cols);
    for (int i = 0; i < out.rows; ++i)
    {
        for (int j = 0; j < out.cols; ++j)
        {
            double tmp = 0;
            for (int k = 0; k < A.cols; ++k)
                tmp += A.get(i, k) * B.get(k, j);
            out.set(i, j, tmp);
        }
    }
    return out;
}

Matrix operator*(const Matrix &A, const double b) {
    Matrix out(A.rows, A.cols);
    for (int i = 0; i < A.rows; ++i) {
        for (int j = 0; j < A.cols; ++j) {
            out.set(i, j, A.get(i,j) * b );
        }
    }
    return out;
}

Matrix operator* (double b, const Matrix &A) { return operator* (A, b); }

std::ostream& operator<<(std::ostream& os, const Matrix& A) { return os << A.toString(); }

// ### Operator overloading


#endif //NUMBER_RECOGNITION_MATRIX_H
