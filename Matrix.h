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

public:
    const int rows;
    const int cols;

    Matrix(int r, int c) : arr(r * c), rows(r), cols(c) { clear(); };

    Matrix(double v[], int num) : rows(num), cols(1), arr (num) {
        for (int i = 0; i < arr.size(); ++i) {
            arr[i] = v[i];
        }
    };

    double get(int row, int col) const { return arr[row * cols + col]; };
    void set(int row, int col, double value) { arr[row * cols +  col] = value; };

    void clear()
    {
        for (int row = 0; row < rows; ++row)
        {
            for (int col = 0; col < cols; ++col)
                set(row, col, 0);
        }
    }


    std::string& toString() const
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
        s.replace(s.length()-3, 3,  " ]");
        return s;
    }

    friend std::ostream& operator<< (std::ostream &stream, const Matrix &a);
    friend Matrix operator* (const Matrix &a, const Matrix &b);
    friend Matrix operator* (const Matrix &a, double b);
    friend Matrix operator* (double b, const Matrix &a);
};

// ### Operator overloading

Matrix operator*(const Matrix &a,const Matrix &b) {
    if (a.cols != b.rows)
        throw std::invalid_argument("Invalid matrix multiplication: rows and cols don't match");

    Matrix out(a.rows, b.cols);
    for (int i = 0; i < out.rows; ++i)
    {
        for (int j = 0; j < out.cols; ++j)
        {
            double tmp = 0;
            for (int k = 0; k < a.cols; ++k)
                tmp += a.get(i, k) * b.get(k, j);
            out.set(i, j, tmp);
        }
    }
    return out;
}

Matrix operator*(const Matrix &a, const double b) {
    Matrix out(a.rows, a.cols);
    for (int i = 0; i < a.rows; ++i) {
        for (int j = 0; j < a.cols; ++j) {
            out.set(i, j, a.get(i,j) * b );
        }
    }
    return out;
}

Matrix operator* (double b, const Matrix &a) { return operator* (a, b); }

std::ostream& operator<<(std::ostream& os, const Matrix& a) { return os << a.toString(); }

// ### Operator overloading


#endif //NUMBER_RECOGNITION_MATRIX_H
