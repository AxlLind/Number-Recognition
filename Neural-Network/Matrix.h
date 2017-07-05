//
// Created by Axel Lindeberg on 2017-07-01.
//

#ifndef NUMBER_RECOGNITION_MATRIX_H
#define NUMBER_RECOGNITION_MATRIX_H


#include <vector>
#include <cstdlib>
#include <string>
#include <cmath>

class Matrix {

    std::vector<double> arr;
    double get(int row, int col) const { return arr[row * cols + col]; };
    void set(int row, int col, double value) { arr[row * cols +  col] = value; };

public:
    int rows, cols;

    Matrix(int r, int c) : arr(r * c), rows(r), cols(c) {
        if (r < 1 || c < 1)
            throw std::invalid_argument("Matrix can't have less than 1 row and/or column");

        clear();
    }

    Matrix(std::vector<double> v, int num) : rows(num), cols(1), arr (num)
    {
        for (int i = 0; i < arr.size(); ++i)
            arr[i] = v[i];
    }

    Matrix transpose() const
    {
        Matrix A(cols, rows);
        for (int i = 0; i < A.rows; ++i) {
            for (int j = 0; j < A.cols; ++j) {
                A(i,j, get(j,i));
            }
        }
        return A;
    }

    Matrix scalarMulti (const Matrix A) const
    {
        if (rows != A.rows || cols != A.cols)
            throw std::invalid_argument("Matrices need to be of same size");

        Matrix OUT(rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                OUT(i,j, get(i,j) * A(i,j) );
            }
        }

        return OUT;
    }

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

    double vectorLength()
    {
        if (cols != 1)
            throw std::invalid_argument("Not a vector, has more than 1 column");

        double out = 0;
        for (int i = 0; i < rows; ++i)
            out += get(i,0) * get(i,0); // should have only 1 column
        return sqrt(out);
    }

    void setRow(int row, std::vector<double> v)
    {
        if (v.size() != cols)
            throw std::invalid_argument("Row does not match matrix row length");

        for (int j = 0; j < cols; ++j)
            set(row, j, v[j]);
    }

    void setColumn(const int col, const std::vector<double> v)
    {
        if (v.size() != rows)
            throw std::invalid_argument("Column does not match matrix column length");

        for (int i = 0; i < rows; ++i)
            set(i, col, v[i]);
    }

    double operator()(int row, int col) const { return get(row, col); }
    void operator()(int row, int col, double value) { set(row, col, value); }

    friend std::ostream& operator<< (std::ostream &stream, const Matrix &A);
    friend Matrix operator* (const Matrix &A, const Matrix &B);
    friend Matrix operator* (const Matrix &A, double b);
    friend Matrix operator* (double b, const Matrix &A);
    friend Matrix operator+ (const Matrix &A, const Matrix &B);
    friend void operator+= ( Matrix &A, const Matrix &B);
    friend Matrix operator- (const Matrix &A, const Matrix &B);
    friend void operator-= ( Matrix &A, const Matrix &B);

    std::string toString() const
    {
        std::string s = "[";
        for (int row = 0; row < rows; ++row)
        {
            for (int col = 0; col < cols; ++col)
            {
                if (get(row, col) > 0) s += " ";
                s += std::to_string(get(row, col)) + " ";
            }
            s += "\n ";
        }
        int i = s.size() > 3 ? 3 : s.size(); // fixes 1x1 matrix
        s.replace(s.length()-i, i,  " ]");
        return s;
    }
};




// ### Operator overloading

Matrix operator*(const Matrix &A,const Matrix &B)
{
    if (A.cols != B.rows)
        throw std::invalid_argument("Invalid matrix multiplication: rows and columns do not match correctly");

    Matrix OUT(A.rows, B.cols);
    for (int i = 0; i < OUT.rows; ++i)
    {
        for (int j = 0; j < OUT.cols; ++j)
        {
            double tmp = 0;
            for (int k = 0; k < A.cols; ++k)
                tmp += A.get(i, k) * B.get(k, j);
            OUT.set(i, j, tmp);
        }
    }
    return OUT;
}

Matrix operator*(const Matrix &A, const double b)
{
    Matrix OUT(A.rows, A.cols);
    for (int i = 0; i < A.rows; ++i)
    {
        for (int j = 0; j < A.cols; ++j)
            OUT(i, j, A(i,j) * b );
    }
    return OUT;
}

Matrix operator* (double b, const Matrix &A) { return operator* (A, b); }

Matrix operator+ (const Matrix &A, const Matrix &B)
{
    if (A.rows != B.rows || A.cols != B.cols)
        throw std::invalid_argument("Matrix dimensions do not match for addition");

    Matrix OUT(A.rows, A.cols);
    for (int i = 0; i < OUT.rows; ++i)
    {
        for (int j = 0; j < OUT.cols; ++j)
            OUT(i,j, A(i,j) + B(i,j) );
    }
    return OUT;
}

void operator+= ( Matrix &A, const Matrix &B)
{
    for (int i = 0; i < A.rows; ++i)
    {
        for (int j = 0; j < A.cols; ++j)
            A(i,j, A(i,j) + B(i,j) );
    }
}

Matrix operator- (const Matrix &A, const Matrix &B) { return operator+(A, -1 * B); }

void operator-= ( Matrix &A, const Matrix &B) { operator+= (A, -1 * B); }

std::ostream& operator<<(std::ostream& os, const Matrix& A) { return os << A.toString(); }

// ### Operator overloading


#endif NUMBER_RECOGNITION_MATRIX_H
