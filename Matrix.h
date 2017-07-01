//
// Created by Axel Lindeberg on 2017-07-01.
//

#ifndef NUMBER_RECOGNITION_MATRIX_H
#define NUMBER_RECOGNITION_MATRIX_H


#include <vector>
#include <cstdlib>
#include <string>

class Matrix {

private:
    std::vector<double> arr;

public:
    int rows;
    int cols;

    Matrix(int r, int c) : arr( r * c )
    {
        rows  = r;
        cols = c;
        clear();
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
        std::string s;
        s = "[";
        for (int row = 0; row < rows; ++row)
        {
            for (int col = 0; col < cols; ++col)
                s += std::to_string( get(row, col) ) + " ";
            s += "\n ";
        }
        s.replace(s.length()-3, 3,  "]");
        return s;
    }

    friend std::ostream& operator<< (std::ostream &stream, const Matrix &a);

    static Matrix multiply(const Matrix& a, const Matrix& b)
    {
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
};

std::ostream& operator<<(std::ostream& os, const Matrix& a)
{
    os << a.toString();
    return os;
}

#endif //NUMBER_RECOGNITION_MATRIX_H
