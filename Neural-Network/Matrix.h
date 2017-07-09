//
// Created by Axel Lindeberg on 2017-07-01.
//

#ifndef NUMBER_RECOGNITION_MATRIX_H
#define NUMBER_RECOGNITION_MATRIX_H


#include <vector>
#include <cmath>

/**
 * Class that gives the functionality of matrices from linear algebra.
 * Implements operations like matrix-multiplication, scalar multiplication etc.
 * Overloads several operators such as the *-operator for ease of use.
 *
 * Stores the matrix-data as a std::vector in the background and gives an api to the
 * matrix via the ()-operator, as a get:er A(row,col) and set:er A(row,col, value).
 * Example:
 *
 *   Matrix A(2,3);    // constructs a 2x3 matrix
 *   A(0,0, 5)         // sets element 0,0 equal to 5
 *   double d = A(0,0) // variable d is now equal to 5
 *   A(0,0, A(0,0)+1)  // Adds 1 to element 0,0
 *
 * NOTE: Only supports double as element type
 */
class Matrix {

    std::vector<double> arr;

    double get(int i, int j) const { return arr[i * cols + j]; };
    void set(int i, int j, double value) { arr[i * cols +  j] = value; };

 public:
    const int rows, cols;


    /**
     * Constructs a r x c matrix and sets every element in the matrix to zero
     * @param r - number of rows
     * @param c - number of columns
     */
    Matrix(int r, int c) : arr(r * c), rows(r), cols(c) {
        if (r < 1 || c < 1)
            throw std::invalid_argument("Matrix::Constructor() - Matrix can't have less than 1 row and/or column");

        clear();
    }

    /**
     * Initializes matrix with one column (as a vector equivalent).
     */
    Matrix(std::vector<double>& v) : rows(v.size()), cols(1), arr (v.size()) {
        if (v.size() < 1)
            throw std::invalid_argument("Matrix::Constructor() - Matrix (Vector constructor) needs at least 1 element");

        for (int i = 0; i < arr.size(); ++i)
            arr[i] = v[i];
    }
    // ### Constructors

    /**
     * Transpose of the matrix
     */
    Matrix T() const {
        Matrix A(cols, rows);
        for (int i = 0; i < A.rows; ++i) {
            for (int j = 0; j < A.cols; ++j)
                A(i,j, get(j,i));
        }
        return A;
    }

    /**
     * Multiplies matrix element-wise with other matrix.
     * Matrices need to have the same dimensions.
     */
    Matrix scalarMulti (const Matrix& A) const {
        if (rows != A.rows || cols != A.cols)
            throw std::invalid_argument("Matrix::scalarMulti() - Matrices need to be of same size");

        Matrix OUT(rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j)
                OUT(i,j, get(i,j) * A(i,j) );
        }

        return OUT;
    }

    /**
     * Sets every element in the matrix to 0
     */
    void clear() {
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col)
                set(row, col, 0);
        }
    }

    /**
     * Sets every element to a random value
     */
    void randomizeValues() {
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col)
                set(row, col, 2 * ((double)rand() / (double) RAND_MAX) - 1);
        }
    }

    /**
     * Returns the length of the 'Matrix'. Only works when the
     * Matrix is a vector, i.e has one column
     * @return vector-length
     */
    double vectorLength() const {
        if (cols != 1)
            throw std::invalid_argument("Matrix::vectorLength() - Not a vector, has more than 1 column");

        double out = 0;
        for (int i = 0; i < rows; ++i)
            out += get(i,0) * get(i,0);

        return sqrt(out);
    }

    /**
     * Sets the speficied row equal to the values of the passed vector
     */
    void setRow(int row, const std::vector<double>& v) {
        if (v.size() != cols)
            throw std::invalid_argument("Matrix::setRow() - Row does not match matrix row size");

        for (int j = 0; j < cols; ++j)
            set(row, j, v[j]);
    }

    /**
     * Sets the speficied column equal to the values of the passed vector
     */
    void setColumn(int col, const std::vector<double>& v) {
        if (v.size() != rows)
            throw std::invalid_argument("Matrix::setColumn() - Column does not match matrix column size");

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

    std::string toString() const {
        std::string s = "[";
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                if (get(row, col) > 0)
                    s += " ";

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

Matrix operator*(const Matrix& A, const Matrix& B) {
    if (A.cols != B.rows)
        throw std::invalid_argument("Invalid matrix multiplication: rows and columns do not match correctly");

    Matrix OUT(A.rows, B.cols);
    for (int i = 0; i < OUT.rows; ++i) {
        for (int j = 0; j < OUT.cols; ++j) {
            double tmp = 0;
            for (int k = 0; k < A.cols; ++k)
                tmp += A.get(i, k) * B.get(k, j);

            OUT.set(i, j, tmp);
        }
    }
    return OUT;
}

Matrix operator*(const Matrix& A, const double b) {
    Matrix OUT(A.rows, A.cols);
    for (int i = 0; i < A.rows; ++i) {
        for (int j = 0; j < A.cols; ++j)
            OUT(i, j, A(i,j) * b );
    }
    return OUT;
}

Matrix operator* (double b, const Matrix& A) { return operator* (A, b); }

Matrix operator+ (const Matrix& A, const Matrix& B) {
    if (A.rows != B.rows || A.cols != B.cols)
        throw std::invalid_argument("Invalid matrix addition: dimensions do not match");

    Matrix OUT(A.rows, A.cols);
    for (int i = 0; i < OUT.rows; ++i) {
        for (int j = 0; j < OUT.cols; ++j)
            OUT(i,j, A(i,j) + B(i,j) );
    }
    return OUT;
}

void operator+= (Matrix& A, const Matrix& B) {
    for (int i = 0; i < A.rows; ++i) {
        for (int j = 0; j < A.cols; ++j)
            A(i,j, A(i,j) + B(i,j) );
    }
}

Matrix operator- (const Matrix& A, const Matrix& B) { return operator+(A, -1 * B); }

void operator-= ( Matrix& A, const Matrix& B) { operator+= (A, -1 * B); }

std::ostream& operator<<(std::ostream& os, const Matrix& A) { return os << A.toString(); }

// ### Operator overloading


#endif NUMBER_RECOGNITION_MATRIX_H
