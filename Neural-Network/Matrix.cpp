#include <iostream>
#include <algorithm>
#include <numeric>
#include <type_traits>
#include <functional>
#include <cmath>

#define throw_err(s) throw std::out_of_range(s);

// constructors
template<typename T>
Matrix<T>::Matrix(int rows, int cols) : m_rows(rows), m_cols(cols) {
    static_assert(std::is_move_constructible<T>::value, "Matrix - Type has to be move constructible");
    static_assert(std::is_move_assignable<T>::value,    "Matrix - Type has to be move assignable");
    if (rows < 0 || cols < 0)
        throw_err("Matrix::constructor - dimensions have to be positive");
    if ((rows == 0 || cols == 0) && rows + cols != 0)
        throw_err("Matrox::constructor - cannot have 0xM or Nx0 dimensions")
    m_vec = new T[m_rows * m_cols];
    reset();
}

template<typename T>
Matrix<T>::Matrix(std::initializer_list<T> &&s) {
    static_assert(std::is_move_constructible<T>::value, "Matrix - Type has to be move constructible");
    static_assert(std::is_move_assignable<T>::value,    "Matrix - Type has to be move assignable");
    int size = sqrt(s.size());
    if (size * size != s.size())
        throw_err("Matrix::constructor - initializer list is not square");
    m_rows = size;
    m_cols = size;
    m_vec = new T[m_rows * m_cols];
    std::copy(s.begin(), s.end(), begin());
}

template<typename T>
Matrix<T>::Matrix(const Matrix<T> &a) : m_rows(a.rows()), m_cols(a.cols()) {
    m_vec = new T[m_rows * m_cols];
    std::copy(a.begin(), a.end(), begin());
}

template<typename T>
Matrix<T>::Matrix(Matrix<T> &&a) : m_rows(a.rows()), m_cols(a.cols()) {
    std::swap(m_vec, a.m_vec); // other matrix's destructor takes care of deleting our array
}

template<typename T>
void Matrix<T>::operator=(const Matrix<T> &a) {
    if (&a == this) // self assignment
        return;
    delete[] m_vec;
    m_rows = a.rows(); m_cols = a.cols();
    m_vec = new T[m_rows * m_cols];
    std::copy(a.begin(), a.end(), begin());
}

template<typename T>
void Matrix<T>::operator=(Matrix<T> &&a) {
    // self assignment should not happen with the move
    m_rows = a.rows();
    m_cols = a.cols();
    std::swap(m_vec, a.m_vec);
}

template<typename T>
T Matrix<T>::operator()(int i, int j) const {
    if (i >= m_rows || j >= m_cols || i < 0 || j < 0)
        throw_err("Matrix::operator() - Index out of range");
    return m_vec[i * m_cols + j];
}

template<typename T>
T& Matrix<T>::operator()(int i, int j) {
    if (i >= m_rows || j >= m_cols || i < 0 || j < 0)
        throw_err("Matrix::operator() - Index out of range");
    return m_vec[i * m_cols + j];
}

template<typename T>
Matrix<T> operator+(const Matrix<T> &a, const Matrix<T> &b) {
    if (a.rows() != b.rows() || a.cols() != b.cols())
        throw_err("Matrix::addition - Not same dimensions");
    Matrix<T> c(a.rows(), a.cols());
    std::transform(a.begin(), a.end(), b.begin(), c.begin(), std::plus<T>());
    return c;
}

template<typename T>
Matrix<T> operator-(const Matrix<T> &a, const Matrix<T> &b) {
    if (a.rows() != b.rows() || a.cols() != b.cols())
        throw_err("Matrix::subtraction - Not same dimensions");
    Matrix<T> c(a.rows(), a.cols());
    std::transform(a.begin(), a.end(), b.begin(), c.begin(), std::minus<T>());
    return c;
}

template<typename T>
Matrix<T> operator*(const Matrix<T> &a, const Matrix<T> &b) {
    if (a.cols() != b.rows())
        throw_err("Matrix::multiplication - Not correct dimensions");
    Matrix<T> c(a.rows(), b.cols());
    for (int i = 0; i < c.rows(); ++i) {
        for (int j = 0; j < c.cols(); ++j) {
            T tmp = T();
            for (int k = 0; k < a.cols(); ++k)
                tmp += a(i,k) * b(k,j);
            c(i,j) = tmp;
        }
    }
    return c;
}

template<typename T>
Matrix<T> operator*(const T &t, const Matrix<T> &a) { return operator*(a,t); }

template<typename T>
Matrix<T> operator*(const Matrix<T> &a, const T &t) {
    Matrix<T> m(a.rows(), a.cols());
    std::transform(a.begin(), a.end(), m.begin(), [&t](T &elem){ return elem * t; });
    return m;
}

template<typename T>
Matrix<T> Matrix<T>::scalar_multi(const Matrix<T> &a) const {
    if (m_rows != a.rows() || m_cols != a.cols())
        throw std::invalid_argument("Matrix::scalarMulti() - Matrices need to be of same size");
    Matrix b(m_rows, m_cols);
    std::transform(begin(), end(), a.begin(), b.begin(), std::multiplies<T>());
    return b;
}

template<typename T>
std::ostream& operator<<(std::ostream &os, const Matrix<T> &a) {
    for (int i = 0; i < a.rows(); ++i) {
        std::for_each(a.begin(i), a.end(i), [&os](const T &t){ os << t << ' '; });
        os << '\n';
    }
    return os;
}

template<typename T>
std::istream& operator>>(std::istream &is, Matrix<T> &a) {
    std::for_each(a.begin(), a.end(), [&is](T &t){ is >> t; });
    return is;
}

template<typename T>
void Matrix<T>::randomize() {
    std::for_each(begin(), end(), [](T &t){ t = 2 * double(rand()) / RAND_MAX - 1; });
}

template<typename T>
Matrix<T> Matrix<T>::transpose() const {
    Matrix<T> a(m_cols, m_rows);
    for (int i = 0; i < m_rows; ++i)
        for (int j = 0; j < m_cols; ++j)
            a(j,i) = (*this)(i,j);
    return a;
}
