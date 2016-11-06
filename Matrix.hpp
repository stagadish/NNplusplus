//
//  Matrix.hpp
//  Neural Net
//
//  A matrix object, which includes basic operations such as
//  matrix transpose and dot product.
//
//  Created by Gil Dekel on 8/19/16.
//  Last edited by Joshua Pratt on 10/01/16.
//

#ifndef MATRIX_HPP_
#define MATRIX_HPP_

#include <initializer_list>
#include <iterator>
#include <utility>  // std::swap, std::move and std::pair
#include "MatrixExceptions.hpp"

class Matrix {
   public:
    /**********************************************************
     * Constructors
     **********************************************************/

    // Basic ctor to initialize a matrix of size m by n.
    // All matrix positions will be initialized to 0.
    Matrix(const size_t m = 0, const size_t n = 0);

    // Iterator ctor
    template <typename IT>
    Matrix(const IT begin, const IT end, const size_t m, const size_t n)
        : Matrix(m, n) {
            size_t size = std::distance(begin, end);
        if (m * n != size) {
            throw MatrixDimensionsMismatch(std::make_pair(m, n), std::make_pair(size, 1));
        }
        std::copy(begin, end, matrix_);
    }

    // Initializer list ctor
    template <typename T>
    Matrix(const std::initializer_list<T> list)
        : Matrix(list.begin(), list.end(), 1,
                 std::distance(list.begin(), list.end())) {}

    // COPY ctor
    Matrix(const Matrix &rhs);

    // Copy assignment operator
    Matrix &operator=(const Matrix &rhs);

    // MOVE ctor
    Matrix(Matrix &&rhs);

    // Move assignment operator
    Matrix &operator=(Matrix &&rhs);

    // dealloc matrix_ (dtor)
    ~Matrix();

    /**********************************************************
     * Operator Overloads
     **********************************************************/

    // A substitute to operator[] for a 2D arrays
    double &operator()(const size_t row, const size_t col);

    const double &operator()(const size_t row, const size_t col) const;

    // EQUALITY
    bool operator==(const Matrix &rhs) const;
    bool operator!=(const Matrix &rhs) const { return !((*this) == rhs); }

    // ADDITION
    Matrix &operator+=(const Matrix &rhs);

    Matrix &operator+=(const double scalar) {
        return apply([&scalar](double &x) { x += scalar; });
    }

    // Term by term addition operator for two matricies.
    friend Matrix operator+(const Matrix &lhs, const Matrix &rhs) {
        return Matrix(lhs) += rhs;
    }

    // Term by term addition operator for matrix and scalar.
    friend Matrix operator+(const Matrix &lhs, const double scalar) {
        return Matrix(lhs) += scalar;
    }

    // Allowing for the scalar addition commutative property.
    friend Matrix operator+(const double scalar, const Matrix &rhs) {
        return rhs + scalar;
    }

    // SUBTRACTION
    Matrix &operator-=(const Matrix &rhs);

    Matrix &operator-=(const double scalar) {
        return apply([&scalar](double &x) { x -= scalar; });
    }

    // Term by term subtraction operator for two matricies.
    friend Matrix operator-(const Matrix &lhs, const Matrix &rhs) {
        return Matrix(lhs) -= rhs;
    }

    // Term by term subtraction operator for matrix and scalar.
    friend Matrix operator-(const Matrix &lhs, const double scalar) {
        return Matrix(lhs) -= scalar;
    }

    // Term by term subtraction operator for scalar and matrix.
    friend Matrix operator-(const double scalar, const Matrix &rhs) {
        return -rhs + scalar;
    }

    // MULTIPLICATION
    Matrix &operator*=(const Matrix &rhs);

    Matrix &operator*=(const double scalar) {
        return apply([&scalar](double &x) { x *= scalar; });
    }

    // "Regular", term by term multiplication operator.
    // See function dot(Matrix &rhs) for dot product.
    friend Matrix operator*(const Matrix &lhs, const Matrix &rhs) {
        return Matrix(lhs) *= rhs;
    }

    // "Regular" scalar multiplication over matrix.
    friend Matrix operator*(const Matrix &lhs, const double scalar) {
        return Matrix(lhs) *= scalar;
    }

    // Allowing for the scalar multiplication commutative property.
    friend Matrix operator*(const double scalar, const Matrix &rhs) {
        return rhs * scalar;
    }

    // DIVISION
    Matrix &operator/=(const Matrix &rhs);

    Matrix &operator/=(const double scalar) {
        return apply([&scalar](double &x) { x /= scalar; });
    }

    // Term by term division operator for two matricies.
    friend Matrix operator/(const Matrix &lhs, const Matrix &rhs) {
        return Matrix(lhs) /= rhs;
    }

    // Term by term division operator for matrix and scalar.
    friend Matrix operator/(const Matrix &lhs, const double scalar) {
        return Matrix(lhs) /= scalar;
    }

    // Term by term division operator for scalar and matrix.
    friend Matrix operator/(const double scalar, const Matrix &rhs) {
        return rhs.apply([&scalar](double &x) { x = scalar / x; });
    }

    // Unary minus operator for Matrix term by term negation
    Matrix operator-() const { return -1 * (*this); }

    // Print matrix
    friend std::ostream &operator<<(std::ostream &os, const Matrix &rhs);

    /**********************************************************
     * Other Functions
     **********************************************************/

    // A simple matrix algebra dot product operation.
    // Return a 0 by 0 matrix if the dimensions do not match.
    // See operator*(Matrix &rhs) for term by term multiplication.
    Matrix dot(const Matrix &rhs) const;
    Matrix dotT(const Matrix &rhs) const;
    Matrix Tdot(const Matrix &rhs) const;

    // Get number of rows (M)xN
    constexpr size_t getNumOfRows() const { return n_rows; }

    // Get number of columns Mx(N)
    constexpr size_t getNumOfCols() const { return n_cols; }

    // Get number of entries M*N
    constexpr size_t size() const { return n_rows * n_cols; }

    // Transpose the matrix MxN -> NxM
    Matrix T() const;

    // Applies a function (double->double) to all the values of a matrix
    template <typename Func>
    Matrix &apply(Func functor) {
        for (size_t i = 0; i < size(); ++i) {
            functor(matrix_[i]);
        }
        return *this;
    }

    // Const form
    template <typename Func>
    Matrix apply(Func functor) const {
        return Matrix(*this).apply(functor);
    }

    // Get the coordinates of the largest value in the matrix.
    // Will return the coordinates of the earliest larger val.
    std::pair<size_t, size_t> getMaxVal() const;

    // Print the matrix to std::cout
    void printMtrx() const;

   private:
    size_t n_rows;      // (M)xN
    size_t n_cols;      // Mx(N)
    double *matrix_;    // A pointer to the array.
    double **rowPtrs_;  // An array of row pointers.
                        // used to avoid repeated arithmetics
                        // at each access to the matrix.
};

#endif /* MATRIX_HPP_ */
