//
//  Matrix.cpp
//  Neural Net
//
//  Created by Gil Dekel on 8/19/16.
//  Last edited by Joshua Pratt on 10/01/16.
//

#include <cmath>  // INFINITY
#include <iostream>

#include "Matrix.hpp"

/*
 * Private members for reference
 *
 * size_t n_rows;      // (M)xN
 * size_t n_cols;      // Mx(N)
 * double *matrix_;     // A pointer to the array.
 * double **rowPtrs_;   // An array of row pointers.
 *                      // used to avoid repeated arithmetics
 *                      // at each access to the matrix.
 *
 */

/**********************************************************
 * Constructors
 **********************************************************/

Matrix::Matrix(const size_t m, const size_t n) : n_rows{m}, n_cols{n} {
    matrix_ = new double[size()]();
    rowPtrs_ = new double *[n_rows];

    for (size_t i = 0; i < n_rows; ++i) {
        rowPtrs_[i] = matrix_ + i * n_cols;
    }
}

Matrix::Matrix(const Matrix &rhs) : Matrix(rhs.n_rows, rhs.n_cols) {
    std::copy(rhs.matrix_, rhs.matrix_ + size(), matrix_);
}

Matrix &Matrix::operator=(const Matrix &rhs) {
    if (this != &rhs) {
        if (n_rows != rhs.n_rows || n_cols != rhs.n_cols) {
            Matrix copy(rhs);
            std::swap(*this, copy);
        } else {
            std::copy(rhs.matrix_, rhs.matrix_ + size(), matrix_);
        }
    }
    return *this;
}

Matrix::Matrix(Matrix &&rhs)
    : n_rows{rhs.n_rows},
      n_cols{rhs.n_cols},
      matrix_{rhs.matrix_},
      rowPtrs_{rhs.rowPtrs_} {
    rhs.n_rows = 0;
    rhs.n_cols = 0;
    rhs.matrix_ = nullptr;
    rhs.rowPtrs_ = nullptr;
}

Matrix &Matrix::operator=(Matrix &&rhs) {
    std::swap(n_rows, rhs.n_rows);
    std::swap(n_cols, rhs.n_cols);
    std::swap(matrix_, rhs.matrix_);
    std::swap(rowPtrs_, rhs.rowPtrs_);
    return *this;
}

Matrix::~Matrix() {
    delete[] matrix_;
    delete[] rowPtrs_;
}

/**********************************************************
 * Operator Overloads
 **********************************************************/

double &Matrix::operator()(const size_t row, const size_t col) {
    return rowPtrs_[row][col];
}

const double &Matrix::operator()(const size_t row, const size_t col) const {
    return rowPtrs_[row][col];
}

bool Matrix::operator==(const Matrix &rhs) const {
    if (rhs.n_rows != n_rows) return false;
    if (rhs.n_cols != n_cols) return false;

    for (size_t i = 0; i < size(); ++i) {
        if (matrix_[i] != rhs.matrix_[i]) return false;
    }
    return true;
}

void checkMatrixDimensionMatch(const Matrix& lhs, const Matrix &rhs) {
    if (lhs.getNumOfRows() != rhs.getNumOfRows() || lhs.getNumOfCols() != rhs.getNumOfCols()) {
        throw MatrixDimensionsMismatch(
            std::make_pair(lhs.getNumOfRows(), lhs.getNumOfCols()),
            std::make_pair(rhs.getNumOfRows(), rhs.getNumOfCols()));
    }
}

// ADDITION
Matrix &Matrix::operator+=(const Matrix &rhs) {
    checkMatrixDimensionMatch(*this, rhs);
    for (size_t i = 0; i < size(); ++i) {
        matrix_[i] += rhs.matrix_[i];
    }
    return *this;
}

// SUBTRACTION
Matrix &Matrix::operator-=(const Matrix &rhs) {
    checkMatrixDimensionMatch(*this, rhs);
    for (size_t i = 0; i < size(); ++i) {
        matrix_[i] -= rhs.matrix_[i];
    }
    return *this;
}

// MULTIPLICATION
Matrix &Matrix::operator*=(const Matrix &rhs) {
    checkMatrixDimensionMatch(*this, rhs);
    for (size_t i = 0; i < size(); ++i) {
        matrix_[i] *= rhs.matrix_[i];
    }
    return *this;
}

// DIVISION
Matrix &Matrix::operator/=(const Matrix &rhs) {
    checkMatrixDimensionMatch(*this, rhs);
    for (size_t i = 0; i < size(); ++i) {
        matrix_[i] /= rhs.matrix_[i];
    }
    return *this;
}

// PRINT MATRIX
std::ostream &operator<<(std::ostream &os, const Matrix &rhs) {
    os << "[";
    for (auto i = 0U; i < rhs.getNumOfRows(); ++i) {
        os << "[";
        for (auto j = 0U; j < rhs.getNumOfCols(); ++j) {
            if (j != 0) {
                os << ", ";
            }
            os << rhs(i, j);
        }
        os << "]";
    }
    os << "]";
    return os;
}

/**********************************************************
 * Other Functions
 **********************************************************/

Matrix Matrix::dot(const Matrix &rhs) const {
    if (this->n_cols != rhs.n_rows) {
        throw MatrixInnerDimensionsMismatch(
            std::make_pair(n_rows, n_cols),
            std::make_pair(rhs.n_rows, rhs.n_cols));
    }
    Matrix dproduct(n_rows, rhs.n_cols);
    for (size_t i = 0; i < dproduct.n_rows; ++i) {
        for (size_t j = 0; j < dproduct.n_cols; ++j) {
            double &dot = dproduct.rowPtrs_[i][j];
            for (size_t k = 0; k < n_cols; ++k) {
                dot += rowPtrs_[i][k] * rhs.rowPtrs_[k][j];
            }
        }
    }
    return dproduct;
}

Matrix Matrix::T() const {
    Matrix T(n_cols, n_rows);
    for (size_t i = 0; i < n_rows; ++i) {
        for (size_t j = 0; j < n_cols; ++j) {
            T.rowPtrs_[j][i] = rowPtrs_[i][j];
        }
    }
    return T;
}

std::pair<size_t, size_t> Matrix::getMaxVal() const {
    std::pair<size_t, size_t> max{-1, -1};
    double maxVal = -INFINITY;

    for (size_t i = 0; i < n_rows; ++i) {
        for (size_t j = 0; j < n_cols; ++j) {
            if (rowPtrs_[i][j] >= maxVal) {
                maxVal = rowPtrs_[i][j];
                max = std::make_pair(i, j);
            }
        }
    }
    return max;
}
