//
//  Matrix.cpp
//  Neural Net
//
//  Created by Gil Dekel on 8/19/16.
//  Last edited by Joshua Pratt on 10/01/16.
//

#include "Matrix.hpp"

/*
 * Private members for reference
 *
 * size_t m_size_;      // (M)xN
 * size_t n_size_;      // Mx(N)
 * double *matrix_;     // A pointer to the array.
 * double **rowPtrs_;   // An array of row pointers.
 *                      // used to avoid repeated arithmetics
 *                      // at each access to the matrix.
 *
 */


/**********************************************************
 * Constructors
 **********************************************************/

Matrix::Matrix(const size_t m, const size_t n) : m_size_{m}, n_size_{n} {
    matrix_ = new double[m_size_ * n_size_]();
    rowPtrs_ = new double*[m_size_];

    for (size_t i = 0; i < m_size_; ++i) {
        rowPtrs_[i] = &matrix_[i*n_size_];
    }
}

Matrix::Matrix(const Matrix &rhs) : m_size_{rhs.m_size_}, n_size_{rhs.n_size_} {
    matrix_ = new double[m_size_ * n_size_]();
    rowPtrs_ = new double*[m_size_];

    for (size_t i = 0; i < m_size_ * n_size_; ++i) {
        matrix_[i] = rhs.matrix_[i];
    }

    for (size_t i = 0; i < m_size_; ++i) {
        rowPtrs_[i] = &matrix_[i*n_size_];
    }

}

Matrix& Matrix::operator=(const Matrix &rhs) {
    if (this != &rhs) {
        Matrix copy{rhs};
        std::swap(*this, copy);
    }
    return *this;
}

Matrix::Matrix(Matrix &&rhs) : m_size_{rhs.m_size_}, n_size_{rhs.n_size_}, matrix_{rhs.matrix_}, rowPtrs_{rhs.rowPtrs_} {
    rhs.m_size_ = 0;
    rhs.n_size_ = 0;
    rhs.matrix_ = nullptr;
    rhs.rowPtrs_ = nullptr;
}

Matrix& Matrix::operator=(Matrix &&rhs) {
    std::swap(m_size_, rhs.m_size_);
    std::swap(n_size_, rhs.n_size_);
    std::swap(matrix_, rhs.matrix_);
    std::swap(rowPtrs_, rhs.rowPtrs_);
    return *this;
}

Matrix::~Matrix() {
    delete [] matrix_;
    delete [] rowPtrs_;
}

/**********************************************************
 * Operator Overloads
 **********************************************************/

double& Matrix::operator()(const size_t row, const size_t col) {
    return rowPtrs_[row][col];
}

const double& Matrix::operator()(const size_t row, const size_t col) const {
    return rowPtrs_[row][col];
}


// ADDITION
Matrix& Matrix::operator+=(const Matrix & rhs) {
    if (m_size_ != rhs.m_size_ || n_size_ != rhs.n_size_) {
        throw MatrixDimensionsMismatch();
    }
    for (size_t i = 0; i < m_size_ * n_size_; ++i) {
        matrix_[i] += rhs.matrix_[i];
    }
    return *this;
}

Matrix& Matrix::operator+=(const double scalar) {
    for (size_t i = 0; i < m_size_ * n_size_; ++i) {
        matrix_[i] += scalar;
    }
    return *this;
}



// SUBTRACTION
Matrix& Matrix::operator-=(const Matrix & rhs) {
    if (m_size_ != rhs.m_size_ || n_size_ != rhs.n_size_) {
        throw MatrixDimensionsMismatch();
    }
    for (size_t i = 0; i < m_size_ * n_size_; ++i) {
        matrix_[i] -= rhs.matrix_[i];
    }
    return *this;
}

Matrix& Matrix::operator-=(double scalar) {
    for (size_t i = 0; i < m_size_ * n_size_; ++i) {
        matrix_[i] -= scalar;
    }
    return *this;
}



// MULTIPLICATION
Matrix& Matrix::operator*=(const Matrix & rhs) {
    if (m_size_ != rhs.m_size_ || n_size_ != rhs.n_size_) {
        throw MatrixDimensionsMismatch();
    }
    for (size_t i = 0; i < m_size_ * n_size_; ++i) {
        matrix_[i] *= rhs.matrix_[i];
    }
    return *this;
}

Matrix& Matrix::operator*=(const double scalar) {
    for (size_t i = 0; i < m_size_ * n_size_; ++i) {
        matrix_[i] *= scalar;
    }
    return *this;
}



//DIVISION
Matrix& Matrix::operator/=(const Matrix & rhs) {
    if (m_size_ != rhs.m_size_ || n_size_ != rhs.n_size_) {
        throw MatrixDimensionsMismatch();
    }
    for (size_t i = 0; i < m_size_ * n_size_; ++i) {
        matrix_[i] /= rhs.matrix_[i];
    }
    return *this;
}

Matrix& Matrix::operator/=(const double scalar) {
    for (size_t i = 0; i < m_size_ * n_size_; ++i) {
        matrix_[i] /= scalar;
    }
    return *this;
}



// UNARY NEGATION
Matrix Matrix::operator-() const {
    Matrix neg{*this};
    for (size_t i = 0; i < m_size_ * n_size_; ++i) {
        neg.matrix_[i] = -neg.matrix_[i];
    }
    return neg;
}

// PRINT MATRIX
std::ostream& operator<<(std::ostream& os, const Matrix& rhs) {
    os << "[";
    for (auto i = 0U; i < rhs.getNumOfRows(); ++i) {
        os << "[";
        for (auto j = 0U; j < rhs.getNumOfCols(); ++j) {
            if(j != 0) {
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

Matrix Matrix::dot(const Matrix& rhs) const {
    if (this->n_size_ != rhs.m_size_) {
        throw MatrixInnerDimensionsMismatch();
    }
    Matrix rhs_T{rhs.T()};
    Matrix dproduct(m_size_, rhs.n_size_);

    for (size_t i = 0; i < m_size_; ++i) {
        for (size_t j = 0; j < rhs_T.m_size_; ++j) {
            double dot = 0;
            for (size_t k = 0; k < n_size_; ++k) {
                dot += rowPtrs_[i][k] * rhs_T.rowPtrs_[j][k];
            }
            dproduct.rowPtrs_[i][j] = dot;
        }
    }
    return dproduct;
}

size_t Matrix::getNumOfRows() const { return m_size_; }
size_t Matrix::getNumOfCols() const { return n_size_; }

Matrix Matrix::T() const {
    Matrix T(n_size_, m_size_);
    for (size_t i = 0; i < m_size_; ++i) {
        for (size_t j = 0; j < n_size_; ++j) {
            T.rowPtrs_[j][i] = rowPtrs_[i][j];
        }
    }
    return T;
}

std::pair<size_t, size_t> Matrix::getMaxVal() const {
    long int maxI = -1;
    long int maxJ = -1;
    double maxVal = -INFINITY;

    for (size_t i = 0; i < m_size_; ++i) {
        for (size_t j = 0; j < n_size_; ++j) {
            if (rowPtrs_[i][j] >= maxVal) {
                maxVal = rowPtrs_[i][j];
                maxI = i;
                maxJ = j;
            }
        }
    }
    return std::pair<size_t, size_t>(maxI, maxJ);
}

void Matrix::printMtrx() const {
    for (size_t i = 0; i < m_size_; ++i) {
        for (size_t j = 0; j < n_size_; ++j) {
            std::cout << rowPtrs_[i][j] << "\t\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

/**********************************************************
 * Non-member, Friend Functions
 **********************************************************/

// ADDITION
Matrix operator+(const Matrix &lhs, const Matrix &rhs) {
    if (lhs.m_size_ != rhs.m_size_ || lhs.n_size_ != rhs.n_size_) {
        throw MatrixDimensionsMismatch();
    }
    Matrix ret = lhs;
    return ret += rhs;
}

Matrix operator+(const Matrix &lhs, const double scalar) {
    Matrix ret = lhs;
    return ret += scalar;
}

Matrix operator+(const double scalar, const Matrix &rhs) {
    Matrix ret = rhs;
    return ret += scalar;
}



// SUBTRACTION
Matrix operator-(const Matrix &lhs, const Matrix &rhs) {
    if (lhs.m_size_ != rhs.m_size_ || lhs.n_size_ != rhs.n_size_) {
        throw MatrixDimensionsMismatch();
    }
    Matrix ret = lhs;
    return ret -= rhs;
}

Matrix operator-(const Matrix &lhs, double scalar) {
    Matrix ret = lhs;
    return ret -= scalar;
}

Matrix operator-(const double scalar, const Matrix &rhs) {
    return -rhs+scalar;
}



// MULTIPLICATION
Matrix operator*(const Matrix &lhs, const Matrix &rhs) {
    if (lhs.m_size_ != rhs.m_size_ || lhs.n_size_ != rhs.n_size_) {
        throw MatrixDimensionsMismatch();
    }
    Matrix ret = lhs;
    return ret *= rhs;
}

Matrix operator*(const Matrix &lhs, const double scalar) {
    Matrix ret = lhs;
    return ret *= scalar;
}

Matrix operator*(const double scalar, const Matrix &rhs) {
    return rhs*scalar;
}



//DIVISION
Matrix operator/(const Matrix &lhs, const Matrix &rhs) {
    if (lhs.m_size_ != rhs.m_size_ || lhs.n_size_ != rhs.n_size_) {
        throw MatrixDimensionsMismatch();
    }
    Matrix ret = lhs;
    return ret /= rhs;
}

Matrix operator/(const Matrix &lhs, double scalar) {
    Matrix ret = lhs;
    return ret /= scalar;
}

Matrix operator/(const double scalar, const Matrix &rhs) {
    Matrix ret = rhs;
    for (size_t i = 0; i < rhs.m_size_ * rhs.n_size_; ++i) {
        ret.matrix_[i] = scalar/rhs.matrix_[i];
    }
    return ret;
}


