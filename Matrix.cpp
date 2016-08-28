//
//  Matrix.cpp
//  Neural Net
//
//  Created by Gil Dekel on 8/19/16.
//  Last edited by Gil Dekel on 8/28/16.
//

#include "Matrix.hpp"

/*
 * Private members for reference
 *
 * size_t m_size_;     // (M)xN
 * size_t n_size_;     // Mx(N)
 * std::vector<double> matrix_;
 *
 */


/**********************************************************
 * Constructors
 **********************************************************/

Matrix::Matrix(size_t m, size_t n) : m_size_{m}, n_size_{n}, matrix_{std::vector<double>(m*n)} { }

/**********************************************************
 * Operator Overloads
 **********************************************************/

double& Matrix::operator()(size_t row, size_t col) {
    return matrix_[transformIJ(row, col)];
}

const double& Matrix::operator()(size_t row, size_t col) const {
    return matrix_[transformIJ(row, col)];
}


// ADDITION
Matrix& Matrix::operator+=(const Matrix & rhs) {
    std::transform(matrix_.begin(), matrix_.end(), rhs.matrix_.begin(), matrix_.begin(), std::plus<double>());
    return *this;
}

Matrix& Matrix::operator+=(double scalar) {
    std::transform(matrix_.begin(), matrix_.end(), matrix_.begin(), [scalar](double i){ return i+scalar; });
    return *this;
}



// SUBTRACTION
Matrix& Matrix::operator-=(const Matrix & rhs) {
    std::transform(matrix_.begin(), matrix_.end(), rhs.matrix_.begin(), matrix_.begin(), std::minus<double>());
    return *this;
}

Matrix& Matrix::operator-=(double scalar) {
    std::transform(matrix_.begin(), matrix_.end(), matrix_.begin(), [scalar](double i){ return i-scalar; });
    return *this;
}



// MULTIPLICATION
Matrix& Matrix::operator*=(const Matrix & rhs) {
    std::transform(matrix_.begin(), matrix_.end(), rhs.matrix_.begin(), matrix_.begin(), std::multiplies<double>());
    return *this;
}

Matrix& Matrix::operator*=(double scalar) {
    std::transform(matrix_.begin(), matrix_.end(), matrix_.begin(), [scalar](double i){ return i*scalar; });
    return *this;
}



//DIVISION
Matrix& Matrix::operator/=(const Matrix & rhs) {
    std::transform(matrix_.begin(), matrix_.end(), rhs.matrix_.begin(), matrix_.begin(), std::divides<double>());
    return *this;
}

Matrix& Matrix::operator/=(double scalar) {
    std::transform(matrix_.begin(), matrix_.end(), matrix_.begin(), [scalar](double i){ return i/scalar; });
    return *this;
}



// UNARY NEGATION
Matrix Matrix::operator-() const {
    Matrix neg{*this};
    std::transform(neg.matrix_.begin(), neg.matrix_.end(), neg.matrix_.begin(), [](double i){ return -i; });
    return neg;
}




/**********************************************************
 * Other Functions
 **********************************************************/

Matrix Matrix::dot(const Matrix& rhs) const {
    if (this->n_size_ == rhs.m_size_) {
        Matrix dproduct(m_size_, rhs.n_size_);
        
        for (size_t Arows = 0; Arows < m_size_; ++Arows) {
            for (size_t Acol = 0; Acol < n_size_; ++Acol) {
                for (size_t Xcol = 0; Xcol < rhs.n_size_; ++Xcol) {
                    dproduct.matrix_[dproduct.transformIJ(Arows, Xcol)] +=
                    matrix_[transformIJ(Arows, Acol)] * rhs.matrix_[rhs.transformIJ(Acol, Xcol)];
                }
            }
        }
        return dproduct;
    } else
        throw MatrixInnderDimensionsMismatch();
}


size_t Matrix::getNumOfRows() const { return m_size_; }
size_t Matrix::getNumOfCols() const { return n_size_; }

Matrix Matrix::T() const {
    Matrix T(n_size_, m_size_);
    for (size_t i = 0; i < m_size_; ++i) {
        for (size_t j = 0; j < n_size_; ++j) {
            T.matrix_[T.transformIJ(j, i)] = matrix_[transformIJ(i, j)];
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
            if (matrix_[transformIJ(i,j)] >= maxVal) {
                maxVal = matrix_[transformIJ(i,j)];
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
            std::cout << matrix_[transformIJ(i, j)] << "\t\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

/**********************************************************
 * Private Functions
 **********************************************************/

size_t Matrix::transformIJ(size_t i, size_t j) const {
    return i * n_size_ + j;
}

/**********************************************************
 * Non-member, Friend Functions
 **********************************************************/

// ADDITION
Matrix operator+(Matrix lhs, const Matrix &rhs) {
    if (lhs.m_size_ == rhs.m_size_ && lhs.n_size_ == rhs.n_size_) {
        return lhs += rhs;
    } else
        throw MatrixDimensionsMismatch();
}

Matrix operator+(Matrix lhs, double scalar) {
    return lhs += scalar;
}

Matrix operator+(double scalar, Matrix rhs) {
    return rhs += scalar;
}



// SUBTRACTION
Matrix operator-(Matrix lhs, const Matrix &rhs) {
    if (lhs.m_size_ == rhs.m_size_ && lhs.n_size_ == rhs.n_size_) {
        return lhs -= rhs;
    } else
        throw MatrixDimensionsMismatch();
}

Matrix operator-(Matrix lhs, double scalar) {
    return lhs -= scalar;
}

Matrix operator-(double scalar, Matrix rhs) {
    return -rhs += scalar;
}



// MULTIPLICATION
Matrix operator*(Matrix lhs, const Matrix &rhs) {
    if (lhs.m_size_ == rhs.m_size_ && lhs.n_size_ == rhs.n_size_) {
        return lhs *= rhs;
    } else
        throw MatrixDimensionsMismatch();
}

Matrix operator*(Matrix lhs, double scalar) {
    return lhs *= scalar;
}

Matrix operator*(double scalar, Matrix rhs) {
    return rhs *= scalar;
}



//DIVISION
Matrix operator/(Matrix lhs, const Matrix &rhs) {
    if (lhs.m_size_ == rhs.m_size_ && lhs.n_size_ == rhs.n_size_) {
        return lhs /= rhs;
    } else
        throw MatrixDimensionsMismatch();
}

Matrix operator/(Matrix lhs, double scalar) {
    return lhs /= scalar;
}

Matrix operator/(double scalar, Matrix rhs) {
    std::transform(rhs.matrix_.begin(), rhs.matrix_.end(), rhs.matrix_.begin(), [scalar](double i){ return scalar/i; });
    return rhs;
}


