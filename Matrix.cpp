//
//  Matrix.cpp
//  Neural Net
//
//  Created by Gil Dekel on 8/19/16.
//  Last edited by Gil Dekel on 8/25/16.
//  Copyright © 2016 Gil Dekel. All rights reserved.
//

#include "Matrix.hpp"

/*
 * Private members for reference
 *
 * size_t m_size_;     // (M)xN
 * size_t n_size_;     // Mx(N)
 * double *matrix_;    // a pointer to the array.
 *
 */



/**********************************************************
 * Constructors
 **********************************************************/

Matrix::Matrix(size_t m, size_t n) : m_size_{m}, n_size_{n} {
    matrix_ = std::vector<double>(m_size_ * n_size_);
}

/**********************************************************
 * Operator Overloads
 **********************************************************/

double& Matrix::operator()(size_t row, size_t col) {
    return matrix_[transformIJ(row, col)];
}

const double& Matrix::operator()(size_t row, size_t col) const {
    return matrix_[transformIJ(row, col)];
}

Matrix Matrix::dot(const Matrix& rhs) const {
    Matrix dproduct;
    if (this->n_size_ == rhs.m_size_) {
        dproduct = Matrix(m_size_, rhs.n_size_);
        
        for (size_t Arows = 0; Arows < m_size_; ++Arows) {
            for (size_t Acol = 0; Acol < n_size_; ++Acol) {
                for (size_t Xcol = 0; Xcol < rhs.n_size_; ++Xcol) {
                    dproduct(Arows, Xcol) += (*this)(Arows,Acol) * rhs(Acol,Xcol);
                }
            }
        }
    }
    return dproduct;
}

Matrix Matrix::operator*(const Matrix &rhs) const {
    Matrix product;
    if (this->m_size_ == rhs.m_size_ && this->n_size_ == rhs.n_size_) {
        product = Matrix(m_size_, n_size_);
        
        for (size_t i = 0; i < m_size_; ++i) {
            for (size_t j = 0; j < n_size_; ++j) {
                product(i,j) = (*this)(i,j) * rhs(i,j);
            }
        }
    }
    return product;
}

Matrix Matrix::operator*(double scalar) const {
    Matrix tmp{*this};
    for (int i = 0; i < m_size_*n_size_; ++i) {
        tmp.matrix_[i] *= scalar;
    }
    return tmp;
}

Matrix Matrix::operator+(const Matrix &rhs) const {
    Matrix sum;
    if (this->m_size_ == rhs.m_size_ && this->n_size_ == rhs.n_size_) {
        sum = Matrix(m_size_, n_size_);
        
        for (size_t i = 0; i < m_size_; ++i) {
            for (size_t j = 0; j < n_size_; ++j) {
                sum(i,j) = (*this)(i,j) + rhs(i,j);
            }
        }
    }
    return sum;
}

Matrix Matrix::operator+(double scalar) const {
    Matrix sum(m_size_, n_size_);
    for (size_t i = 0; i < m_size_; ++i) {
        for (size_t j = 0; j < n_size_; ++j) {
            sum(i,j) = (*this)(i,j) + scalar;
        }
    }
    return sum;
}

Matrix Matrix::operator-(const Matrix &rhs) const {
    Matrix diff;
    if (this->m_size_ == rhs.m_size_ && this->n_size_ == rhs.n_size_) {
        diff = Matrix(m_size_, n_size_);
        
        for (size_t i = 0; i < m_size_; ++i) {
            for (size_t j = 0; j < n_size_; ++j) {
                diff(i,j) = (*this)(i,j) - rhs(i,j);
            }
        }
    }
    return diff;
}

Matrix Matrix::operator-(double scalar) const {
    Matrix diff(m_size_, n_size_);
    for (size_t i = 0; i < m_size_; ++i) {
        for (size_t j = 0; j < n_size_; ++j) {
            diff(i,j) = (*this)(i,j) - scalar;
        }
    }
    return diff;
}

Matrix Matrix::operator-() const {
    Matrix neg(m_size_, n_size_);
    for (size_t i = 0; i < m_size_; ++i) {
        for (size_t j = 0; j < n_size_; ++j) {
            neg(i,j) = -(*this)(i,j);
        }
    }
    return neg;
}

/**********************************************************
 * Other Functions
 **********************************************************/

size_t Matrix::getNumOfRows() const { return m_size_; }
size_t Matrix::getNumOfCols() const { return n_size_; }

Matrix Matrix::transpose() const {
    Matrix T(n_size_,m_size_);
    
    for (int i = 0; i < m_size_; ++i) {
        for (int j = 0; j < n_size_; ++j) {
            T(j,i) = (*this)(i, j);
        }
    }
    return T;
}

std::pair<size_t, size_t> Matrix::getMaxVal() const {
    long int maxI = -1;
    long int maxJ = -1;
    double maxVal = -INFINITY;
    
    for (int i = 0; i < m_size_; ++i) {
        for (int j = 0; j < n_size_; ++j) {
            if ((*this)(i,j) >= maxVal) {
                maxVal = (*this)(i,j);
                maxI = i;
                maxJ = j;
            }
        }
    }
    
    return std::pair<size_t, size_t>(maxI, maxJ);
}

void Matrix::printMtrx() const {
    for (int i = 0; i < m_size_; ++i) {
        for (int j = 0; j < n_size_; ++j) {
            std::cout << (*this)(i, j) << "\t\t";
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
 * Non-member, External Functions
 **********************************************************/

Matrix operator*(double scalar, const Matrix& rhs) {
    return rhs*scalar;
}

Matrix operator+(double scalar, const Matrix& rhs) {
    return rhs+scalar;
}

Matrix operator-(double scalar, const Matrix& rhs) {
    Matrix sum(rhs.m_size_, rhs.n_size_);
    for (size_t i = 0; i < rhs.m_size_; ++i) {
        for (size_t j = 0; j < rhs.n_size_; ++j) {
            sum(i,j) = scalar - rhs(i,j);
        }
    }
    return sum;
}