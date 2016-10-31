//
//  MatrixExceptions.h
//  Neural Net
//
//  Created by Gil Dekel on 8/28/16.
//  Last edited by Gil Dekel on 8/28/16.
//

#ifndef MATRIX_EXCEPTIONS_HPP
#define MATRIX_EXCEPTIONS_HPP

#include <exception>

class MatrixDimensionsMismatch : public std::exception {
    const char* what() const noexcept {return "Matrix dimensions must agree.\n";}
};

class MatrixInnerDimensionsMismatch : public std::exception  {
    const char* what() const noexcept {return "Matrix inner dimensions must agree.\n";}
};


#endif /* MATRIX_EXCEPTIONS_HPP */
