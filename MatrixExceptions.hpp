//
//  MatrixExceptions.h
//  Neural Net
//
//  Created by Gil Dekel on 8/28/16.
//  Last edited by Joshua Pratt on 10/01/16.
//

#ifndef MATRIX_EXCEPTIONS_HPP
#define MATRIX_EXCEPTIONS_HPP

#include <exception>
#include <string>
#include <utility>

class MatrixDimensionsMismatch : public std::exception {
   public:
    MatrixDimensionsMismatch(std::pair<size_t, size_t> expected,
                             std::pair<size_t, size_t> actual)
        : expected_size{expected}, actual_size{actual} {}

   private:
    const char* what() const noexcept {
        std::string expected = std::to_string(expected_size.first) + "x" +
                               std::to_string(expected_size.second);
        std::string actual = std::to_string(actual_size.first) + "x" +
                             std::to_string(actual_size.second);
        std::string what = "Matrix dimensions must be equal. \n\tGot [" +
                           actual + "] expected [" + expected + "].\n";
        return what.c_str();
    }

   protected:
    std::pair<size_t, size_t> expected_size;
    std::pair<size_t, size_t> actual_size;
};

class MatrixInnerDimensionsMismatch : public MatrixDimensionsMismatch {
   public:
    MatrixInnerDimensionsMismatch(std::pair<size_t, size_t> expected,
                                  std::pair<size_t, size_t> actual)
        : MatrixDimensionsMismatch(expected, actual) {}

   private:
    const char* what() const noexcept {
        std::string expected = std::to_string(expected_size.first) + "x[" +
                               std::to_string(expected_size.second) + "]";
        std::string actual = "[" + std::to_string(actual_size.first) + "]x" +
                             std::to_string(actual_size.second);
        std::string what = "Matrix inner dimensions must be equal. \n\tGot " +
                           actual + " expected " + expected + ".\n";
        return what.c_str();
    }
};

#endif /* MATRIX_EXCEPTIONS_HPP */
