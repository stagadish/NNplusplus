//
//  tests.cpp
//  Neural Net
//
//  A bunch of tests for Matrix.hpp & NeuralNet.hpp
//
//  Created by Gil Dekel on 8/19/16.
//  Last edited by Joshua Pratt on 10/01/16.
//

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "Matrix.hpp"
#include "NeuralNet.hpp"

bool printSuccess = 0;
unsigned int failedTests;
unsigned int totalTests;
auto test_start = std::chrono::high_resolution_clock::now();

std::string red() { return "\e[31m"; }
std::string green() { return "\e[32m"; }
std::string yellow() { return "\e[33m"; }

std::string reset() { return "\e[0m"; }

void parseInput(const std::string &fileName, std::vector<Matrix> &inputs,
                std::vector<Matrix> &targetOutputs) {
    std::ifstream in(fileName);
    if (in.fail()) {
        std::cout << "ERROR:: CANNOT READ FROM FILE: '" << fileName << "'\n";
        exit(1);
    }

    std::string instance;
    while (getline(in, instance)) {
        std::stringstream ss(instance);
        size_t target;
        double pixel;

        ss >> target;
        Matrix newTarget(1, 10);
        for (unsigned int m = 0; m < newTarget.getNumOfRows(); ++m) {
            for (unsigned int n = 0; n < newTarget.getNumOfCols(); ++n) {
                newTarget(m, n) = 0.01;
            }
        }
        newTarget(0, target) = 0.99;

        targetOutputs.push_back(std::move(newTarget));
        //        std::cout << "Max value is at: " << result.first << "," <<
        //        result.second << " and the value is: " << result.second <<
        //        std::endl << std::endl << std::endl;

        Matrix instanceInput(1, 784);
        size_t count = 0;
        while (ss >> pixel) {
            instanceInput(0, count) = ((pixel / 255) * 0.99) + 0.01;
            ++count;
        }
        inputs.push_back(std::move(instanceInput));
    }
}

std::string getCurrTime() {
    time_t t = time(0);  // get time now
    struct tm *now = localtime(&t);
    std::string currTime =
        ((now->tm_hour < 10) ? "0" + std::to_string(now->tm_hour)
                             : std::to_string(now->tm_hour));
    currTime += ":" + ((now->tm_min < 10) ? "0" + std::to_string(now->tm_min)
                                          : std::to_string(now->tm_min));
    currTime += ":" + ((now->tm_sec < 10) ? "0" + std::to_string(now->tm_sec)
                                          : std::to_string(now->tm_sec));
    currTime += ' ' + std::to_string(now->tm_mon + 1) + '/' +
                std::to_string(now->tm_mday) + '/' +
                std::to_string(now->tm_year + 1900);

    return currTime;
}

void check(const std::string &testName, const bool check,
           const unsigned int verbosity = 1) {
    totalTests++;

    if (check) {
        if (verbosity > 1) {
            std::cout << green() << "\tTest Success: " << testName << "\n";
            std::cout << reset();
        }
    } else {
        if (verbosity > 0) {
            std::cout << red() << "\tTest Fail:    " << testName << "\n";
            std::cout << reset();
        }
        failedTests++;
    }
}

void testBatch(const std::string &batchName) {
    std::cout << green() << "Testing " << batchName << "\n" << reset();
    failedTests = 0;
    totalTests = 0;

    test_start = std::chrono::high_resolution_clock::now();
}

void testResults() {
    auto test_end = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::duration<double, std::milli>(test_end - test_start)
                   .count();
    unsigned int passedTests = totalTests - failedTests;
    if (passedTests > 0) {
        std::cout << green() << "\tPassed " << passedTests << " tests\n";
    }
    if (failedTests > 0) {
        std::cout << red() << "\tFailed " << failedTests << " tests\n";
    }

    double accuracy = passedTests / double(totalTests);
    if (accuracy != 0 && accuracy != 1) {
        if (accuracy > 0.75) {
            std::cout << green();
        } else if (accuracy > 0.25) {
            std::cout << yellow();
        } else {
            std::cout << red();
        }

        std::cout << "\tAccuracy " << 100 * accuracy << "% on " << totalTests
                  << " tests.\n";
    }
    std::cout << yellow() << "Test time: " << end / 1000 << "s"
              << "\n";
    std::cout << reset() << "\n";
}

void testMatrix() {
    testBatch("Matrix Class");
    size_t m = 1000;
    size_t n = 1000;
    const int init_v = 2;
    const int add_v = 10;
    bool match;

    Matrix mtrx(m, n);

    check("Initialization size (rows)", mtrx.getNumOfRows() == m);
    check("Initialization size (columns)", mtrx.getNumOfCols() == n);

    match = true;
    mtrx.apply([&match](const double &x) {
        if (x != 0) match = false;
    });
    check("Initialize all mtrx entries to 0 by default", match);

    mtrx.apply([](double &x) { x = init_v; });

    match = true;
    mtrx.apply([&match](const double &x) {
        if (x != init_v) match = false;
    });
    check("Initialize all mtrx entries to " + std::to_string(init_v), match);

    Matrix mtrxB(n, m);
    check("Initialization size B (rows)", mtrx.getNumOfRows() == n);
    check("Initialization size B (columns)", mtrx.getNumOfCols() == m);

    int count = 1;
    mtrxB.apply([&count](double &x) { x = count++; });

    match = true;
    count = 1;
    for (size_t i = 0; i < mtrxB.getNumOfRows() && match; ++i) {
        for (size_t j = 0; j < mtrxB.getNumOfCols() && match; ++j) {
            if (mtrxB(i, j) != count) {
                match = false;
            }
            ++count;
        }
    }
    check("Initialize all entries to their index", match);

    check("Equality  a == a", mtrxB == mtrxB);

    match = true;
    const Matrix const_cpy(mtrxB);
    for (size_t i = 0; i < mtrxB.getNumOfRows() && match; ++i) {
        for (size_t j = 0; j < mtrxB.getNumOfCols() && match; ++j) {
            if (mtrxB(i, j) != const_cpy(i, j)) {
                // This uses two different operator()s
                // Do not replace with == as that uses both as const
                match = false;
            }
        }
    }
    check("Const equality (const a)(i, j) == a(i, j)", match);

    check("Inquality a != b", mtrx != mtrxB);

    Matrix B_T = mtrxB.T();
    match = true;
    for (size_t i = 0; i < mtrxB.getNumOfRows() && match; ++i) {
        for (size_t j = 0; j < mtrxB.getNumOfCols() && match; ++j) {
            if (mtrxB(i, j) != B_T(j, i)) {
                match = false;
            }
        }
    }
    check("Transpose m(i, j) == m(j, i)", match);
    check("Transpose of Transpose of m == m", mtrxB.T().T() == mtrxB);

    match = true;
    Matrix mtrx_add_BT(mtrx + B_T);
    for (size_t i = 0; i < mtrx.getNumOfRows() && match; ++i) {
        for (size_t j = 0; j < mtrx.getNumOfCols() && match; ++j) {
            if (mtrx_add_BT(i, j) != mtrx(i, j) + B_T(i, j)) {
                match = false;
            }
        }
    }
    check("Addition commutativity (a+a)(i, j) == a(i, j)+b(i, j)", match);

    match = true;
    Matrix BT_add_mtrx(B_T + mtrx);
    for (size_t i = 0; i < mtrx.getNumOfRows() && match; ++i) {
        for (size_t j = 0; j < mtrx.getNumOfCols() && match; ++j) {
            if (BT_add_mtrx(i, j) != mtrx(i, j) + B_T(i, j)) {
                match = false;
            }
        }
    }
    check("Addition commutativity (b+a)(i, j) == a(i, j)+b(i, j)", match);

    mtrx += add_v;
    match = true;
    for (size_t i = 0; i < mtrx.getNumOfRows() && match; ++i) {
        for (size_t j = 0; j < mtrx.getNumOfCols() && match; ++j) {
            if (mtrx(i, j) != init_v + add_v) {
                match = false;
            }
        }
    }
    check("Internal addition of a scalar and a matrix", match);

    const Matrix tmp = mtrx;
    mtrx += B_T;
    check("Internal addition of a matrix and a matrix", mtrx == tmp + B_T);

    Matrix tmp_2 = mtrx;
    tmp_2 += add_v;
    match = true;
    for (size_t i = 0; i < mtrx.getNumOfRows() && match; ++i) {
        for (size_t j = 0; j < mtrx.getNumOfCols() && match; ++j) {
            if (tmp_2(i, j) != mtrx(i, j) + add_v) {
                match = false;
            }
        }
    }
    check("Internal addition of a matrix and a matrix", match);

    Matrix zero(m, n);
    check("Negation a-a == 0", mtrxB - mtrxB == zero);

    check("Negation of a matrix and a matrix", tmp == mtrx - B_T);

    mtrx -= B_T;
    check("Internal negation of a matrix and a matrix", tmp == mtrx);

    check("Matrix negation by scalar -a == -1*a", -mtrx == -1 * mtrx);

    check("Matrix multiplication by scalar a+a == 2*a",
          mtrx + mtrx == 2 * mtrx);

    testResults();
    /*
    std::cout << "mtrx*B_T:\n";
    std::cout << (mtrx * B_T) << "\n";

    std::cout << "B_T*mtrx:\n";
    std::cout << (B_T * mtrx) << "\n";

    std::cout << "mtrx+2:\n";
    std::cout << (mtrx + 2) << "\n";

    std::cout << "2+mtrx:\n";
    std::cout << (2 + mtrx) << "\n";

    std::cout << "mtrx-2:\n";
    std::cout << (mtrx - 2) << "\n";

    std::cout << "2-mtrx:\n";
    std::cout << (2 - mtrx) << "\n";

    std::cout << "mtrx*2:\n";
    std::cout << (mtrx * 2) << "\n";

    std::cout << "2*mtrx:\n";
    std::cout << (2 * mtrx) << "\n";

    std::cout << "mtrx/B_T:\n";
    std::cout << (mtrx / B_T) << "\n";

    std::cout << "B_T/mtrx:\n";
    std::cout << (B_T / mtrx) << "\n";

    std::cout << "mtrx/2:\n";
    std::cout << (mtrx / 2) << "\n";

    std::cout << "2/mtrx:\n";
    std::cout << (2 / mtrx) << "\n";

    std::cout << "mtrx/=2:\n";
    std::cout << (mtrx /= 2) << "\n";

    std::cout << "mtrx*=2:\n";
    std::cout << (mtrx *= 2) << "\n";

    std::cout << "mtrx*=2:\n";
    std::cout << (mtrx *= 2) << "\n";

    std::cout << "neg = -mtrx:\n";
    Matrix neg{-mtrx};
    std::cout << neg << "\n";

    std::cout << "-neg:\n";
    std::cout << (-neg) << "\n";

    std::cout << "mtrx.dot(mtrxB):\n";
    std::cout << (mtrx.dot(mtrxB)) << "\n";

    std::cout << "mtrxB.dot(mtrx):\n";
    mtrxB.dot(mtrx);

    throw MatrixDimensionsMismatch();
    throw MatrixInnerDimensionsMismatch();
    */
}

void testNeuralNet() {
    testBatch("NeuralNet Class");

    NeuralNet NN(784, 397, 10, 1, 0.1);

    std::cout << "Parsing TESTING data...\n";

    std::vector<Matrix> testInputs;
    std::vector<Matrix> testTargetOutputs;
    parseInput("data/training_data/mnist_train_100.txt", testInputs,
               testTargetOutputs);
    std::cout << "Number of instances: " << testInputs.size() << std::endl;
    std::cout << "Size of inputs: " << testInputs.size()
              << " and size of targetOutputs: " << testTargetOutputs.size()
              << std::endl;
    std::cout << "Size of inputs matrices: " << testInputs[0].getNumOfRows()
              << "," << testInputs[0].getNumOfCols()
              << " and size of targetOutputs matrices: "
              << testTargetOutputs[0].getNumOfRows() << ","
              << testTargetOutputs[0].getNumOfCols() << std::endl;

    testBatch("Untrained NeuralNet");
    for (unsigned int i = 0; i < testInputs.size(); ++i) {
        Matrix result = NN.queryNet(testInputs[i]);

        std::pair<size_t, size_t> resultVal = result.getMaxVal();
        std::pair<size_t, size_t> targetVal =
            testTargetOutputs[i].T().getMaxVal();

        check("Classification of sample " + std::to_string(i),
              resultVal == targetVal, 0);
    }
    testResults();

    std::cout << "Parsing TESTING data...\n";
    NN.loadNetwork("saved_nets/2016-8-24--07-12-33.nn");

    testBatch("Trained NeuralNet");
    for (unsigned int i = 0; i < testInputs.size(); ++i) {
        Matrix result = NN.queryNet(testInputs[i]);

        std::pair<size_t, size_t> resultVal = result.getMaxVal();
        std::pair<size_t, size_t> targetVal =
            testTargetOutputs[i].T().getMaxVal();

        check("Classification of sample " + std::to_string(i),
              resultVal == targetVal, 0);
    }
    check("NeuralNet accuracy maintained after save/load", failedTests == 3);
    testResults();
}

int main(int argc, const char *argv[]) {
    printSuccess = (argc > 1 && std::string(argv[1]) == std::string("-v"));
    testMatrix();
    testNeuralNet();
    return 0;
}
