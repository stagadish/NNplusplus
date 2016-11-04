//
//  tests.cpp
//  Neural Net
//
//  A bunch of tests for Matrix.hpp & NeuralNet.hpp
//
//  Created by Gil Dekel on 8/19/16.
//  Last edited by Joshua Pratt on 10/01/16.
//

#include <iostream>
#include <fstream>
#include <cstdlib>      //std::rand, std::srand
#include <numeric>      //std::accumulate
#include <sstream>
#include <cstdlib>
#include <string>
#include <cmath>

#include "Matrix.hpp"
#include "NeuralNet.hpp"

const bool printSuccess = 0;
unsigned int failedTests;
unsigned int totalTests;

std::string red() {
    return "\x1B[31m";
}

std::string green() {
    return "\x1B[32m";
}

std::string reset() {
    return "\x1B[0m";
}

int myrandom(int i) { return std::rand()%i;}

void parseInput(const std::string &fileName, std::vector<Matrix> &inputs, std::vector<Matrix> &targetOutputs) {
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
        Matrix newTarget(1,10);
        for (unsigned int m = 0; m < newTarget.getNumOfRows(); ++m) {
            for (unsigned int n = 0; n < newTarget.getNumOfCols(); ++n) {
                newTarget(m,n) = 0.01;
            }
        }
        newTarget(0,target) = 0.99;

        targetOutputs.push_back(std::move(newTarget));
        //        std::cout << "Max value is at: " << result.first << "," << result.second << " and the value is: " << result.second << std::endl << std::endl << std::endl;

        Matrix instanceInput(1,784);
        size_t count = 0;
        while (ss >> pixel) {
            instanceInput(0,count) = ((pixel/255)*0.99)+0.01;
            ++count;
        }
        inputs.push_back(std::move(instanceInput));
    }

}

std::string getCurrTime() {
    time_t t = time(0);   // get time now
    struct tm * now = localtime( & t );
    std::string currTime = ((now->tm_hour < 10) ? "0" + std::to_string(now->tm_hour) : std::to_string(now->tm_hour));
    currTime += ":" + ((now->tm_min < 10) ? "0" + std::to_string(now->tm_min) : std::to_string(now->tm_min));
    currTime += ":" + ((now->tm_sec < 10) ? "0" + std::to_string(now->tm_sec) : std::to_string(now->tm_sec));
    currTime += ' ' + std::to_string(now->tm_mon + 1) + '/' + std::to_string(now->tm_mday) + '/' + std::to_string(now->tm_year + 1900);

    return currTime;
}

void check(const std::string& testName, const bool check) {
    totalTests++;

    if(check) {
        if(printSuccess) {
            std::cout << green() << "\tTest Success: " << testName << "\n" << reset();
        }
    } else {
        std::cout << red() << "\tTest Fail:    " << testName << "\n" << reset();
        failedTests++;
    }
}

void testBatch(const std::string& batchName) {
    std::cout << green() << batchName << " Tests\n" << reset();
    failedTests = 0;
    totalTests = 0;
}

void testResults() {
    if(totalTests != 0) {
        if(failedTests == 0) {
            std::cout << green() << "Passed "<< totalTests <<" tests.";
        } else {
            std::cout << red() << "Failed "<< failedTests << " tests of " << totalTests <<".";
        }
        std::cout << reset() << "\n\n";
    }
}

void testMatrix() {
    testBatch("Matrix Class");
    size_t m = 1000;
    size_t n = 1000;
    const int init_v = 2;
    const int add_v = 10;
    bool match;

    Matrix mtrx(m,n);

    check("Initialization size (rows)", mtrx.getNumOfRows() == m);
    check("Initialization size (columns)", mtrx.getNumOfCols() == n);

    match = true;
    mtrx.apply([&match](double x) {
            if(x != 0) match = false;
            return x;
            });
    check("Initialize all mtrx entries to 0 by default", match);

    mtrx.apply([](double) {return init_v;});

    match = true;
    mtrx.apply([&match](double x) {
            if(x != init_v) match = false;
            return x;
            });
    check("Initialize all mtrx entries to "+std::to_string(init_v), match);

    Matrix mtrxB(n,m);
    check("Initialization size B (rows)", mtrx.getNumOfRows() == n);
    check("Initialization size B (columns)", mtrx.getNumOfCols() == m);

    int count = 1;
    mtrxB.apply([&count](double) {return count++;});

    match = true;
    count = 1;
    for (size_t i = 0; i < mtrxB.getNumOfRows() && match; ++i) {
        for (size_t j = 0; j < mtrxB.getNumOfCols() && match; ++j) {
            if(mtrxB(i,j) != count) {
                match = false;
            }
            ++count;
        }
    }
    check("Initialize all entries to their index", match);

    check("Equality  a == a", mtrxB == mtrxB);
    check("Inquality a != b", mtrx != mtrxB);

    Matrix B_T = mtrxB.T();
    match = true;
    for (size_t i = 0; i < mtrxB.getNumOfRows() && match; ++i) {
        for (size_t j = 0; j < mtrxB.getNumOfCols() && match; ++j) {
            if(mtrxB(i,j) != B_T(j, i)) {
                match = false;
            }
        }
    }
    check("Transpose m(i, j) == m(j, i)", match);
    check("Transpose of Transpose of m == m", mtrxB.T().T() == mtrxB);

    match = true;
    Matrix mtrx_add_BT(mtrx+B_T);
    for (size_t i = 0; i < mtrx.getNumOfRows() && match; ++i) {
        for (size_t j = 0; j < mtrx.getNumOfCols() && match; ++j) {
            if(mtrx_add_BT(i,j) != mtrx(i, j)+B_T(i, j)) {
                match = false;
            }
        }
    }
    check("Addition commutativity (a+a)(i, j) == a(i, j)+b(i, j)", match);

    match = true;
    Matrix BT_add_mtrx(B_T+mtrx);
    for (size_t i = 0; i < mtrx.getNumOfRows() && match; ++i) {
        for (size_t j = 0; j < mtrx.getNumOfCols() && match; ++j) {
            if(BT_add_mtrx(i,j) != mtrx(i, j)+B_T(i, j)) {
                match = false;
            }
        }
    }
    check("Addition commutativity (b+a)(i, j) == a(i, j)+b(i, j)", match);

    mtrx += add_v;
    match = true;
    for (size_t i = 0; i < mtrx.getNumOfRows() && match; ++i) {
        for (size_t j = 0; j < mtrx.getNumOfCols() && match; ++j) {
            if(mtrx(i,j) != init_v+add_v) {
                match = false;
            }
        }
    }
    check("Internal addition of a scalar and a matrix", match);

    const Matrix tmp = mtrx;
    mtrx += B_T;
    check("Internal addition of a matrix and a matrix", mtrx == tmp+B_T);

    Matrix tmp_2 = mtrx;
    tmp_2 += add_v;
    match = true;
    for (size_t i = 0; i < mtrx.getNumOfRows() && match; ++i) {
        for (size_t j = 0; j < mtrx.getNumOfCols() && match; ++j) {
            if(tmp_2(i,j) != mtrx(i,j)+add_v) {
                match = false;
            }
        }
    }
    check("Internal addition of a matrix and a matrix", match);

    Matrix zero(m, n);
    check("Negation a-a == 0", mtrxB-mtrxB == zero);

    check("Negation of a matrix and a matrix", tmp == mtrx-B_T);

    mtrx -= B_T;
    check("Internal negation of a matrix and a matrix", tmp == mtrx);

    check("Matrix negation by scalar -a == -1*a", -mtrx == -1*mtrx);

    check("Matrix multiplication by scalar a+a == 2*a", mtrx+mtrx == 2*mtrx);

    testResults();
    return;

    std::cout << "mtrx*B_T:\n";
    (mtrx*B_T).printMtrx();

    std::cout << "B_T*mtrx:\n";
    (B_T*mtrx).printMtrx();

    std::cout << "mtrx+2:\n";
    (mtrx+2).printMtrx();

    std::cout << "2+mtrx:\n";
    (2+mtrx).printMtrx();

    std::cout << "mtrx-2:\n";
    (mtrx-2).printMtrx();

    std::cout << "2-mtrx:\n";
    (2-mtrx).printMtrx();

    std::cout << "mtrx*2:\n";
    (mtrx*2).printMtrx();

    std::cout << "2*mtrx:\n";
    (2*mtrx).printMtrx();

    std::cout << "mtrx/B_T:\n";
    (mtrx/B_T).printMtrx();

    std::cout << "B_T/mtrx:\n";
    (B_T/mtrx).printMtrx();

    std::cout << "mtrx/2:\n";
    (mtrx/2).printMtrx();

    std::cout << "2/mtrx:\n";
    (2/mtrx).printMtrx();

    std::cout << "mtrx/=2:\n";
    (mtrx/=2).printMtrx();

    std::cout << "mtrx*=2:\n";
    (mtrx*=2).printMtrx();

    std::cout << "mtrx*=2:\n";
    (mtrx*=2).printMtrx();


    std::cout << "neg = -mtrx:\n";
    Matrix neg{-mtrx};
    neg.printMtrx();

    std::cout << "-neg:\n";
    (-neg).printMtrx();

    std::cout << "mtrx.dot(mtrxB):\n";
    auto t_start = std::chrono::high_resolution_clock::now();
    mtrx.dot(mtrxB);
    (mtrx.dot(mtrxB)).printMtrx();
    auto t_end = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::duration<double, std::milli>(t_end-t_start).count();
    std::cout << "END time: " << end/1000 << "s" << std::endl;

    std::cout << "mtrxB.dot(mtrx):\n";
    t_start = std::chrono::high_resolution_clock::now();
    mtrxB.dot(mtrx);
    (mtrxB.dot(mtrx)).printMtrx();
    t_end = std::chrono::high_resolution_clock::now();
    end = std::chrono::duration<double, std::milli>(t_end-t_start).count();
    std::cout << "END time: " << end/1000 << "s" << std::endl;

    throw MatrixDimensionsMismatch();
    throw MatrixInnerDimensionsMismatch();
}

void testNeuralNet() {
    testBatch("NeuralNet Class");

    NeuralNet NN(784, 397, 10, 1, 0.1);

    std::cout << "Parsing TESTING data...\n";

    std::vector<Matrix> testInputs;
    std::vector<Matrix> testTargetOutputs;
    parseInput("data/training_data/mnist_train_100.txt", testInputs, testTargetOutputs);
    std::cout << "Number of instances: " << testInputs.size() << std::endl;
    std::cout << "Size of inputs: " << testInputs.size() << " and size of targetOutputs: " << testTargetOutputs.size() << std::endl;
    std::cout << "Size of inputs matrices: " << testInputs[0].getNumOfRows() << "," << testInputs[0].getNumOfCols() << " and size of targetOutputs matrices: " << testTargetOutputs[0].getNumOfRows() << "," << testTargetOutputs[0].getNumOfCols() << std::endl;


    double numOfTests = testInputs.size();
    double success = 0;
    double fails = 0;

    std::cout << "\n\nTESTING BEGINS!" << std::endl;
    auto t_start = std::chrono::high_resolution_clock::now();
    for (unsigned int i = 0; i < testInputs.size(); ++i) {
        Matrix result = NN.queryNet(testInputs[i]);

        std::pair<size_t, size_t> resultVal = result.getMaxVal();
        std::pair<size_t, size_t> targetVal = testTargetOutputs[i].T().getMaxVal();

        if (resultVal == targetVal) {
            ++success;
        } else {
            ++fails;
        }

    }
    auto t_end = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::duration<double, std::milli>(t_end-t_start).count();

    std::cout << "\tNumber of successful classifications: " << success << "\n\tNumber of failed classifications: " <<
        fails << "\n\tAccuracy: " << success/numOfTests << std::endl;
    std::cout << "TESTING ENDED!\nEND time: " << end/1000 << "s" << std::endl;

    NN.loadNetwork("saved_nets/2016-8-24--07-12-33.nn");

    std::cout << "Parsing TESTING data...\n";

    numOfTests = testInputs.size();
    success = 0;
    fails = 0;

    std::cout << "\n\nTESTING BEGINS!" << std::endl;
    t_start = std::chrono::high_resolution_clock::now();
    for (unsigned int i = 0; i < testInputs.size(); ++i) {
        Matrix result = NN.queryNet(testInputs[i]);

        std::pair<size_t, size_t> resultVal = result.getMaxVal();
        std::pair<size_t, size_t> targetVal = testTargetOutputs[i].T().getMaxVal();

        if (resultVal == targetVal) {
            ++success;
        } else {
            ++fails;
        }

    }
    t_end = std::chrono::high_resolution_clock::now();
    end = std::chrono::duration<double, std::milli>(t_end-t_start).count();

    std::cout << "\tNumber of successful classifications: " << success << "\n\tNumber of failed classifications: " <<
        fails << "\n\tAccuracy: " << success/numOfTests << std::endl;
    std::cout << "TESTING ENDED!\nEND time: " << end/1000 << "s" << std::endl;
    testResults();
}

int main(int /*argc*/, const char * /*argv*/[]) {
    testMatrix();
    testNeuralNet();
    return 0;
}

