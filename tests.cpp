//
//  tests.cpp
//  Neural Net
//
//  A bunch of tests for Matrix.hpp & NeuralNet.hpp
//
//  Created by Gil Dekel on 8/19/16.
//  Last edited by Gil Dekel on 8/30/16.
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
        for (int m = 0; m < newTarget.getNumOfRows(); ++m) {
            for (int n = 0; n < newTarget.getNumOfCols(); ++n) {
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

int main(int argc, const char * argv[]) {
// ****************************************************************************************************************
//                                                Matrix Class Tests
// ****************************************************************************************************************
//    
//    size_t m = 1000;
//    size_t n = 1000;
//
//    Matrix mtrx(m,n);
//    Matrix mtrxB(n,m);
//
//    int count = 1;
//    for (int i = 0; i < mtrxB.getNumOfRows(); ++i) {
//        for (int j = 0; j < mtrxB.getNumOfCols(); ++j) {
//            mtrx(j,i) = 2;
//            mtrxB(i,j) = count;
//            ++count;
//        }
//    }
//
//    std::cout << "Initialize mrtx\n";
//    mtrx.printMtrx();

//    std::cout << "Initialize mrtxB\n";
//    mtrxB.printMtrx();

//    std::cout << "Transpose mrtxB\n";
//    Matrix B_T = mtrxB.T();
//    B_T.printMtrx();
    
//
//    std::cout << "mtrx+B_T:\n";
//    (mtrx+B_T).printMtrx();
//
//    std::cout << "B_T+mtrx:\n";
//    (B_T+mtrx).printMtrx();
//    
//    std::cout << "mtrx += 10\n";
//    (mtrx += 10).printMtrx();
//    
//    std::cout << "mtrx += B_T:\n";
//    (mtrx += B_T).printMtrx();
//    
//    std::cout << "mtrx:\n";
//    mtrx.printMtrx();
//
//    std::cout << "mtrx-B_T:\n";
//    (mtrx-B_T).printMtrx();
//    
//    std::cout << "B_T-mtrx:\n";
//    (B_T-mtrx).printMtrx();
//    
//    std::cout << "mtrx -= 10\n";
//    (mtrx -= 10).printMtrx();
//    
//    std::cout << "mtrx -= B_T:\n";
//    (mtrx -= B_T).printMtrx();
//    
//    std::cout << "mtrx:\n";
//    mtrx.printMtrx();
//
//    std::cout << "mtrx-B_T:\n";
//    (mtrx-B_T).printMtrx();
//
//    std::cout << "B_T-mtrx:\n";
//    (B_T-mtrx).printMtrx();
//
//    std::cout << "mtrx*B_T:\n";
//    (mtrx*B_T).printMtrx();
//
//    std::cout << "B_T*mtrx:\n";
//    (B_T*mtrx).printMtrx();
//
//    std::cout << "mtrx+2:\n";
//    (mtrx+2).printMtrx();
//
//    std::cout << "2+mtrx:\n";
//    (2+mtrx).printMtrx();
//
//    std::cout << "mtrx-2:\n";
//    (mtrx-2).printMtrx();
//
//    std::cout << "2-mtrx:\n";
//    (2-mtrx).printMtrx();
//
//    std::cout << "mtrx*2:\n";
//    (mtrx*2).printMtrx();
//    
//    std::cout << "2*mtrx:\n";
//    (2*mtrx).printMtrx();
//    
//    std::cout << "mtrx/B_T:\n";
//    (mtrx/B_T).printMtrx();
//    
//    std::cout << "B_T/mtrx:\n";
//    (B_T/mtrx).printMtrx();
//    
//    std::cout << "mtrx/2:\n";
//    (mtrx/2).printMtrx();
//    
//    std::cout << "2/mtrx:\n";
//    (2/mtrx).printMtrx();
//    
//    std::cout << "mtrx/=2:\n";
//    (mtrx/=2).printMtrx();
//
//    std::cout << "mtrx*=2:\n";
//    (mtrx*=2).printMtrx();
//    
//    std::cout << "mtrx*=2:\n";
//    (mtrx*=2).printMtrx();
//
//    
//    std::cout << "neg = -mtrx:\n";
//    Matrix neg{-mtrx};
//    neg.printMtrx();
//    
//    std::cout << "-neg:\n";
//    (-neg).printMtrx();
//
//    std::cout << "mtrx.dot(mtrxB):\n";
//    auto t_start = std::chrono::high_resolution_clock::now();
//    mtrx.dot(mtrxB);
////    (mtrx.dot(mtrxB)).printMtrx();
//    auto t_end = std::chrono::high_resolution_clock::now();
//    auto end = std::chrono::duration<double, std::milli>(t_end-t_start).count();
//    std::cout << "END time: " << end/1000 << "s" << std::endl;
//
//    std::cout << "mtrxB.dot(mtrx):\n";
//    t_start = std::chrono::high_resolution_clock::now();
//    mtrxB.dot(mtrx);
////    (mtrxB.dot(mtrx)).printMtrx();
//    t_end = std::chrono::high_resolution_clock::now();
//    end = std::chrono::duration<double, std::milli>(t_end-t_start).count();
//    std::cout << "END time: " << end/1000 << "s" << std::endl;
//
//    throw MatrixDimensionsMismatch();
//    throw MatrixInnderDimensionsMismatch();
    
// ****************************************************************************************************************
//                                            NeuralNet Class Tests
// ****************************************************************************************************************
    NeuralNet NN(784, 397, 10, 1, 0.1);
    
    std::cout << "Parsing TESTING data...\n";
    
    std::vector<Matrix> testInputs;
    std::vector<Matrix> testTargetOutputs;
    parseInput("data/test_data/mnist_test.txt", testInputs, testTargetOutputs);
    std::cout << "Number of instances: " << testInputs.size() << std::endl;
    std::cout << "Size of inputs: " << testInputs.size() << " and size of targetOutputs: " << testTargetOutputs.size() << std::endl;
    std::cout << "Size of inputs matrices: " << testInputs[0].getNumOfRows() << "," << testInputs[0].getNumOfCols() << " and size of targetOutputs matrices: " << testTargetOutputs[0].getNumOfRows() << "," << testTargetOutputs[0].getNumOfCols() << std::endl;
    
    
    double numOfTests = testInputs.size();
    double success = 0;
    double fails = 0;
    
    std::cout << "\n\nTESTING BEGINS!" << std::endl;
    auto t_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < testInputs.size(); ++i) {
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
    for (int i = 0; i < testInputs.size(); ++i) {
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
    
    return 0;
}
