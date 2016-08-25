//
//  main.cpp
//  Neural Net
//
//  Created by Gil Dekel on 8/21/16.
//  Copyright © 2016 Gil Dekel. All rights reserved.
//  Last edited by Gil Dekel on 8/25/16.
//

#include <iostream>
#include <fstream>
#include <cstdlib>      //std::rand, std::srand
#include <numeric>      //std::accumulate
#include <sstream>
#include <string>
#include <cmath>


#include "NeuralNet.hpp"
#include "Matrix.hpp"

void parseInput(const std::string &fileName, std::vector<Matrix> &inputs, std::vector<Matrix> &targetOutputs);
std::string getCurrTime();
int myrandom(int i) { return std::rand()%i;}

int main(int argc, const char * argv[]) {
    std::srand(unsigned(std::time(0)));
    
    //***********
    // Parameters
    //***********
    
    int CV_k = 10;          // k == 10 in k-fold cross validation
    
    int nodes_in = 784;                         // input is 28x28 pics of hand written digits parsed into 1x784 horizontal vectors
    int nodes_out = 10;                         // output is a vector of size 10, each location represents a different number between 0-9
    int nodes_hidd = (nodes_in+nodes_out)/2;    // number of hidden nodes per hidden lyer is the average between in and out nodes
    int num_hidd_layers = 1;                    // number of hidden layers
    double LR = 0.1;                            // the learning rate of the net
    
    //*********************************************************
    // Prepare data for training using 10-fold cross validation
    //*********************************************************
    
    std::cout << "Parsing TRAINING data...\n";
    
    std::vector<Matrix> trainingInputs;
    std::vector<Matrix> trainingTargetOutputs;
    parseInput("data/training_data/mnist_train_100.txt", trainingInputs, trainingTargetOutputs);
    std::cout << "Number of instances: " << trainingInputs.size() << std::endl;
    std::cout << "Size of inputs: " << trainingInputs.size() << " and size of targetOutputs: " << trainingTargetOutputs.size() << std::endl;
    std::cout << "Size of inputs matrices: " << trainingInputs[78].getNumOfRows() << "," << trainingInputs[78].getNumOfCols() << " and size of targetOutputs matrices: " << trainingTargetOutputs[78].getNumOfRows() << "," << trainingTargetOutputs[78].getNumOfCols() << std::endl;

    
    // Get a vector of shuffled indices. Effectively shuffling the data.
    std::vector<int> shuffledIdxs(trainingInputs.size());
    for (int i = 0; i < shuffledIdxs.size(); ++i) {
        shuffledIdxs[i] = i;
    }
    std::random_shuffle(shuffledIdxs.begin(), shuffledIdxs.end(), myrandom);

    size_t cvClusters = shuffledIdxs.size()/CV_k;
    
    //****************************************
    // Begin 10-fold cross validation training
    //****************************************
    
    std::vector<NeuralNet*> NNs(CV_k);
    std::vector<double> accuracies(CV_k);
    size_t total_num_of_tests = 0;
    
    std::cout << "\n\nTRAINING BEGINS!\n\twith " << CV_k << "-fold cross validation.\n\tSTART time: " << getCurrTime() << std::endl << std::endl;
    for (int i = 0; i < CV_k; ++i) {
        
        size_t numOfTrainingInstances = 0;
        
        std::cout << "************ Training k = " << i+1 << " ************" << "\nSTART time: " << getCurrTime() << std::endl;
        NNs[i] = new NeuralNet(nodes_in, nodes_hidd, nodes_out, num_hidd_layers, LR);
//        double count = trainingInputs.size();
        
        for (int j = 0; j < shuffledIdxs.size(); ++j) {
            if (j < cvClusters*i || j > cvClusters*i + cvClusters-1) {
                ++numOfTrainingInstances;
//                std::cout << (j+1)/count * 100 << "%" << '\r';
                NNs[i]->trainingCycle(trainingInputs[shuffledIdxs[j]], trainingTargetOutputs[shuffledIdxs[j]]);
            }
        }

        size_t numOfTests = 0;
        double success = 0;
        double fails = 0;

        std::cout << "Testing k = " << i+1 << std::endl;
        for (int j = 0; j < shuffledIdxs.size(); ++j) {
            if (j >= cvClusters*i && j <= cvClusters*i + cvClusters-1) {
                ++numOfTests;
                Matrix result = NNs[i]->queryNet(trainingInputs[shuffledIdxs[j]]);
                
                std::pair<size_t, size_t> resultVal = result.getMaxVal();
                std::pair<size_t, size_t> targetVal = trainingTargetOutputs[shuffledIdxs[j]].transpose().getMaxVal();
                
                if (resultVal == targetVal) {
                    ++success;
                } else {
                    ++fails;
                }
            }
        }
        
        accuracies[i] = success/numOfTests;
        total_num_of_tests += numOfTests;
        std::cout << "\tNumber of training instances: " << numOfTrainingInstances << "\n\tNumber of tests: " << numOfTests << "\n\tAccuracy: " << accuracies[i]*100 << "%";
        std::cout << "\nEND time: " << getCurrTime() << std::endl << std::endl;
    }
    
    //********************************
    // Compute the confidence interval
    //********************************
    
    // Find the Mean and standard deviation of the accuracies
    double mean_accu = 0;
    for (int k = 0; k < accuracies.size(); ++k) {
        mean_accu += accuracies[k];
    }
    mean_accu = mean_accu/accuracies.size();
    
    double standard_deviation = 0;
    for (int i = 0; i < accuracies.size(); ++i) {
        standard_deviation += pow((accuracies[i]-mean_accu),2);
    }
    standard_deviation = sqrt(standard_deviation/accuracies.size());
    
    
    // Find the margin of error for a 95% confidence interval assuming n => 30
    double standard_error = standard_deviation/(sqrt(accuracies.size()));
    double margin_of_error = standard_error*1.96;
    
    std::pair<double, double> confidence_interval(mean_accu-margin_of_error, mean_accu+margin_of_error);
    
    // Find the best performing Net
    double maxAccuVal = 0;
    int idxOfBestNN = -1;
    for (int k = 0; k < accuracies.size(); ++k) {
        if (maxAccuVal < accuracies[k]) {
            maxAccuVal = accuracies[k];
            idxOfBestNN = k;
        }
    }
    
    //****************************
    // Print final training report
    //****************************
    
    std::cout << "TRAINING ENDED!\n\twith " << CV_k << "-fold cross validation.\n\tEND time: " << getCurrTime() << std::endl << std::endl;
    std::cout << "\tTotal num of tests: " << total_num_of_tests << "\n\tMean accuracy: " << mean_accu*100 << "%";
    std::cout << "\n\t95% Confidence Interval: " << confidence_interval.first*100 << "% to " << confidence_interval.second*100
            << "%\t[" << mean_accu*100 << "% ± " << margin_of_error*100 << "%]\n";
    std::cout << "\t***********************************************************\n";
    std::cout << "\t* Best performing NN is net No. " << idxOfBestNN+1 << " with accuracy of: " << accuracies[idxOfBestNN]*100 << "%\n";
    std::cout << "\t***********************************************************\n\n\n\n";
    
    
    

    //********************************************
    // Prepare data for testing using the best net
    //********************************************
    
    std::cout << "Parsing TESTING data...\n";
    
    std::vector<Matrix> testInputs;
    std::vector<Matrix> testTargetOutputs;
    parseInput("data/test_data/mnist_test_10.txt", testInputs, testTargetOutputs);
    std::cout << "Number of instances: " << testInputs.size() << std::endl;
    std::cout << "Size of inputs: " << testInputs.size() << " and size of targetOutputs: " << testTargetOutputs.size() << std::endl;
    std::cout << "Size of inputs matrices: " << testInputs[0].getNumOfRows() << "," << testInputs[0].getNumOfCols() << " and size of targetOutputs matrices: " << testTargetOutputs[0].getNumOfRows() << "," << testTargetOutputs[0].getNumOfCols() << std::endl;
    
    
    
//    std::cout << "Training begins!\n";
//    NeuralNet NN(784, 393, 10, 1, 0.1);
//    double count = trainingInputs.size();
//    for (int i = 0; i < trainingInputs.size(); ++i) {
//        std::cout << (i+1)/count * 100 << "%" << '\r';
//        NN.trainingCycle(trainingInputs[i], trainingTargetOutputs[i]);
//    }
//    std::cout << "\nTraining Ended!\n\n\n\n";
//    
//
    
    double numOfTests = testInputs.size();
    double success = 0;
    double fails = 0;
    
    std::cout << "\n\nTESTING BEGINS!" << std::endl;
    for (int i = 0; i < testInputs.size(); ++i) {
        Matrix result = NNs[idxOfBestNN]->queryNet(testInputs[i]);
        
        std::pair<size_t, size_t> resultVal = result.getMaxVal();
        std::pair<size_t, size_t> targetVal = testTargetOutputs[i].transpose().getMaxVal();

        if (resultVal == targetVal) {
            ++success;
        } else {
            ++fails;
        }

    }
    
    std::cout << "\tNumber of successful classifications: " << success << "\n\tNumber of failed classifications: " <<
        fails << "\n\tAccuracy: " << success/numOfTests << std::endl;
    std::cout << "TESTING ENDED!\n";
    
    NNs[idxOfBestNN]->saveNetwork();
    
    return 0;
}

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