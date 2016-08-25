//
//  NeuralNet.cpp
//  Neural Net
//
//  Created by Gil Dekel on 8/19/16.
//  Copyright © 2016 Gil Dekel. All rights reserved.
//

#include <fstream>

#include "NeuralNet.hpp"

/*
 * Private members for reference
 *
 * size_t inNodes_;
 * size_t hiddNodes_;
 * size_t outNodes_;
 * size_t hiddLayers_;
 * double LR_;
 *
 * std::vector<Matrix*> weights_;
 * std::vector<Matrix*> outputs_;
 *
 */

/**********************************************************
 * Constructors, Copy/Move/Assignment operators, Destructor
 **********************************************************/

NeuralNet::NeuralNet(size_t inputNodes, size_t hiddenNodes, size_t outputNodes, size_t hiddenLayers, double learningRate )
    : inNodes_{inputNodes}, hiddNodes_{hiddenNodes}, outNodes_{outputNodes}, hiddLayers_{hiddenLayers}, LR_{learningRate} {
    weights_ = std::vector<Matrix*>{1 + hiddLayers_};       // A vector to store the current weights/parameters
    outputs_ = std::vector<Matrix*>{2 + hiddLayers_};       // A vector to store the last outputs of each layer
    
    for (int i = 0; i < weights_.size(); ++i) {
        size_t currLayer = 0;
        size_t nextLayer = 0;
        
        if (i == 0) {
            currLayer = inNodes_;
            nextLayer = hiddNodes_;
        } else if (i == weights_.size()-1) {
            currLayer = hiddNodes_;
            nextLayer = outNodes_;
        } else {
            currLayer = hiddNodes_;
            nextLayer = hiddNodes_;
        }
        
        weights_[i] = new Matrix(nextLayer, currLayer);
        initializeNet(weights_[i], nextLayer);
    }
    
    for (int i = 0; i < outputs_.size(); ++i) {
        size_t numOfNodes = 0;
        
        if (i == 0) {
            numOfNodes = inNodes_;
        } else if (i == outputs_.size()-1) {
            numOfNodes = outNodes_;
        } else {
            numOfNodes = hiddNodes_;
        }
        
        outputs_[i] = new Matrix(numOfNodes, 1);
    }
}

NeuralNet::NeuralNet(const std::string &filename) {
    if (filename.substr(filename.length()-3).compare(".nn") != 0) {
        std::cout << "ERROR:: FILE MUST BE OF TYPE *.nn\n";
        exit(1);
    }
    
    std::ifstream in(filename);
    if (in.fail()) {
        std::cout << "ERROR:: CANNOT READ FROM FILE: '" << filename << "'\n";
        exit(1);
    }
    
    in >> inNodes_ >> hiddNodes_ >> outNodes_ >> hiddLayers_ >> LR_;
    weights_ = std::vector<Matrix*>{1 + hiddLayers_};       // A vector to store the current weights/parameters
    outputs_ = std::vector<Matrix*>{2 + hiddLayers_};       // A vector to store the last outputs of each layer
    
    size_t Mrows = 0, Ncols = 0;
    double nextVal = 0;
    
    for (int i = 0; i < weights_.size(); ++i) {
        in >> Mrows >> Ncols;
        weights_[i] = new Matrix(Mrows, Ncols);
        
        for (int m = 0; m < Mrows; ++m) {
            for (int n = 0; n < Ncols; ++n) {
                in >> nextVal;
                (*weights_[i])(m,n) = nextVal;
            }
        }
    }
    
    for (int i = 0; i < outputs_.size(); ++i) {
        size_t numOfNodes = 0;
        
        if (i == 0) {
            numOfNodes = inNodes_;
        } else if (i == outputs_.size()-1) {
            numOfNodes = outNodes_;
        } else {
            numOfNodes = hiddNodes_;
        }
        
        outputs_[i] = new Matrix(numOfNodes, 1);
    }
}

NeuralNet::NeuralNet(const NeuralNet &rhs) : inNodes_{rhs.inNodes_}, hiddNodes_{rhs.hiddNodes_}, outNodes_{rhs.outNodes_}, hiddLayers_{rhs.hiddLayers_}, LR_{rhs.LR_} {
    weights_ = std::vector<Matrix*>(rhs.weights_.size());
    outputs_ = std::vector<Matrix*>(rhs.outputs_.size());
    
    for (int i = 0; i < rhs.weights_.size(); ++i) {
        *weights_[i] = *rhs.weights_[i];
    }
    for (int i = 0; i < rhs.outputs_.size(); ++i) {
        *outputs_[i] = *rhs.outputs_[i];
    }
}

NeuralNet& NeuralNet::operator=(const NeuralNet &rhs) {
    if (this != &rhs) {
        NeuralNet copy{rhs};
        std::swap(*this, copy);
    }
    return *this;
}

NeuralNet::NeuralNet(NeuralNet &&rhs): inNodes_{rhs.inNodes_}, hiddNodes_{rhs.hiddNodes_}, outNodes_{rhs.outNodes_}, hiddLayers_{rhs.hiddLayers_}, LR_{rhs.LR_} {
    weights_ = std::move(rhs.weights_);
    outputs_ = std::move(rhs.outputs_);
    
    rhs.inNodes_ = 0;
    rhs.hiddNodes_ = 0;
    rhs.outNodes_ = 0;
    rhs.hiddLayers_ = 0;
    rhs.LR_ = 0;
    rhs.weights_ = std::vector<Matrix*>(0);
    rhs.outputs_ = std::vector<Matrix*>(0);
}

NeuralNet& NeuralNet::operator=(NeuralNet &&rhs) {
    std::swap(inNodes_, rhs.inNodes_);
    std::swap(hiddNodes_, rhs.hiddNodes_);
    std::swap(outNodes_, rhs.outNodes_);
    std::swap(hiddLayers_, rhs.hiddLayers_);
    std::swap(LR_, rhs.LR_);
    std::swap(weights_, rhs.weights_);
    std::swap(outputs_, rhs.outputs_);
    return *this;
}

NeuralNet::~NeuralNet() {
    for (int i = 0; i < weights_.size(); ++i) {
        delete weights_[i];
        weights_[i] = nullptr;
    }
    
    for (int i = 0; i < outputs_.size(); ++i) {
        delete outputs_[i];
        outputs_[i] = nullptr;
    }
}

/**********************************************************
 * Other Functions
 **********************************************************/

Matrix NeuralNet::queryNet(const Matrix &inputList) const {
    Matrix finalOutput(std::move(inputList.transpose()));
    *outputs_[0] = finalOutput;
    
    for (int i = 0; i < weights_.size(); ++i) {
        finalOutput = weights_[i]->dot(finalOutput);
        
        for (int m = 0; m < finalOutput.getNumOfRows(); ++ m) {
            for (int n = 0; n < finalOutput.getNumOfCols(); ++n) {
                finalOutput(m,n) = activationFunction(finalOutput(m,n));
            }
        }
        *outputs_[i+1] = finalOutput;
    }
    
    return finalOutput;
}

void NeuralNet::trainingCycle(const Matrix &inputList, const Matrix &targetOutput) {
    Matrix currOutput = queryNet(inputList);                     // Returned transposed
    Matrix currTargetOut = targetOutput.transpose();
    Matrix currLayerErrors = currTargetOut-currOutput;            // Calculate the final output layer's error
    
    for (long int i = weights_.size()-1; i >= 0; --i) {
        Matrix prevLayerErrors = weights_[i]->transpose().dot(currLayerErrors);
        
        Matrix deltaWeights = currLayerErrors*currOutput;
        deltaWeights = (1-currOutput)*deltaWeights;
        Matrix prevHiddLayerOutsT = outputs_[i]->transpose();
        deltaWeights = deltaWeights.dot(prevHiddLayerOutsT);
        deltaWeights = LR_*deltaWeights;
        Matrix *newWeights = new Matrix(*weights_[i]);
        *newWeights = *newWeights+deltaWeights;
        delete weights_[i];
        weights_[i] = newWeights;
        
        
        currLayerErrors = prevLayerErrors;
        currOutput = *outputs_[i];
    }
}

void NeuralNet::saveNetwork(const std::string &name) const {
    std::string fileName;
    
    if (name.empty())
        fileName = (getCurrTime() + ".nn");
    else
        fileName = (name + ".nn");
    
    std::ofstream out(fileName);
    if (out.fail()) {
        std::cout << "ERROR:: Fails writing to file " << (getCurrTime() + ".nn") << std::endl;
        exit(1);
    }
    
    out << inNodes_ << " " << hiddNodes_ << " " << outNodes_ << " " << hiddLayers_ << " " << LR_ << std::endl;
    
    for (int i = 0; i < weights_.size(); ++i) {
        out << weights_[i]->getNumOfRows() << " " << weights_[i]->getNumOfCols() << std::endl;
        for (int m = 0; m < weights_[i]->getNumOfRows(); ++m) {
            for (int n = 0; n < weights_[i]->getNumOfCols(); ++n) {
                out << (*weights_[i])(m,n) << " ";
            }
            out << std::endl;
        }
    }
}

void NeuralNet::loadNetwork(const std::string &name) {
    *this = NeuralNet(name);
}

/**********************************************************
 * Private Functions
 **********************************************************/

void NeuralNet::initializeNet(Matrix *wMtrx, size_t nextLayer) {
    std::default_random_engine generator((std::random_device()()));
    std::normal_distribution<double> distribution(0.0, std::pow(nextLayer, -0.5));
    
    for (int m = 0; m < wMtrx->getNumOfRows(); ++m) {
        for (int n = 0; n < wMtrx->getNumOfCols(); ++n) {
            (*wMtrx)(m,n) = distribution(generator);
        }
    }
}

// The activation function. Currently using Sigmoid function.
double NeuralNet::activationFunction(double x) const {
    return 1/(1+std::exp(-x));      // Change here to deploy any other activation function for the nodes.
}

std::string NeuralNet::getCurrTime() const {
    time_t t = time(0);   // get time now
    struct tm * now = localtime( & t );
    std::string currTime = std::to_string(now->tm_year + 1900) + '-' + std::to_string(now->tm_mon + 1) + '-' + std::to_string(now->tm_mday);
    currTime += "--" + ((now->tm_hour < 10) ? "0" + std::to_string(now->tm_hour) : std::to_string(now->tm_hour));
    currTime += "-" + ((now->tm_min < 10) ? "0" + std::to_string(now->tm_min) : std::to_string(now->tm_min));
    currTime += "-" + ((now->tm_sec < 10) ? "0" + std::to_string(now->tm_sec) : std::to_string(now->tm_sec));
    
    return currTime;
}
