//
//  NeuralNet.hpp
//  Neural Net
//
//  Created by Gil Dekel on 8/19/16.
//  Last edited by Joshua Pratt on 10/01/16.
//

#ifndef NEURALNET_HPP_
#define NEURALNET_HPP_

#include <iostream>
#include <vector>
#include "Matrix.hpp"


class NeuralNet {
public:

    /**********************************************************
     * Constructors
     **********************************************************/

    // Default ctor to initialize a new neural net object.
    NeuralNet(const size_t inputNodes = 1, const size_t hiddenNodes = 1,
              const size_t outputNodes = 1, const size_t hiddenLayers = 1, const double learningRate = 0.1);

    // A ctor that loads a saved neural net.
    // Expecting a filename with file type *.nn
    // Initializes a new net wit the weights saved
    // in the file.
    explicit NeuralNet(const std::string &filename);

    /**********************************************************
     * Other Functions
     **********************************************************/

    // Feed forward the input Matrix and return the
    // net's prediction in a Matrix.
    Matrix queryNet(const Matrix &inputList);

    // A single feed forward and back propagation with weight updates.
    void trainingCycle(const Matrix &inputList, const Matrix &targetOutput);

    // Train on a vector of input, output cases
    void trainAll(const std::vector<std::pair<Matrix, Matrix> > &training, bool (*eval)(NeuralNet &, const std::vector<std::pair<Matrix, Matrix> > &, const int));

    // A method to save the current state of the net.
    // Files are saved as *.nn.
    // If no file name is supplied, file will be saved
    // with the time and date the net was saved.
    void saveNetwork(const std::string &name = "") const;

    // Load an existing net into the current object.
    void loadNetwork(const std::string &name);

private:
    // Initializes the weights (parameters) between the different layers
    // Initial values are being drawn from a normal distribution centered
    // at 0, with standard deviation of (number_of_inputs_to_nodes_in_next_layer)^(-0.5)
    Matrix initializeMatrix(const size_t m, const size_t n) const;

    // The activation function. Currently using Sigmoid function.
    static double activationFunction(double &x);

    // A utility function to get the current time in a string.
    // Used to name neural nets when saved.
    std::string getCurrTime() const;

    // Prints the weights of the Net
    friend std::ostream& operator<<(std::ostream& os, const NeuralNet& rhs);

    size_t inNodes_;        // Number of input nodes
    size_t hiddNodes_;      // Number of nodes per hidden layer
    size_t outNodes_;       // Number of output nodes
    size_t hiddLayers_;     // Number of hidden layers
    double LR_;             // The learning rate.

    std::vector<Matrix> weights_;  // The weight matrices
    std::vector<Matrix> outputs_;  // Intermediate outputs of all the layers are required for training.
                                    // this is where they are stored.
};



#endif /* NEURALNET_HPP_ */
