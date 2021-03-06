#include <cstdlib>
#include <fstream>
#include <iostream>
#include <math.h>
#include <vector>

#include "integerNeuralNet.h"

using namespace std;

// Constructor
integerNeuralNet::integerNeuralNet(int numIn, int numHid, int numOut, int maxN,
                                   int maxW)
    : sizeInput(numIn), sizeHidden(numHid), sizeOutput(numOut), maxNeuron(maxN),
      maxWeight(maxW), neuronsInput(numIn + 1), neuronsHidden(numHid + 1),
      neuronsOutput(numOut), weightsInputToHidden(numIn + 1, numHid),
      weightsHiddenToOutput(numHid + 1, numOut)
{
    maxNeuron = (int)pow(2, maxNeuron - 1);
    maxWeight = (int)pow(2, maxWeight - 1);
    biasNeuron = -1 * maxNeuron + 1;

    activationTable = new int[10 * maxNeuron];
    for (int i = 0; i < 10 * maxNeuron; i++) {
        activationTable[i] = 0;
    }

    // Initialize layers
    neuronsInput.setZero();
    neuronsHidden.setZero();
    neuronsOutput.setZero();

    // Initialize weights
    weightsInputToHidden.setZero();
    weightsHiddenToOutput.setZero();
}

// Destructor
integerNeuralNet::~integerNeuralNet() { delete[] activationTable; }

int integerNeuralNet::activationFunction(int in)
{
    // Look Up Table used for integer activation function
    if (in >= 5 * maxNeuron)
        return maxNeuron;
    else if (in <= -5 * maxNeuron)
        return 0;
    else
        return activationTable[in + 5 * maxNeuron];
}

void integerNeuralNet::feedForward(Eigen::VectorXi in)
{
    neuronsInput << in, biasNeuron;

    neuronsHidden << (weightsInputToHidden.transpose() * neuronsInput),
        biasNeuron;
    for (int j = 0; j < sizeHidden; j++) {
        neuronsHidden(j) = activationFunction(neuronsHidden(j));
    }

    neuronsOutput = weightsHiddenToOutput.transpose() * neuronsHidden;
    for (int k = 0; k < sizeOutput; k++) {
        neuronsOutput(k) = activationFunction(neuronsOutput(k));
    }
}

bool integerNeuralNet::saveWeights(string outFile)
{
    fstream output;
    output.open(outFile, ios::out);

    if (output.is_open()) {
        output << "Dimensions:\n"
               << sizeInput << " " << sizeHidden << " " << sizeOutput << endl;

        output << "Weights Input To Hidden:\n";
        for (int i = 0; i <= sizeInput; i++) {
            for (int j = 0; j < sizeHidden; j++)
                output << weightsInputToHidden(i, j) << " ";
            output << endl;
        }

        output << "Weights Hidden To Output:\n";
        for (int j = 0; j <= sizeHidden; j++) {
            for (int k = 0; k < sizeOutput; k++)
                output << weightsHiddenToOutput(j, k) << " ";
            output << endl;
        }

        output.close();
        return true;
    } else {
        return false;
    }
}

bool integerNeuralNet::loadWeights(string inFile)
{
    fstream input;
    input.open(inFile, ios::in);

    if (input.is_open()) {
        int numInput, numHidden, numOutput;
        string line = "";

        getline(input, line); // Dimensions Label
        input >> numInput >> numHidden >> numOutput;
        getline(input, line); // Clear line feed and newline characters

        if ((numInput == sizeInput) && (numHidden == sizeHidden) &&
            (numOutput == sizeOutput)) {
            getline(input, line); // Weights Label
            for (int i = 0; i <= sizeInput; i++) {
                for (int j = 0; j < sizeHidden; j++)
                    input >> weightsInputToHidden(i, j);
                getline(input, line); // Clear line feed and newline characters
            }

            getline(input, line); // Weights Label
            for (int j = 0; j <= sizeHidden; j++) {
                for (int k = 0; k < sizeOutput; k++)
                    input >> weightsHiddenToOutput(j, k);
                getline(input, line); // Clear line feed and newline characters
            }
            input.close();
            return true;
        } else {
            input.close();
            return false;
        }
    } else {
        return false;
    }
}

bool integerNeuralNet::loadActivationTable(string inFile)
{
    fstream input;
    input.open(inFile, ios::in);

    if (input.is_open()) {
        for (int i = 0; i < 10 * maxNeuron; i++) {
            input >> activationTable[i];
        }

        input.close();
        return true;
    } else {
        return false;
    }
}

int integerNeuralNet::classify(Eigen::VectorXi in)
{
    int max = -1 * maxNeuron;
    int result = 0;

    feedForward(in);

    for (int k = 0; k < sizeOutput; k++) {
        if (neuronsOutput(k) > max) {
            max = neuronsOutput(k);
            result = k;
        }
    }

    return result;
}

bool integerNeuralNet::convertFPWeights(string inFile, string outFile)
{
    double temp;
    double max = getMaxFPWeight(inFile);

    fstream input;
    input.open(inFile, ios::in);

    if (input.is_open()) {
        int numInput, numHidden, numOutput;
        string line = "";

        getline(input, line); // Dimensions Label
        input >> numInput >> numHidden >> numOutput;
        getline(input, line); // Clear line feed and newline characters

        if ((numInput == sizeInput) && (numHidden == sizeHidden) &&
            (numOutput == sizeOutput)) {
            getline(input, line); // Weights Label
            for (int i = 0; i <= sizeInput; i++) {
                for (int j = 0; j < sizeHidden; j++) {
                    input >> temp;
                    weightsInputToHidden(i, j) =
                        (int)((temp / max) * (double)maxWeight);
                }
                getline(input, line); // Clear line feed and newline characters
            }

            getline(input, line); // Weights Label
            for (int j = 0; j <= sizeHidden; j++) {
                for (int k = 0; k < sizeOutput; k++) {
                    input >> temp;
                    weightsHiddenToOutput(j, k) =
                        (int)((temp / max) * (double)maxWeight);
                }
                getline(input, line); // Clear line feed and newline characters
            }
            input.close();
            saveWeights(outFile);
            return true;
        } else {
            input.close();
            return false;
        }
    } else {
        return false;
    }
}

double integerNeuralNet::getMaxFPWeight(string inFile)
{
    fstream input;
    input.open(inFile, ios::in);

    double temp;
    double max = 0.0;

    if (input.is_open()) {
        int numInput, numHidden, numOutput;
        string line = "";

        getline(input, line); // Dimensions Label
        input >> numInput >> numHidden >> numOutput;
        getline(input, line); // Clear line feed and newline characters

        if ((numInput == sizeInput) && (numHidden == sizeHidden) &&
            (numOutput == sizeOutput)) {
            getline(input, line); // Weights Label
            for (int i = 0; i <= sizeInput; i++) {
                for (int j = 0; j < sizeHidden; j++) {
                    input >> temp;
                    if (fabs(temp) > max)
                        max = fabs(temp);
                }
                getline(input, line); // Clear line feed and newline characters
            }

            getline(input, line); // Weights Label
            for (int j = 0; j <= sizeHidden; j++) {
                for (int k = 0; k < sizeOutput; k++) {
                    input >> temp;
                    if (fabs(temp) > max)
                        max = fabs(temp);
                }
                getline(input, line); // Clear line feed and newline characters
            }
            input.close();
            return max;
        } else {
            input.close();
            return 0.0;
        }
    } else {
        return 0.0;
    }
}

bool integerNeuralNet::buildActivationTable(string outFile)
{
    fstream output;
    output.open(outFile, ios::out);

    if (output.is_open()) {
        for (int i = 0; i < 10 * maxNeuron; i++) {
            activationTable[i] =
                (int)((double)maxNeuron /
                      (1.0 + exp(-((double)i - (5.0 * (double)maxNeuron)) /
                                 (double)maxNeuron)));
            output << activationTable[i] << endl;
        }

        output.close();
        return true;
    } else {
        return false;
    }
}

bool integerNeuralNet::convertFPInputs(string inFile, string outFile)
{
    fstream input;
    input.open(inFile, ios::in);
    fstream output;
    output.open(outFile, ios::out);

    if (input.is_open() && output.is_open()) {
        // Find largest value to scale all inputs
        double max, temp;
        int tempInt;
        input >> max;
        max = fabs(max);

        while (input >> temp) {
            if (fabs(temp) > max)
                max = fabs(temp);
        }
        input.close();

        input.open(inFile, ios::in);

        while (!input.eof()) {
            for (int i = 0; i < sizeInput; i++) {
                input >> temp;
                tempInt = (int)((double)maxNeuron * (temp / max));
                output << tempInt << " ";
            }
            output << endl;
        }
        input.close();
        output.close();

        return true;
    } else {
        return false;
    }
}

bool integerNeuralNet::dumpTrace(ofstream &trace)
{
    if (trace.is_open()) {
        trace << "===================================\n"
              << "== Input layer:" << endl;
        trace << neuronsInput;

        trace << "\n\n)== Hidden layer:" << endl;
        trace << neuronsHidden;

        trace << "\n\n== Output layer:" << endl;
        trace << neuronsOutput;

        trace << "===================================\n" << endl;

        return true;
    } else {
        return false;
    }
}