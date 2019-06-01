#ifndef IntegerNeuralNet
#define IntegerNeuralNet

#include <string>

#include "ext/eigen-library/Eigen/Core"

using namespace std;
class integerNeuralNet
{
private:
	// Layer Sizes - Set at initialization
	int sizeInput, sizeHidden, sizeOutput;

	// Integer Ranges - Bound of integer scales in power of 2
	int maxNeuron, maxWeight;
	int *activationTable;

	// Bias neuron value
	int biasNeuron;

	// Layer Neurons - Eigen vectors
	Eigen::VectorXi neuronsInput;
	Eigen::VectorXi neuronsHidden;
	Eigen::VectorXi neuronsOutput;

	// Layer Weights - Eigen matrices
	Eigen::MatrixXi weightsInputToHidden;
	Eigen::MatrixXi weightsHiddenToOutput;

	// Functions - Private member functions
	int activationFunction(int in);
	void feedForward(Eigen::VectorXi in);

public:
	// Constructor and Destructor
	integerNeuralNet(int numIn, int numHid, int numOut, int maxN, int maxW);
	~integerNeuralNet();

	// Saving and Loading
	bool saveWeights(string outFile);
	bool loadWeights(string inFile);
	bool loadActivationTable(string inFile);

	// Classifying
	int classify(Eigen::VectorXi);

	// Helper functions for new networks without integer weights or activation LUTs
	bool convertFPWeights(string inFile, string outFile);
	double getMaxFPWeight(string inFile);
	bool buildActivationTable(string outFile);
	bool convertFPInputs(string inFile, string outFile);

	// Tracing functions
	bool dumpTrace(ofstream &trace);
};

#endif
