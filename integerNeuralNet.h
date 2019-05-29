#ifndef IntegerNeuralNet
#define IntegerNeuralNet

#include <string>

using namespace std;
class integerNeuralNet
{
private:
	// Layer Sizes - Set at initialization
	int sizeInput, sizeHidden, sizeOutput;

	// Integer Ranges - Bound of integer scales in power of 2
	int maxNeuron, maxWeight;
	int *activationTable;

	// Layer Neurons - Dynamically allocated arrays
	int *neuronsInput;
	int *neuronsHidden;
	int *neuronsOutput;

	// Layer Weights - Dynamically allocated arrays
	int **weightsInputToHidden;
	int **weightsHiddenToOutput;

	// Functions - Private member functions
	int activationFunction(int in);
	void feedForward(int *in);

public:
	// Constructor and Destructor
	integerNeuralNet(int numIn, int numHid, int numOut, int maxN, int maxW);
	~integerNeuralNet();

	// Saving and Loading
	bool saveWeights(string outFile);
	bool loadWeights(string inFile);
	bool loadActivationTable(string inFile);

	// Classifying
	int classify(int *in);

	// Helper functions for new networks without integer weights or activation LUTs
	bool convertFPWeights(string inFile, string outFile);
	double getMaxFPWeight(string inFile);
	bool buildActivationTable(string outFile);
	bool convertFPInputs(string inFile, string outFile);
};

#endif
