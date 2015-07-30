#ifndef IntegerNeuralNet
#define IntegerNeuralNet

class integerNeuralNet
{
	private:
		// Layer Sizes - Set at initialization
		int sizeInput, sizeHidden, sizeOutput;

		// Integer Ranges - Bound of integer scales in power of 2
		int maxNeuron, maxWeight;
		int* activationTable;

		// Layer Neurons - Dynamically allocated arrays
		int* neuronsInput;
		int* neuronsHidden;
		int* neuronsOutput;

		// Layer Weights - Dynamically allocated arrays
		int** weightsInputToHidden;
		int** weightsHiddenToOutput;

		// Functions - Private member functions
		int activationFunction(int in);
		void feedForward(int* in);

	public:
		// Constructor and Destructor
		integerNeuralNet(int numIn, int numHid, int numOut, int maxN, int maxW);
		~integerNeuralNet();

		// Saving and Loading
		bool saveWeights(char* outFile);
		bool loadWeights(char* inFile);
		bool loadActivationTable(char* inFile);

		// Classifying
		int classify(int* in);

		// Helper functions for new networks without integer weights or activation LUTs
		bool convertFPWeights(char* inFile, char* outFile);
		double getMaxFPWeight(char* inFile);
		bool buildActivationTable(char* outFile);
		bool convertFPInputs(char* inFile, char* outFile);
};

#include "integerNeuralNet.cpp"
#endif
