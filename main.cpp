#include "integerNeuralNet.h"
#include <iostream>
#include <fstream>

#define ENABLE_PARSEC_HOOKS

#ifdef ENABLE_PARSEC_HOOKS
#include "hooks.h"
#endif

using namespace std;
int main()
{
#ifdef ENABLE_PARSEC_HOOKS
   __parsec_bench_begin(__custom_integer_nn);
#endif

	integerNeuralNet nn(400,30,10,12,12);
	// The integer neural net is a slight modification of the floating-point one
	// In order to be easier to implement in hardware, all operations are performed on integers of a given accuracy
	// Integer networks are very difficult to train directly
	// This is due to many of the training operations mathematically relying on floating-point numbers (probabilities)
	// As such, training is typically done using a floating-point network which is then converted
	// Several helper functions are provided to allow conversion from a floating-point network

	// ***
	// Conversion operations
	// ***

	// To convert floating-point intputs to integers of the defined bit-depth, use convertFPInputs
	nn.convertFPInputs("input.txt","integerInput.txt");

	// To convert saved weights from a floating-point network for use with an integer one, use convertFPWeights
	nn.convertFPWeights("weights.txt","integerWeights.txt");

	// Because the best activation functions tend to rely on floating-point operations,
	// the activation function is usually saved in memory from a pre-computed table
	// The table may be computed according to the defined bit-depth with the buildActivationTable function
	nn.buildActivationTable("activation.txt");

	// ***
	// Standard operations
	// ***

	// Loading integer saved weights
	nn.loadWeights("integerWeights.txt");

	// Loading pre-computed activation table
	nn.loadActivationTable("activation.txt");

	// Prepare to load test data
	int num_data = 5000;
	int **input;
	input = new int *[num_data];
	for(int i = 0; i < num_data; i++)
		input[i] = new int[400];
	int *output;
	output = new int[num_data];

	// Load test data
	fstream inputs, outputs;
	inputs.open("integerInput.txt",ios::in);
	outputs.open("output.txt",ios::in);

	for (int i = 0; i < num_data; i++)
	{
		for (int j = 0; j < 400; j++)
			inputs >> input[i][j];
		outputs >> output[i];
	}

	inputs.close();
	outputs.close();

	// ***
	// Testing operations
	// ***

	// Test overall integer network's accuracy
	int correct = 0;
	double accuracy = 0.0;

#ifdef ENABLE_PARSEC_HOOKS
	__parsec_roi_begin();
#endif
	for (int i = 0; i < num_data; i++)
	{
		if (nn.classify(input[i]) == output[i])
		{
			correct++;
		}

	}
#ifdef ENABLE_PARSEC_HOOKS
    __parsec_roi_end();
#endif

	accuracy = (double)correct/(double)num_data;
	cout << "Accuracy: " << accuracy << endl;

	// Testing some outputs
	cout << "Value: " << output[30] << ", Result: " << nn.classify(input[30]) << endl;
	cout << "Value: " << output[829] << ", Result: " << nn.classify(input[829]) << endl;
	cout << "Value: " << output[1300] << ", Result: " << nn.classify(input[1300]) << endl;
	cout << "Value: " << output[3670] << ", Result: " << nn.classify(input[3670]) << endl;
	cout << "Value: " << output[4800] << ", Result: " << nn.classify(input[4800]) << endl;

	// ***
	// Cleanup
	// ***

	for(int i = 0; i < num_data; i++)
		delete[] input[i];
	delete[] input;
	delete[] output;

#ifdef ENABLE_PARSEC_HOOKS
    __parsec_bench_end();
#endif

	return 0;
}
