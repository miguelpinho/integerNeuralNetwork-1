#include <fstream>
#include <iostream>

#include "ext/eigen-library/Eigen/Core"
#include "integerNeuralNet.h"

#define ENABLE_PARSEC_HOOKS 1
#define ENABLE_TRACING 0
#define ENABLE_WEIGHT_CONVERSION 1

#if ENABLE_PARSEC_HOOKS
#include "hooks.h"
#endif

using namespace std;
int main()
{
#if ENABLE_PARSEC_HOOKS
    __parsec_bench_begin(__custom_integer_nn);
#endif

    int bits_neurons = 12, bits_weights = 12;

    // Original floating point neural-net files.
    string input_file = "fp-files/input.txt";
    string weights_file = "fp-files/weights.txt";
    string output_file = "fp-files/output.txt";
    // Integer neural-net files generated.
    string int_input_file =
        "int-files/integerInput_" + to_string(bits_neurons) + "bits.txt";
    string int_weights_file =
        "int-files/integerWeights_" + to_string(bits_weights) + "bits.txt";
    string activation_file =
        "int-files/activation_" + to_string(bits_neurons) + "bits.txt";
    // Tracing output file.
    string trace_file = "trace.out";

    // TODO: change numIn, numHid, numOut to come from files.
    integerNeuralNet nn(400, 30, 10, bits_neurons, bits_weights);
    // The integer neural net is a slight modification of the floating-point one
    // In order to be easier to implement in hardware, all operations are
    // performed on integers of a given accuracy Integer networks are very
    // difficult to train directly This is due to many of the training
    // operations mathematically relying on floating-point numbers
    // (probabilities) As such, training is typically done using a
    // floating-point network which is then converted Several helper functions
    // are provided to allow conversion from a floating-point network

    // ***
    // Conversion operations
    // ***

#if ENABLE_WEIGHT_CONVERSION
    // To convert floating-point intputs to integers of the defined bit-depth,
    // use convertFPInputs
    nn.convertFPInputs(input_file, int_input_file);

    // To convert saved weights from a floating-point network for use with an
    // integer one, use convertFPWeights
    nn.convertFPWeights(weights_file, int_weights_file);

    // Because the best activation functions tend to rely on floating-point
    // operations, the activation function is usually saved in memory from a
    // pre-computed table The table may be computed according to the defined
    // bit-depth with the buildActivationTable function
    nn.buildActivationTable(activation_file);
#endif

    // ***
    // Standard operations
    // ***

    // Loading integer saved weights
    nn.loadWeights(int_weights_file);

    // Loading pre-computed activation table
    nn.loadActivationTable(activation_file);

    // Prepare to load test data
    int num_data = 5000;

    Eigen::MatrixXi input(400, num_data);
    Eigen::VectorXi output(num_data);

    // Load test data
    fstream inputs, outputs;
    inputs.open(int_input_file, ios::in);
    outputs.open(output_file, ios::in);

    for (int i = 0; i < num_data; i++) {
        for (int j = 0; j < 400; j++)
            inputs >> input(j, i);
        outputs >> output(i);
    }

    inputs.close();
    outputs.close();

    // ***
    // Testing operations
    // ***

    // Test overall integer network's accuracy
    int correct = 0;
    double accuracy = 0.0;

#if ENABLE_TRACING
    // Open trace file on append mode
    ofstream trace;
    trace.open(trace_file, ios_base::app);
#endif

#if ENABLE_PARSEC_HOOKS
    __parsec_roi_begin();
#endif
    for (int i = 0; i < num_data; i++) {
        if (nn.classify(input.col(i)) == output(i)) {
            correct++;
        }

#if ENABLE_TRACING
        trace << "===================================\n"
              << "==== Sample " << i << endl;

        nn.dumpTrace(trace);

        trace << endl;
#endif
    }
#if ENABLE_PARSEC_HOOKS
    __parsec_roi_end();
#endif

    accuracy = (double)correct / (double)num_data;
    cout << "Accuracy: " << accuracy << endl;

    // Testing some outputs
    cout << "Value: " << output[30]
         << ", Result: " << nn.classify(input.col(30)) << endl;
    cout << "Value: " << output[829]
         << ", Result: " << nn.classify(input.col(829)) << endl;
    cout << "Value: " << output[1300]
         << ", Result: " << nn.classify(input.col(1300)) << endl;
    cout << "Value: " << output[3670]
         << ", Result: " << nn.classify(input.col(3670)) << endl;
    cout << "Value: " << output[4800]
         << ", Result: " << nn.classify(input.col(4800)) << endl;

    // ***
    // Cleanup
    // ***

#if ENABLE_TRACING
    trace.close();
#endif

#ifdef ENABLE_PARSEC_HOOKS
    __parsec_bench_end();
#endif

    return 0;
}
