#pragma once
class TrainingSmaple
{
public:
	MatrixXf inputs;
	MatrixXf outputs;
};

class NeuralNet
{
public:
	int input_index;
	int *hidden_index;
	int output_index;

	MatrixXf input_values;
	MatrixXf *hidden_values;
	MatrixXf output_values;

	MatrixXf *raw_values;

	uint32_t training_samples_n;

	int *activations_functions;

	int hidden_layers_len;

	MatrixXf *weights;
	MatrixXf *biases;

	TrainingSmaple *training_samples;

	NeuralNet(std::string layers_scheme);
	NeuralNet();

	void set_activations_funtions(std::string activation_functions_scheme);

	void set_inputs(MatrixXf inputs);

	void feedforward();
	

	void set_training_samples(TrainingSmaple samples[],int size);

	int is_ready();
	
private:
	// TODO: fix this 
	int ready = 1;

	void generate_weights();
	void generate_biases();

	void finish(std::string layer_scheme);
};



