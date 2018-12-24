#pragma once
class Trainer
{
public:
	NeuralNet net;

	float learning_rate = 0.5f;
	uint32_t learning_cycle = 10;

	Trainer(NeuralNet neural_network);
	Trainer();

	NeuralNet get_network();

	void train();


private:

	int ready = 0;

	MatrixXf *training_layers;
	MatrixXf *training_weights;
	MatrixXf *training_biases;
	MatrixXf *training_activation_fp_values;

	MatrixXf *new_weights;
	MatrixXf *new_biases;



	float calculate_cost(float target, float prediction);
	float calculate_cost_p(float target, float prediction);
	
	void create_training_net();

	void backpropagation(float cost_p, int output_index);

	void update();

	MatrixXf db_pre_layer(int layer_index);
	MatrixXf db_pre_weights(int weights_index);
	MatrixXf d_cost_p_weights(int weights_index, float cost_p, int output_index);
	MatrixXf d_cost_p_biases(int biases_index, float cost_p, int output_index);
};