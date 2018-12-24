#include "AptNet.h"


void add_biases(MatrixXf &x, MatrixXf biases)
{
	for (int i = 0; i < x.cols(); i++)
	{
		x(0, i) += biases(0, i);
	}
}

//..............................................................................................



// TODO: handle errors NOTE: later with error_handler.cpp/.h
NeuralNet::NeuralNet(std::string layer_scheme)
{
	int counter = 0;
	for (uint16_t i = 0; i < layer_scheme.length(); i++)
	{
		if(layer_scheme[i] == ',')
		{
			counter++;
		}
	}
	hidden_layers_len = counter - 1;
	finish(layer_scheme);
	hidden_values = new MatrixXf[hidden_layers_len];

	generate_weights();
	generate_biases();



	raw_values = new MatrixXf[hidden_layers_len + 1];
}

NeuralNet::NeuralNet()
{

}

// TODO: improve the importing system
void NeuralNet::set_activations_funtions(std::string activation_functions_scheme)
{
	activations_functions = new int[hidden_layers_len + 1];
	std::string num = "";
	for (int j = 0; j < hidden_layers_len + 1; j++)
	{
		num = "";
		for (uint16_t i = 0; i < activation_functions_scheme.length(); i++)
		{
			if (activation_functions_scheme[i] == ',')
			{
				activations_functions[j] = std::stoi(num);
				activation_functions_scheme[i] = ' ';
				break;
			}
			num += activation_functions_scheme[i];
			activation_functions_scheme[i] = ' ';
		}
	}
}

void NeuralNet::feedforward()
{
	if (hidden_layers_len != 0)
	{
		
		hidden_values[0] = input_values * weights[0];
		add_biases(hidden_values[0], biases[0]);
		raw_values[0] = hidden_values[0];
		activation_f::apply_activation_f(hidden_values[0], activations_functions[0]);
		if (hidden_layers_len > 1)
		{
			for (int i = 1; i < hidden_layers_len; i++)
			{
				hidden_values[i] = hidden_values[i - 1] * weights[i];
				add_biases(hidden_values[i], biases[i]);
				raw_values[i] = hidden_values[i];
				activation_f::apply_activation_f(hidden_values[i], activations_functions[i]);
			}
			output_values = hidden_values[hidden_layers_len - 1] * weights[hidden_layers_len];
			add_biases(output_values, biases[hidden_layers_len]);
			raw_values[hidden_layers_len] = output_values;
			activation_f::apply_activation_f(output_values, activations_functions[hidden_layers_len]);
		}
		else
		{
			output_values = hidden_values[0] * weights[1];
			add_biases(output_values, biases[1]);
			raw_values[1] = output_values;
			activation_f::apply_activation_f(output_values, activations_functions[1]);
		}
	}
	else
	{
		output_values = input_values * weights[0];
		add_biases(output_values, biases[0]);
		raw_values[0] = output_values;
		activation_f::apply_activation_f(output_values, activations_functions[0]);
	}
}

void NeuralNet::set_inputs(MatrixXf inputs)
{
	if (input_index == inputs.cols())
	{
		input_values = inputs;
	}
}


void NeuralNet::set_training_samples(TrainingSmaple samples[], int size)
{	// TODO: check the samples
	training_samples_n = size;
	training_samples = new TrainingSmaple[size];
	for (int i = 0; i < size; i++)
	{
		training_samples[i] = samples[i];
	}
}

void NeuralNet::generate_weights()
{
	weights = new MatrixXf[hidden_layers_len + 1];

	if (hidden_layers_len != 0)
	{
		weights[0] = MatrixXf::Random(input_index, hidden_index[0]);
		//std::cout << weights[0];
		if (hidden_layers_len > 1)
		{
			for (int i = 0; i < hidden_layers_len-1; i++)
			{
				weights[i + 1] = MatrixXf::Random(hidden_index[i], hidden_index[i + 1]);
			}
			weights[hidden_layers_len] = MatrixXf::Random(hidden_index[hidden_layers_len - 1], output_index);
		}
		else
		{
			weights[hidden_layers_len] = MatrixXf::Random(hidden_index[hidden_layers_len - 1], output_index);
		}
	}
	else
	{
		weights[0] = MatrixXf::Random(input_index, output_index);
	}

}

void NeuralNet::generate_biases()
{
	biases = new MatrixXf[hidden_layers_len + 1];
	for (int i = 0; i < hidden_layers_len; i++)
	{
		biases[i] = MatrixXf::Random(1, hidden_index[i]);
	}
	biases[hidden_layers_len] = MatrixXf::Random(1, output_index);
}

void NeuralNet::finish(std::string layer_scheme)
{
	std::string num = "";
	for (uint16_t i = 0; i < layer_scheme.length(); i++)
	{
		if (layer_scheme[i] == ',')
		{
			input_index = std::stoi(num);
			layer_scheme[i] = ' ';
			break;
		}
		num += layer_scheme[i];
		layer_scheme[i] = ' ';
	}

	hidden_index = new int[hidden_layers_len];
	num = "";
	for (int j = 0; j < hidden_layers_len; j++)
	{
		for (uint16_t i = 0; i < layer_scheme.length(); i++)
		{
			if (layer_scheme[i] == ',')
			{
				layer_scheme[i] = ' ';
				break;
			}
			num += layer_scheme[i];
			layer_scheme[i] = ' ';
		}
		hidden_index[j] = std::stoi(num);
		num = "";
	}

	output_index = std::stoi(layer_scheme);

	
}

int NeuralNet::is_ready()
{
	if (ready) { return 1; }
	return 0;
}