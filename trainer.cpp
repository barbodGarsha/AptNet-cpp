#include "AptNet.h"

Trainer::Trainer(NeuralNet neural_network)
{
	if (neural_network.is_ready())
	{
		net = neural_network;
		training_biases = new MatrixXf[net.hidden_layers_len + 1];
		training_weights = new MatrixXf[net.hidden_layers_len + 1];
		training_layers = new MatrixXf[net.hidden_layers_len + 2];

		new_biases = new MatrixXf[net.hidden_layers_len + 1];
		new_weights = new MatrixXf[net.hidden_layers_len + 2];

		ready = 1;
	}
	else
	{
		// TODO: 
	}
}

Trainer::Trainer()
{
}

NeuralNet Trainer::get_network()
{
	return net;
}

void Trainer::backpropagation(float cost_p, int output_index)
{
	MatrixXf *z = new MatrixXf[net.hidden_layers_len + 1];
	for (int i = 0; i < net.hidden_layers_len + 1; i++)
	{
		z[i] = d_cost_p_weights(i, cost_p, output_index);
	}
	
	for (int i = 0; i < net.hidden_layers_len + 1; i++)
	{
		for (int x = 0; x < training_layers[i].cols(); x++)
		{
			for (int y = 0; y < training_layers[i + 1].cols(); y++)
			{
				new_weights[i](x, y) += (-learning_rate * z[i](x, y)) / net.output_index;
			}
		}
	}

	z = new MatrixXf[net.hidden_layers_len + 1];
	for (int i = 0; i < net.hidden_layers_len + 1; i++)
	{
		z[i] = d_cost_p_biases(i, cost_p, output_index);
	}

	for (int i = 0; i < net.hidden_layers_len + 1; i++)
	{
		for (int j = 0; j < training_biases[i].rows(); j++)
		{
			new_biases[i](0, j) += (-learning_rate * z[i](0, j)) / net.output_index;
		}
	}

}

void Trainer::update()
{
	for (int i = net.hidden_layers_len; i >= 0; i--)
	{
		MatrixXf x(training_weights[i].cols(), training_weights[i].rows());
		for (int j = 0; j < x.rows(); j++)
		{
			for (int k = 0; k < x.cols(); k++)
			{
				x(j, k) = training_weights[i](k, j) + new_weights[i](k, j);
			}
		}
		net.weights[net.hidden_layers_len - i] = x;
	}

	for (int i = net.hidden_layers_len; i >= 0; i--)
	{
		net.biases[net.hidden_layers_len - i] = training_biases[i] + new_biases[i];
	}

}

void Trainer::train()
{
	if (ready)
	{
		std::cout << std::endl;
		for (uint32_t i = 0; i < learning_cycle; i++)
		{
			float c = 0.0f;
			for (uint32_t j = 0; j < net.training_samples_n; j++)
			{
				std::cout << "Progress: " << (((i*net.training_samples_n) + j) * 100) / (learning_cycle * net.training_samples_n) << "%" << "\r";
				net.set_inputs(net.training_samples[j].inputs);
				net.feedforward();
				create_training_net();
				for (int k = 0; k < net.output_index; k++)
				{
					backpropagation(calculate_cost_p(net.training_samples[j].outputs(0, k), net.output_values(0, k)), k);
					c += calculate_cost(net.training_samples[j].outputs(0, k), net.output_values(0, k));
				}
				update();
			}
		}
		std::cout << "Progress: 100% \r";
		std::cout << std::endl;
		std::cout << std::endl << "training is finished " << std::endl;
	}
	else
	{
		//TODO: 
	}
}

void Trainer::create_training_net()
{

	training_layers[0] = net.output_values;
	for (int i = net.hidden_layers_len - 1; i >= 0; i--)
	{
		training_layers[(net.hidden_layers_len) - i] = net.hidden_values[i];
	}
	training_layers[net.hidden_layers_len + 1] = net.input_values;

	for (int i = net.hidden_layers_len; i >= 0; i--)
	{
		MatrixXf x(net.weights[i].cols(), net.weights[i].rows());
		for (int j = 0; j < x.rows(); j++)
		{
			for (int k = 0; k < x.cols(); k++)
			{
				x(j, k) = net.weights[i](k, j);
			}
		}
		training_weights[net.hidden_layers_len - i] = x;
		new_weights[net.hidden_layers_len - i] = MatrixXf::Zero(x.rows(), x.cols());
	}

	for (int i = net.hidden_layers_len; i >= 0; i--)
	{
		training_biases[net.hidden_layers_len - i] = net.biases[i];
		new_biases[net.hidden_layers_len - i] = MatrixXf::Zero(1, net.biases[i].cols());
	}

	training_activation_fp_values = new MatrixXf[net.hidden_layers_len + 1];
	for (int i = net.hidden_layers_len; i >= 0; i--)
	{
		training_activation_fp_values[net.hidden_layers_len - i] = MatrixXf(net.raw_values[i].rows(), net.raw_values[i].cols());
		activation_f::apply_activation_fp(net.raw_values[i], training_activation_fp_values[net.hidden_layers_len - i], net.activations_functions[i]);
	}

}

float Trainer::calculate_cost(float target, float prediction)
{
	return pow((prediction - target), 2) / 2;
}

float Trainer::calculate_cost_p(float target, float prediction)
{
	return prediction - target;
}

MatrixXf Trainer::db_pre_layer(int layer_index)
{
	MatrixXf x(training_layers[layer_index].cols(), training_layers[layer_index + 1].cols());
	for (int i = 0; i < x.rows(); i++)
	{
		for (int j = 0; j < x.cols(); j++)
		{
			x(i, j) = training_activation_fp_values[layer_index](0, i) * training_weights[layer_index](i, j);
		}
	}
	return x;
}

MatrixXf Trainer::db_pre_weights(int weights_index)
{
	MatrixXf x(training_layers[weights_index].cols(), training_layers[weights_index + 1].cols());
	for (int i = 0; i < training_layers[weights_index].cols(); i++)
	{
		for (int j = 0; j < training_layers[weights_index + 1].cols(); j++)
		{
			x(i, j) = training_activation_fp_values[weights_index](0, i) * training_layers[weights_index + 1](0, j);
		}
	}
	return x;
}

MatrixXf Trainer::d_cost_p_weights(int weights_index, float cost_p, int output_index)
{
	
	if (weights_index == 0)
	{
		MatrixXf z, p = db_pre_weights(0);
		z = MatrixXf::Zero(p.rows(), p.cols());
		for (int i = 0; i < p.cols(); i++)
		{
			z(output_index, i) = cost_p * p(output_index, i);
		}
		return z;
	}

	MatrixXf z, p = db_pre_layer(0);
	z = MatrixXf::Zero(p.rows(), p.cols());
	for (int i = 0; i < p.cols(); i++)
	{
		z(output_index, i) = cost_p * p(output_index, i);
	}
	for (int i = 1; i < weights_index; i++)
	{
		z *= db_pre_layer(i);
	}
	MatrixXf x(training_layers[weights_index].cols(), training_layers[weights_index + 1].cols());
	for (int i = 0; i < z.cols(); i++)
	{
		for (int j = 0; j < training_layers[weights_index + 1].cols(); j++)
		{
			x(i, j) = z(output_index,i) * db_pre_weights(weights_index)(i, j);
		}
	}
	return x;
}

MatrixXf Trainer::d_cost_p_biases(int biases_index, float cost_p, int output_index)
{

	if (biases_index == 0)
	{
		MatrixXf z;
		z = MatrixXf::Zero(1, net.output_index);
		z(0, output_index) = 1;
		return cost_p * z;
	}
	MatrixXf z, p = db_pre_layer(0);
	z = MatrixXf::Zero(p.rows(), p.cols());
	for (int i = 0; i < p.cols(); i++)
	{
		z(output_index, i) = cost_p * p(output_index, i);
	}
	for (int i = 1; i < biases_index; i++)
	{
		z *= db_pre_layer(i);
	}

	MatrixXf x(1, training_biases[biases_index].cols());
	for (int i = 0; i < z.cols(); i++)
	{
		for (int j = 0; j < x.cols(); j++)
		{
			x(0, j) = z(0, i);
		}
	}
	return x;
}