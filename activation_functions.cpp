#include "AptNet.h"

//... Sigmoid ...........................................................................................
float sigmoid(float x)
{
	return 1 / float(1 + exp(-x));
}

float sigmoid_p(float x)
{
	return sigmoid(x) * (1 - sigmoid(x));
}

int sigmoid_array(MatrixXf ref_array, MatrixXf &destination)
{
	for (int i = 0; i < ref_array.cols(); i++)
	{
		destination(0, i) = sigmoid(ref_array(0, i));
	}
	return 0;
}

int sigmoid_p_array(MatrixXf ref_array, MatrixXf &destination)
{
	for (int i = 0; i < ref_array.cols(); i++)
	{
		destination(0, i) = sigmoid_p(ref_array(0, i));
	}
	return 0;
}
//... Other functions ...........................................................................................


namespace activation_f 
{
	int apply_activation_f(MatrixXf &ref_array, int activation_funtion)
	{
		switch (activation_funtion)
		{
		case ActivationFunctions::SIGMOID:
			sigmoid_array(ref_array, ref_array);
			break;
		default:
			break;
		}
		return 0;
	}
	int apply_activation_f(MatrixXf ref_array, MatrixXf &destination, int activation_funtion)
	{
		switch (activation_funtion)
		{
		case ActivationFunctions::SIGMOID:
			sigmoid_array(ref_array, destination);
			break;
		default:
			break;
		}
		return 0;
	}
	
	int apply_activation_fp(MatrixXf &ref_array, int activation_funtion)
	{
		switch (activation_funtion)
		{
		case ActivationFunctions::SIGMOID:
			sigmoid_p_array(ref_array, ref_array);
			break;
		default:
			break;
		}
		return 0;
	}
	int apply_activation_fp(MatrixXf ref_array, MatrixXf &destination, int activation_funtion)
	{
		switch (activation_funtion)
		{
		case ActivationFunctions::SIGMOID:
			sigmoid_p_array(ref_array, destination);
			break;
		default:
			break;
		}
		return 0;
	}
}