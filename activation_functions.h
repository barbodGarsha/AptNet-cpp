#pragma once
class ActivationFunctions
{
	public:
		static const int SIGMOID = 0;
		static const int TANH = 1;

};

namespace activation_f
{
	int apply_activation_f(MatrixXf &ref_array, int activation_funtion);
	int apply_activation_f(MatrixXf ref_array, MatrixXf &destination, int activation_funtion);

	int apply_activation_fp(MatrixXf &ref_array, int activation_funtion);
	int apply_activation_fp(MatrixXf ref_array, MatrixXf &destination, int activation_funtion);

}
