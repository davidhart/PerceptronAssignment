#include "Perceptron.h"

#include <ctime>
#include <cmath>

Perceptron::Perceptron(int numInputs) :
	_inputs(numInputs, 0),
	_weights(numInputs, 0),
	_bias(0)
{
}

void Perceptron::RandomiseWeights()
{
	srand((unsigned int)time(0));

	for (unsigned int i = 0; i < _weights.size(); ++i)
	{
		_weights[i] = (rand() % 10000) / 10000.0;
	}
}
double Perceptron::Sum()
{
	double val = 0;
	for (unsigned int i = 0; i < _inputs.size(); ++ i)
	{
		val += _inputs[i] * _weights[i];
	}
	
	return val;
}

double Perceptron::ActivateLogistic()
{
	return 1.0 / (1 + exp(-(Sum() + _bias)));
}

double Perceptron::ActivateStep()
{
	if (Sum() + _bias < 0)
		return 0;
	else
		return 1;
}