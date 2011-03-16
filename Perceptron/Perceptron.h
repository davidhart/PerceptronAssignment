#pragma once

#include <vector>

class Perceptron
{
public:
	Perceptron(int numInputs);

	void RandomiseWeights();

	double Sum();
	double ActivateLogistic();
	double ActivateStep();
	
	double _bias;
	std::vector<double> _inputs;
	std::vector<double> _weights;
};