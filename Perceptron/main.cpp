#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "Perceptron.h"
#include <cmath>

bool ReadData(const std::string& filename, std::vector<double>& data)
{
	std::ifstream file(filename.c_str());

	if (!file.is_open())
		return false;

	data.clear();

	double value;
	while (file.good())
	{
		file >> value;

		if (!file.fail())
			data.push_back(value);
	}

	file.close();

	return true;
}

int main()
{
	std::vector<double> inputData;

	ReadData("data.csv", inputData);

	std::ofstream errorlog ("error.csv");

	const unsigned int numInputs = 3;
	Perceptron p(numInputs);
	p.RandomiseWeights();

	unsigned int numIterations = inputData.size() - numInputs;
	unsigned int numEpocs = 1000;

	double learningrate = 0.05;
	double bestError = -1;

	for (unsigned int e = 0; e < numEpocs; ++e)
	{
		double error = 0;

		for (unsigned int i = 0; i < numIterations; ++i)
		{
			for (int w = 0; w < numInputs; ++w)
				p._inputs[w] = inputData[i+w];

			double delta = inputData[i+numInputs] - p.ActivateLogistic();
			error += delta * delta;

			for (unsigned int w = 0; w < numInputs; ++w)
				p._weights[w] += delta * learningrate * inputData[i+w];

			p._bias += delta * learningrate;
		}

		errorlog << error << "\n";
	}

	errorlog.close();

	std::ofstream outputlog ("output.csv");
	for (unsigned int i = 0; i < numIterations; ++i)
	{
		for (int w = 0; w < numInputs; ++w)
			p._inputs[w] = inputData[i+w];

		double output = p.ActivateLogistic();

		outputlog << output << "\n";
	}

	outputlog.close();

	return 0;
}