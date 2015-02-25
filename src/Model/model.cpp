#include "model.h"

/****************************************************************
* Method definition for Linear Regression
****************************************************************/

linearReg::linearReg (int paramSize, int minibatchSize) {
	m_nParamSize = paramSize;
	m_nMinibatchSize = minibatchSize;
}

linearReg::~linearReg () {
	// nothing to do here
}

float linearReg::computeGrad (float *grad, float *params, float *data, float *label) {
	// Init variables
	float cost = 0.f;
	float diff, predict;
	int sampleOffset;
	memset(grad, 0x00, sizeof(float) * m_nParamSize);

	// Accumulate cost and grad
	for (int sample=0; sample<m_nMinibatchSize; sample++) {
		sampleOffset = sample * m_nParamSize;
		predict = 0.f;
		for (int dim=0; dim<m_nParamSize; dim++) {
			predict += params[dim] * data[sampleOffset + dim];
			diff = params[dim] * data[sampleOffset + dim] - label[sample];
			grad[dim] += data[dim] * diff;
		}
		cost += 0.5 * (predict - label[sample]) * (predict - label[sample]);
	}

	// Average minibatch_cost and grad
	float f_minibatchSize = static_cast<float>(m_nMinibatchSize);
	cost /= f_minibatchSize;
	for (int dim=0; dim<m_nParamSize; dim++) {
		grad[dim] /= f_minibatchSize;
	}

	return cost;
}

/****************************************************************
* Method definition for Softmax Classification
****************************************************************/

softmax::softmax (int paramSize, int minibatchSize, int classNum) {
	m_nParamSize = paramSize;
	m_nMinibatchSize = minibatchSize;
	m_nClassNum = classNum;
}

softmax::~softmax () {
	// nothing to do here
}

float softmax::computeGrad (float *grad, float *params, float *data, float *label) {
	// Init variables
	float crossEntropy = 0.f;
	float diff;
	memset(grad, 0x00, sizeof(float) * m_nParamSize);

	// predictulate cost and grad
	for (int sample=0; sample<m_nMinibatchSize; sample++) {
		for (int dim=0; dim<m_nParamSize; dim++) {
			for (int classIdx=0; classIdx<m_nClassNum; classIdx++) {
				
			}
		}
	}

	// Average minibatch_cost and grad
	float f_minibatchSize = static_cast<float>(m_nMinibatchSize);
	crossEntropy /= f_minibatchSize;
	for (int dim=0; dim<m_nParamSize; dim++) {
		grad[dim] /= f_minibatchSize;
	}

	return crossEntropy;
}