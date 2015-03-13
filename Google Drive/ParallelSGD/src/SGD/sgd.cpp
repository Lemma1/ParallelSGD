#include "sgd.h"

sgdBasic::sgdBasic (int paramSize, float learningRate) {
	m_nParamSize = paramSize;
	m_learningRate = learningRate;
}

sgdBasic::~sgdBasic () {
	// nothing to do for basic sgd
}

void sgdBasic::updateParams (float *params, float *grad) {
	for (int i=0; i<m_nParamSize; i++) {
		params[i] -= m_learningRate * grad[i];
	}
}