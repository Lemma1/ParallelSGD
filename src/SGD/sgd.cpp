#include "sgd.h"

sgdBasic::sgdBasic (ConfReader *confReader, int paramSize) {
	m_nParamSize = paramSize;
	m_learningRate = confReader->getFloat("learning rate");
}

sgdBasic::~sgdBasic () {
	// nothing to do for basic sgd
}

void sgdBasic::updateParams (float *params, float *grad) {
	for (int i=0; i<m_nParamSize; i++) {
		params[i] -= m_learningRate * grad[i];
	}
}