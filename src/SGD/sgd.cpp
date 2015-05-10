#include "sgd.h"

sgdBasic::sgdBasic (ConfReader *confReader, int paramSize) {
	m_stepCount  = 0;
	m_nParamSize = paramSize;	
	m_learningRate = confReader->getFloat("learning rate");
	m_useMomentum  = confReader->getInt("use momentum");
}

sgdBasic::~sgdBasic () {
	// nothing to do for basic sgd
}

void sgdBasic::updateParams (float *params, float *grad, int rank) {
	m_stepCount += 1;

	for (int i=0; i<m_nParamSize; i++) {
		params[i] -= m_learningRate / sqrt(m_stepCount) * grad[i];
	}
}