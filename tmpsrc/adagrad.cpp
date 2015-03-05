#include <math.h>
#include <string.h>
#include "sgd.h"



adagrad::adagrad (int paramSize, float learningRate) {
	m_nParamSize = paramSize;
	m_learningRate = learningRate;

	m_histGradSquare = new float [m_nParamSize];
	for (int i=0; i<m_nParamSize; i++) {
		m_histGradSquare[i] = 0.1f;
	}
}

adagrad::~adagrad () {
	if (!m_histGradSquare) {
		delete [] m_histGradSquare;
	}
}

void adagrad::updateParams (float *params, float *grad) {
	for (int i=0; i<m_nParamSize; i++) {
		m_histGradSquare[i] += grad[i] * grad[i];
		params[i] -= m_learningRate * grad[i] / sqrt(m_histGradSquare[i]);		
	}
}