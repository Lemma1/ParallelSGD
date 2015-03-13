#include <math.h>
#include <string.h>
#include "sgd.h"

adagrad::adagrad (int paramSize, float learningRate) {
	m_nParamSize = paramSize;
	m_learningRate = learningRate;

	m_histSquareGrad = new float [m_nParamSize];
	for (int i=0; i<m_nParamSize; i++) {
		m_histSquareGrad[i] = 0.1f;
	}
}

adagrad::~adagrad () {
	if (!m_histSquareGrad) {
		delete [] m_histSquareGrad;
	}
}

void adagrad::updateParams (float *params, float *grad) {
	for (int i=0; i<m_nParamSize; i++) {
		m_histSquareGrad[i] += grad[i] * grad[i];
		params[i] -= m_learningRate * grad[i] / sqrt(m_histSquareGrad[i]);		
	}
}