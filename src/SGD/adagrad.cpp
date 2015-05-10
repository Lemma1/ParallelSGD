#include <string.h>
#include "sgd.h"

adagrad::adagrad (ConfReader *confReader, int paramSize) {
	m_nParamSize = paramSize;
	m_learningRate = confReader->getFloat("learning rate");
	m_useMomentum  = confReader->getInt("use momentum");
	m_stepCount = 0;

	m_histSquareGrad = new float [m_nParamSize];
	for (int i=0; i<m_nParamSize; i++) {
		m_histSquareGrad[i] = 1.f;
	}
}

adagrad::~adagrad () {
	if (!m_histSquareGrad) {
		delete [] m_histSquareGrad;
	}
}

void adagrad::updateParams (float *params, float *grad, int rank) {
	m_stepCount += 1;
	
	// printf("step[%d]: rank %d\n", m_stepCount, rank);

	for (int i=0; i<m_nParamSize; i++) {
		m_histSquareGrad[i] += grad[i] * grad[i];
		params[i] -= m_learningRate * grad[i] / sqrt(m_histSquareGrad[i]);
	}
	
	// float sum = 0.f;
	// for (int i=0; i<m_nParamSize; i++) {
	// 	m_histSquareGrad[i] += grad[i] * grad[i];
	// 	sum += sqrt(m_histSquareGrad[i]);
	// }
		
	// float rate = m_learningRate * sum / sqrt(sqrt(m_stepCount));
	// printf("step[%d]: sum %f, rate %f, m_learningRate %f\n", m_stepCount, sum, rate, m_learningRate);
	
	// for (int i=0; i<m_nParamSize; i++) {
	// 	params[i] -= rate * grad[i] / sqrt(m_histSquareGrad[i]);
	// }
}