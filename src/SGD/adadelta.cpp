#include <math.h>
#include <string.h>
#include "sgd.h"

adadelta::adadelta (int paramSize, float decayFactor, float stableConst) {
	m_nParamSize = paramSize;	
	m_decayFactor = decayFactor;
	m_stableConst = stableConst;

	m_EGradSquare  = new float [m_nParamSize];
	m_EDeltaSquare = new float [m_nParamSize];

	memset(m_EGradSquare, 0x00, sizeof(float) * m_nParamSize);
	memset(m_EDeltaSquare, 0x00, sizeof(float) * m_nParamSize);
}

adadelta::~adadelta () {
	if (!m_EGradSquare) {
		delete [] m_EGradSquare;
	}
	if (!m_EDeltaSquare) {
		delete [] m_EDeltaSquare;
	}
}

void adadelta::updateParams (float *params, float *grad) {
	float delta;
	for (int i=0; i<m_nParamSize; i++) {
		// accumulate mean squared grad
		m_EGradSquare[i] = m_decayFactor * m_EGradSquare[i] + (1 - m_decayFactor) * grad[i] * grad[i];
		// compute delta
		delta = sqrt(m_EDeltaSquare[i] + m_stableConst) / sqrt(m_EGradSquare[i] + m_stableConst) * grad[i];
		params[i] -= delta;
		// accumulate mean squared delta
		m_EDeltaSquare[i] = m_decayFactor * m_EDeltaSquare[i] + (1 - m_decayFactor) * delta * delta;
	}
}