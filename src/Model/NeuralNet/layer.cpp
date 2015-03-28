#include "layer.h"

/****************************************************************
* Method definition for linearLayer
****************************************************************/

linearLayer::linearLayer (int numNeuron) {
	m_numNeuron = numNeuron;
	m_activation = new float[m_numNeuron];
	m_delta = new float[m_numNeuron];
}

linearLayer::~linearLayer () {
	if (!m_activation) {
		delete [] m_activation;
	}
	if (!m_delta) {
		delete [] m_delta;
	}
}

void linearLayer::activateFunc (float *input) {
	for (int i=0; i<m_numNeuron; ++i) {
		m_activation[i] = input[i];
	}
}

void linearLayer::computeDelta (float *error) {
	for (int i=0; i<m_numNeuron; ++i) {
		m_delta[i] = error[i];
	}
}

/****************************************************************
* Method definition for softmaxLayer
****************************************************************/

softmaxLayer::softmaxLayer (int numNeuron) {
	m_numNeuron = numNeuron;
	m_activation = new float[m_numNeuron];
	m_delta = new float[m_numNeuron];
}

softmaxLayer::~softmaxLayer () {
	if (!m_activation) {
		delete [] m_activation;
	}
	if (!m_delta) {
		delete [] m_delta;
	}
}

void softmaxLayer::activateFunc (float *input) {
	// Pairwise exp
	float sumActivation = 0.f;
	for (int i=0; i<m_numNeuron; ++i) {
		m_activation[i] = exp(input[i]);
		sumActivation += m_activation[i];
	}

	// Normalization
	for (int i=0; i<m_numNeuron; ++i) {
		m_activation[i] /= sumActivation;
	}
}

void softmaxLayer::computeDelta (float *target) {
	for (int i=0; i<m_numNeuron; ++i) {
		m_delta[i] = m_activation[i] - target[i];
	}
}

/****************************************************************
* Method definition for softmaxLayer
****************************************************************/

sigmoidLayer::sigmoidLayer(int numNeuron) {
	m_numNeuron = numNeuron;
	m_activation = new float[m_numNeuron];
	m_delta = new float[m_numNeuron];
}

sigmoidLayer::~sigmoidLayer() {
	if (!m_activation) {
		delete [] m_activation;
	}
	if (!m_delta) {
		delete [] m_delta;
	}
}

void sigmoidLayer::activateFunc (float *input) {
	for (int i=0; i<m_numNeuron; ++i) {
		m_activation[i] = 1 / (1 + exp(-input[i]));
	}
}

void sigmoidLayer::computeDelta (float *error) {
	for (int i=0; i<m_numNeuron; ++i) {
		m_delta[i] = error[i] * m_activation[i] * (1 - m_activation[i]);
	}
}

