#include "sgd.h"

#include <math.h>

DelayedAdadelta::DelayedAdadelta (ConfReader *confReader, int paramSize) {
	m_nParamSize = paramSize;	
	m_decayFactor = confReader->getFloat("adadelta decay factor");
	m_stableConst = confReader->getFloat("adadelta stable const");
	m_useMomentum  = confReader->getInt("use momentum");
	int nProc;
    MPI_Comm_size(MPI_COMM_WORLD, &nProc);
    m_numSlave = nProc - 1;
	m_ESquareGrad  = new float [m_nParamSize];
	m_ESquareDelta = new float [m_nParamSize];

	memset(m_ESquareGrad, 0x00, sizeof(float) * m_nParamSize);
	memset(m_ESquareDelta, 0x00, sizeof(float) * m_nParamSize);

	//initialize grad map
    for (int i=1; i<=m_numSlave; i++) {
    	float *histSquareGrad = new float [m_nParamSize];
    	for (int j=0; j<m_nParamSize; j++) {
			histSquareGrad[j] = 0.1f;
		}
    	m_mapHistSquareGrad[i] = histSquareGrad;
    }
    //Initialize delta map
    for (int i=1; i<=m_numSlave; i++) {
    	float *histSquareDelta = new float [m_nParamSize];
    	for (int j=0; j<m_nParamSize; j++) {
			histSquareDelta[j] = 0.1f;
		}
    	m_mapHistSquareDelta[i] = histSquareDelta;
    }    	
}

DelayedAdadelta::~DelayedAdadelta () {
	if (!m_ESquareGrad) {
		delete [] m_ESquareGrad;
	}
	if (!m_ESquareDelta) {
		delete [] m_ESquareDelta;
	}
	for (int i=1; i<=m_numSlave; ++i) {    	
    	delete [] m_mapHistSquareGrad[i];
    }
    for (int i=1; i<=m_numSlave; ++i) {    	
    	delete [] m_mapHistSquareDelta[i];
    }
}

void DelayedAdadelta::updateParams (float *params, float *grad, int rank) {
	float delta;
	//printf("Start updateParams\n");
	for (int i=0; i<m_nParamSize; i++) {
		// accumulate mean squared grad
		m_ESquareGrad[i] = m_decayFactor * m_ESquareGrad[i] + (1 - m_decayFactor) * grad[i] * grad[i];
		// compute delta
		delta = sqrt(m_mapHistSquareGrad[rank][i] + m_stableConst) / sqrt(m_mapHistSquareGrad[rank][i] + m_stableConst) * grad[i];
		params[i] -= delta;
		// accumulate mean squared delta
		m_ESquareDelta[i] = m_decayFactor * m_ESquareDelta[i] + (1 - m_decayFactor) * delta * delta;
	}
	memcpy(m_mapHistSquareGrad[rank], m_ESquareGrad, sizeof(float) * m_nParamSize);
	memcpy(m_mapHistSquareDelta[rank], m_ESquareDelta, sizeof(float) * m_nParamSize);
	//printf("Finish updateParams\n");
}