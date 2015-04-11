#include <math.h>
#include <string.h>
#include "sgd.h"

futureAdagrad::futureAdagrad (ConfReader *confReader, int paramSize) {
	m_nParamSize = paramSize;
	m_learningRate = confReader->getFloat("learning rate");
	m_useMomentum  = confReader->getInt("use momentum");

	m_histSquareGrad = new float [m_nParamSize];
	for (int i=0; i<m_nParamSize; i++) {
		m_histSquareGrad[i] = 0.1f;
	}

	int nProc;
    MPI_Comm_size(MPI_COMM_WORLD, &nProc);
    m_nSlave = nProc - 1;

    for (int i=1; i<=m_nSlave; ++i) {
    	float *histSquareGrad = new float [m_nParamSize];
    	for (int j=0; j<m_nParamSize; j++) {
			histSquareGrad[j] = 0.1f;
		}
    	m_mapHistSquareGrad[i] = histSquareGrad;
    }
}

futureAdagrad::~futureAdagrad () {
	if (!m_histSquareGrad) {
		delete [] m_histSquareGrad;
	}
	for (int i=1; i<=m_nSlave; ++i) {    	
    	delete [] m_mapHistSquareGrad[i];
    }
}

void futureAdagrad::updateParams (float *params, float *grad, int rank) {
	for (int i=0; i<m_nParamSize; i++) {
		m_histSquareGrad[i] += grad[i] * grad[i];
		m_mapHistSquareGrad[rank][i] = m_histSquareGrad[i] - m_mapHistSquareGrad[rank][i];
		params[i] -= m_learningRate * grad[i] / sqrt(m_mapHistSquareGrad[rank][i]);
	}
	printInfo(m_mapHistSquareGrad[rank]);
	memcpy(m_mapHistSquareGrad[rank], m_histSquareGrad, sizeof(float) * m_nParamSize);
}