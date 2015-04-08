#include <math.h>
#include <mpi.h>
#include <string.h>
#include "sgd.h"

kernelAdadelta::kernelAdadelta (ConfReader *confReader, int paramSize) {
	m_nParamSize = paramSize;
	m_decayFactor = confReader->getFloat("adadelta decay factor");
	m_stableConst = confReader->getFloat("adadelta stable const");
	m_useMomentum  = confReader->getInt("use momentum");

	m_ESquareGrad  = new float [m_nParamSize];
	m_ESquareDelta = new float [m_nParamSize];

	memset(m_ESquareGrad, 0x00, sizeof(float) * m_nParamSize);
	memset(m_ESquareDelta, 0x00, sizeof(float) * m_nParamSize);

	int nProc;
    MPI_Comm_size(MPI_COMM_WORLD, &nProc);
    m_nSlave = nProc - 1;

    for (int i=1; i<=m_nSlave; ++i) {
    	float *ESquareGrad = new float [m_nParamSize];
    	float *ESquareDelta = new float [m_nParamSize];
    	m_mapESquareGrad[i] = ESquareGrad;
    	m_mapESquareDelta[i] = ESquareDelta;
    	m_factor[i] = 1 - m_decayFactor;
    }
}

kernelAdadelta::~kernelAdadelta () {
	if (!m_ESquareGrad) {
		delete [] m_ESquareGrad;
	}
	if (!m_ESquareDelta) {
		delete [] m_ESquareDelta;
	}
	for (int i=1; i<=m_nSlave; ++i) {
		if (!m_mapESquareGrad[i]) {
    		delete [] m_mapESquareGrad[i];
    	}
    	if (!m_mapESquareDelta[i]) {
    		delete [] m_mapESquareDelta[i];
    	}
    }    
}

void kernelAdadelta::updateParams (float *params, float *grad, int rank) {
	float delta;		

	for (int slaveId=1; slaveId<=m_nSlave; slaveId++) {
		m_factor[slaveId] *= m_decayFactor;
	}

	for (int i=0; i<m_nParamSize; i++) {
		float gradSqr_i = grad[i] * grad[i];
		// accumulate mean squared grad
		m_ESquareGrad[i] = m_decayFactor * m_ESquareGrad[i] + (1 - m_decayFactor) * gradSqr_i;
		for (int slaveId=1; slaveId<=m_nSlave; slaveId++) {		
			m_mapESquareGrad[slaveId][i] += m_factor[slaveId] * gradSqr_i;
			m_mapESquareGrad[slaveId][i] *= 0.8;
		}

		// compute delta
		delta = sqrt(m_mapESquareDelta[rank][i] + m_stableConst) / sqrt(m_mapESquareGrad[rank][i] + m_stableConst) * grad[i];
		params[i] -= delta;

		// accumulate mean squared delta
		float deltaSqr_i = delta * delta;
		m_ESquareDelta[i] = m_decayFactor * m_ESquareDelta[i] + (1 - m_decayFactor) * deltaSqr_i;
		for (int slaveId=1; slaveId<=m_nSlave; slaveId++) {		
			m_mapESquareDelta[slaveId][i] += m_factor[slaveId] * deltaSqr_i;
			m_mapESquareDelta[slaveId][i] *= 0.8;
		}
	}

	memcpy(m_mapESquareGrad[rank], m_ESquareGrad, sizeof(float)*m_nParamSize);
	memcpy(m_mapESquareDelta[rank], m_ESquareDelta, sizeof(float)*m_nParamSize);
	m_factor[rank] = (1 - m_decayFactor);
}