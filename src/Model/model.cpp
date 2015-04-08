#include "model.h"
#include <time.h>

/****************************************************************
* Method definition for Linear Regression
****************************************************************/

linearReg::linearReg (ConfReader *confReader, int minibatchSize) {
	m_nParamSize = confReader->getInt("parameter size");
	m_nMinibatchSize = minibatchSize;
}

linearReg::~linearReg () {
	// nothing to do here
}

float linearReg::computeGrad (float *grad, float *params, float *data, float *label) {
	// Init variables
	float cost = 0.f;
	float diff, predict;
	int sampleOffset;
	memset(grad, 0x00, sizeof(float) * m_nParamSize);

	// for (int sample=0; sample<m_nMinibatchSize; sample++) {
	// 	sampleOffset = sample * m_nParamSize;
	// 	for (int dim=0; dim<m_nParamSize; dim++) {
	// 		printf("(%d)(%d,%d): %f\n", sampleOffset, sample, dim, data[sampleOffset + dim]);
	// 	}
	// }

	// Accumulate cost and grad
	for (int sample=0; sample<m_nMinibatchSize; sample++) {
		sampleOffset = sample * m_nParamSize;
		predict = 0.f;
		for (int dim=0; dim<m_nParamSize; dim++) {
			predict += params[dim] * data[sampleOffset + dim];
			printf("%d,%d:%f\n",sample, dim, data[sampleOffset + dim]);
		}

		diff = predict - label[sample];
		printf("%f,%f\n",label[sample], predict);
		for (int dim=0; dim<m_nParamSize; dim++) {
			grad[dim] += data[sampleOffset + dim] * diff;
		}
		cost += 0.5 * diff * diff;
	}

	// Average minibatch_cost and grad
	float f_minibatchSize = static_cast<float>(m_nMinibatchSize);
	cost /= f_minibatchSize;
	for (int dim=0; dim<m_nParamSize; dim++) {
		grad[dim] /= f_minibatchSize;
	}

	return cost;
}

void linearReg::initParams (float *params) {
	srand (time(NULL));
	for (int i=0; i<m_nParamSize; i++) {
        params[i] = SYM_UNIFORM_RAND;
    }
}

/****************************************************************
* Method definition for Softmax Classification
****************************************************************/

softmax::softmax (ConfReader *confReader, int minibatchSize) {
	m_nParamSize = confReader->getInt("parameter size");
	m_nClassNum = confReader->getInt("softmax class num");
	m_nMinibatchSize = minibatchSize;
}

softmax::~softmax () {
	// nothing to do here
}

float softmax::computeGrad (float *grad, float *params, float *data, float *label) {
	// Init variables
	float crossEntropy = 0.f;
	float diff;
	float predictProb;
	int predictLabel;
	memset(grad, 0x00, sizeof(float) * m_nParamSize);

	// predictulate cost and grad
	for (int sample=0; sample<m_nMinibatchSize; sample++) {
		predictProb = 0.f;
		for (int dim=0; dim<m_nParamSize; dim++) {
			for (int classIdx=0; classIdx<m_nClassNum; classIdx++) {
				
			}
		}
	}

	// Average minibatch_cost and grad
	float f_minibatchSize = static_cast<float>(m_nMinibatchSize);
	crossEntropy /= f_minibatchSize;
	for (int dim=0; dim<m_nParamSize; dim++) {
		grad[dim] /= f_minibatchSize;
	}

	return crossEntropy;
}