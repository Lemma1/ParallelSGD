#include "model.h"
#include "float.h"
#include <time.h>

/****************************************************************
* Method definition for Linear Regression
****************************************************************/

linearReg::linearReg (ConfReader *confReader, int minibatchSize) {	
	m_inputSize = confReader->getInt("input size");
	m_nMinibatchSize = minibatchSize;
	m_nParamSize = m_inputSize + 1;
}

linearReg::~linearReg () {
	// nothing to do here
}

float linearReg::computeGrad (float *grad, float *params, float *data, float *label) {
	// Init variables
	float cost = 0.f;
	float diff, predict;
	int dataOffset;
	memset(grad, 0x00, sizeof(float) * m_nParamSize);

	// Accumulate cost and grad
	for (int sample=0; sample<m_nMinibatchSize; sample++) {
		dataOffset = sample * m_nParamSize;
		predict = 0.f;
		for (int dim=0; dim<m_nParamSize; dim++) {
			predict += params[dim] * data[dataOffset + dim];
		}

		diff = predict - label[sample];
		for (int dim=0; dim<m_nParamSize; dim++) {
			grad[dim] += data[dataOffset + dim] * diff;
		}
		cost += 0.5 * diff * diff;
	}

	// Average minibatch_cost and grad
	float f_minibatchSize = static_cast<float>(m_nMinibatchSize);
	cost /= f_minibatchSize;
	for (int dim=0; dim<m_nParamSize; dim++) {
		grad[dim] /= f_minibatchSize;
	}
	printf("Linear Regression Error: %f\n", cost);

	return cost;
}

void linearReg::initParams (float *params) {
	srand (time(NULL));
	for (int i=0; i<m_nParamSize; i++) {
        params[i] = 0;//SYM_UNIFORM_RAND;
    }
}

/****************************************************************
* Method definition for softmaxReg Classification
****************************************************************/

softmaxReg::softmaxReg (ConfReader *confReader, int minibatchSize) {	
	m_inputSize = confReader->getInt("input size");
	m_classNum = confReader->getInt("softmax class num");
	m_nMinibatchSize = minibatchSize;
	m_prob = new float [m_classNum];
	m_oneOnlabel = new float[m_classNum];
	m_nParamSize = m_classNum * (m_inputSize + 1);
	printf("%d,%d,%d\n", m_inputSize, m_classNum, m_nParamSize);
}

softmaxReg::~softmaxReg () {
	if (m_prob != NULL) delete [] m_prob;
	if (m_oneOnlabel != NULL) delete [] m_oneOnlabel;
}

void softmaxReg::initParams (float *params) {
	srand (time(NULL));
	for (int i=0; i<m_nParamSize; i++) {
        params[i] = 0.000001 * SYM_UNIFORM_RAND;
    }
}

float softmaxReg::computeGrad (float *grad, float *params, float *data, float *label) {
	// Init variables
	float crossEntropy = 0.f;
	float correctCount = 0.f;
	int dataOffset;
	int labelInt;
	memset(grad, 0x00, sizeof(float) * m_nParamSize);
	memset(m_prob, 0x00, sizeof(float) * m_classNum);

	// compute prob, grad and error
	for (int sample=0; sample<m_nMinibatchSize; sample++) {
		//**** turn label into one-on vector ****//
		memset(m_oneOnlabel, 0x00, sizeof(float)*m_classNum);
		labelInt = (int)label[sample];
		if (labelInt < 0) {
			labelInt = 0;
		}
		m_oneOnlabel[labelInt] = 1.f;

		//**** compute prob = <W, x> + b ****//
		dataOffset = sample * m_inputSize;
		float maxProb = -FLT_MIN;
		for (int classIdx=0; classIdx<m_classNum; ++classIdx) {
			// non-bias terms
			for (int dim=0; dim<m_inputSize; ++dim) {
				//printf("Data %d is %f\n", dataOffset+dim, data[dataOffset+dim]);
				m_prob[classIdx] += params[classIdx*m_inputSize+dim] * data[dataOffset+dim];
				// if (m_prob[classIdx] != m_prob[classIdx])
				// {
				// 	printf("Wrong at 1\n");
				// 	exit(-1);
				// }
			}
			//printf("sample %d: m_prob[%d]%f\n", sample, classIdx,m_prob[classIdx]);
			// bias terms
			m_prob[classIdx] += params[m_classNum*m_inputSize+classIdx];
			if (m_prob[classIdx] > maxProb) {
			 	maxProb = m_prob[classIdx];
			}
		}

		//**** compute prob = exp(<W, x> + b) / sum(exp(<W, x> + b))) ****//
		float sumProb = 0.f;
		// compute prob = exp(<W, x> + b), sumProb = sum(exp(<W, x> + b))
		for (int classIdx=0; classIdx<m_classNum; ++classIdx) {
			m_prob[classIdx] = exp(m_prob[classIdx]-maxProb);
			sumProb += m_prob[classIdx];
			//printf("m_prob[%d]: %f, sumProb: %f\n", classIdx, m_prob[classIdx], sumProb);
		}
		if (sumProb <= 0.000001)
		{
			
		}	
		// normalize prob = prob ./ sumProb
		//printf("%f\n", sumProb);
		for (int classIdx=0; classIdx<m_classNum; ++classIdx) {
			if (sumProb <= 0.000001) {
				m_prob[classIdx] = 1.f / m_classNum;
			}
			else {
				m_prob[classIdx] /= sumProb;
			}
			// if (m_prob[classIdx] != m_prob[classIdx])
			// {
			// 	printf("Wrong at 2\n");
			// 	printf("%f\n", sumProb);
			// 	exit(-1);
			// }
			//printf("m_prob[%d]: %f\n", classIdx, m_prob[classIdx]);
		}


		//**** compute grad and error based on normalized prob ****//
		for (int classIdx=0; classIdx<m_classNum; ++classIdx) {
			float diff = m_prob[classIdx] - m_oneOnlabel[classIdx];
			// non-bias terms
			for (int dim=0; dim<m_inputSize; ++dim) {
				grad[classIdx*m_inputSize+dim] += data[dataOffset + dim] * diff;
			}
			// bias terms
			grad[m_classNum*m_inputSize+classIdx] += diff;

			// error
			//printf("%d:%f\n", sample,m_prob[classIdx]);
			crossEntropy -= m_oneOnlabel[classIdx] * log(m_prob[classIdx] + 0.000001);
		}

		// compute some statistics
		float maxP = 0.f;
		int maxIndex = -1;
		for (int i=0; i<m_classNum; i++) {
			if (m_prob[i] > maxP) {
				maxP = m_prob[i];
				maxIndex = i;
			}
		}
		if (maxIndex == labelInt) {
			correctCount ++;
		}
	}

	// Average minibatch_cost and grad
	float f_minibatchSize = static_cast<float>(m_nMinibatchSize);
	crossEntropy /= f_minibatchSize;
	for (int dim=0; dim<m_nParamSize; dim++) {
		grad[dim] /= f_minibatchSize;
	}

	printf("Cross Entropy Error: %f\n", crossEntropy);
	printf("Correct rate: %d/%d=%f\n", int(correctCount), m_nMinibatchSize, correctCount / float(m_nMinibatchSize));

	return crossEntropy;
}