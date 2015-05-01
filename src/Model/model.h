#ifndef __MODEL_H__
#define __MODEL_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "confreader.h"

#define SYM_UNIFORM_RAND (2 * ((float) rand() / (RAND_MAX)) - 1)   // rand float in [-1, 1]

using namespace std;

class modelBase
{
public:
	modelBase(){};
	virtual ~modelBase(){};

	/* data */
	int m_nParamSize;
	int m_nMinibatchSize;

	/* method */
	float virtual computeGrad (float *grad, float *params, float *data, float *label) {return 0.f;};
	void virtual initParams (float *params) {};
};

class linearReg: public modelBase
{
public:
	linearReg (ConfReader *confReader, int minibatchSize);
	~linearReg ();

	/* data */
	int m_inputSize;

	/* method */
	float computeGrad (float *grad, float *params, float *data, float *label);
	void initParams (float *params);
};

class softmaxReg: public modelBase
{
public:
	softmaxReg (ConfReader *confReader, int minibatchSize);
	~softmaxReg ();

	/* data */
	int m_inputSize;
	int m_classNum;
	float *m_prob;
	float *m_oneOnlabel;

	/* method */
	float computeGrad (float *grad, float *params, float *data, float *label);
	void initParams (float *params);
};

#endif
