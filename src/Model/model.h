#ifndef __MODEL_H__
#define __MODEL_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "confreader.h"

#define SYM_UNIFORM_RAND (2 * ((float) rand() / (RAND_MAX)) - 1)   // rand float in [-1, 1]

class modelBase
{
public:
	modelBase(){};
	~modelBase(){};

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

	/* method */
	float computeGrad (float *grad, float *params, float *data, float *label);
	void initParams (float *params);
};

class softmax: public modelBase
{
public:
	softmax (ConfReader *confReader, int minibatchSize);
	~softmax ();

	/* data */
	int m_nClassNum;

	/* method */
	float computeGrad (float *grad, float *params, float *data, float *label);
};

#endif
