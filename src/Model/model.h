#ifndef __MODEL_H__
#define __MODEL_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

class modelBase
{
public:
	modelBase();
	~modelBase();

	/* data */
	int m_nParamSize;
	int m_nMinibatchSize;

	/* method */
	float virtual computeGrad (float *grad, float *params, float *data, float *label);
};

class linearReg: public modelBase
{
public:
	linearReg(int paramSize, int minibatchSize);
	~linearReg();

	/* data */

	/* method */
	float computeGrad (float *grad, float *params, float *data, float *label);
};

#endif