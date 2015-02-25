#include <math>

class modelBase
{
public:
	modelBase();
	~modelBase();

	/* data */
	int m_nParamSize;
	int m_nMinibatchSize;

	/* method */
	void virtual computeGrad (float *grad, float *params, float *data, float *label);
};

class linearReg: public modelBase
{
public:
	linearReg(int paramSize, int minibatchSize);
	~linearReg();

	/* data */

	/* method */
	void computeGrad (float *grad, float *params, float *data, float *label);
};