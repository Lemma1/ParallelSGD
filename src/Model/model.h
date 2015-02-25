#include <math>

class modelBase
{
public:
	modelBase();
	~modelBase();

	/* data */
	int paramSize;
	int minibatchSize;

	/* method */
	void virtual computeGrad (float *params, float *data);
};

class linearReg: public modelBase
{
public:
	linearReg(int paramSize, int minibatchSize);
	~linearReg();

	/* data */

	/* method */
	void computeGrad (float *params, float *data);
};