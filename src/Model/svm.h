#ifndef __SVM_H__
#define __SVM_H__

#include "model.h"

class modelSVM :public modelBase
{
    public:
	modelSVM(int, int, float);
	~modelSVM();

	/* data */
	int svm_lambda;

	/* method */
	float computeGrad (float *grad, float *params, float *data, float *label);
	// label need to be +- 1
};
#endif
