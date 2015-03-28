#include <stdio.h>
#include "model.h"
#include "neural_net.h"

int main () {

	int numLayer = 4;
	int numNeuronList[4] = {2, 3, 3, 2};
	int layerTypeList[4] = {0, 1, 1, 1};

	modelBase * model = new feedForwardNN(2, numLayer, numNeuronList, layerTypeList);
	
	int paramSize = model->m_nParamSize;
	printf("paramSize: %d\n", paramSize);
	
	float *params = new float[paramSize];
	model->initParams(params);
	// for (int i=0; i<paramSize; i++) {
	// 	printf("Param %d:%f\n", i, params[i]);
	// }

	float data[4] = {1, 2, 3, 4};
	float label[4] = {1, 0, 0, 1};

	float *grad = new float[paramSize];
	float error = model->computeGrad(grad, params, data, label);
	printf("error: %f\n", error);
	for (int i=0; i<paramSize; i++) {
		printf("Grad %d:%f\n", i, grad[i]);
	}
	return 1;
}