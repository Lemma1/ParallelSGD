#include "sgd.h"

sgdBasic::sgdBasic (int size, float eta) {
	paramSize = size;
	learningRate = eta;
}

sgdBasic::~sgdBasic () {
	// nothing to do for basic sgd
}

void sgdBasic::updateParams (float *params, float *grad) {
	for (int i=0; i<paramSize; i++) {
		params[i] -= learningRate * grad[i];
	}
}