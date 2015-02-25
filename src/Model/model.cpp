#include "model.cpp"

linearReg::linearReg (int paramSize, int minibatchSize) {
	m_nParamSize = paramSize;
	m_nMinibatchSize = minibatchSize;
}

float linearReg::computeGrad (float *grad, float *params, float *data, float *label) {
	// Init variables
	float cost = 0.f;
	float diff;
	memset(grad, 0x00, size(float) * m_nParamSize);

	// Accumulate cost and grad
	for (int sample=0; sample<m_nMinibatchSize; sample++) {
		for (int dim=0; dim<m_nParamSize; dim++) {
			diff = params[dim] * data[dim] - label[dim];
			cost += 0.5 * diff * diff;
			grad[dim] += data[dim] * diff;
		}
	}

	// Average minibatch_cost and grad
	float f_minibatchSize = static_cast<float>(m_nMinibatchSize);
	cost /= f_minibatchSize;
	for (int dim=0; dim<m_nParamSize; dim++) {
		grad[dim] /= f_minibatchSize;
	}

	return cost;
}