#include "svm.h"

#include <algorithm>

modelSVM::modelSVM(ConfReader *confReader, int minibatchSize)
{
    m_nParamSize = confReader->getInt("parameter size");    
    svm_lambda = confReader->getFloat("svm lambda");
    m_nMinibatchSize = minibatchSize;
}

modelSVM::~modelSVM()
{

}

float modelSVM::computeGrad (float *grad, float *params, float *data, float *label)
{
    float cost = 0.f;
    float predict;
    int offset;
    memset(grad, 0x00, sizeof(float) * m_nParamSize); 
   
    float target; //y_t * w^T * x^t
    predict = 0.f; //w^T * x^t
    // Accumulate cost and grad
    for (int i=0; i < m_nMinibatchSize; i++)
    {
	offset = i * m_nParamSize;	
	for (int j=0; j < m_nParamSize; j++)
	{
	    predict += data[offset+j] * params[j];
	}
	target = label[i] * predict;
	if (target > 1)
	{
	    for (int j=0; j < m_nParamSize; j++) grad[j] += svm_lambda * grad[j];
	}
	else
	{
	    for (int j=0; j < m_nParamSize; j++) 
	    {
		grad[j] += svm_lambda * grad[j] - label[i] * params[j];
	    }
	}
	
	//accumulative cost
	//obj = lambda * w^2 + max(0, 1-target)
	cost += std::max(0.f, 1.f-target);
    }
    for (int j=0; j < m_nParamSize; j++)
    {
	cost += svm_lambda * params[j] * params[j];
    }

    //average grad (maybe not used)
    float f_minibatchSize = static_cast<float>(m_nMinibatchSize);
    for (int j=0; j < m_nParamSize; j++) grad[j] /= f_minibatchSize;

    return cost;
}
