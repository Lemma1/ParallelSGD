#include "svm.h"

#include <algorithm>
#include <iostream>

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
    float predict, f_label;
    int offset;
    float correct_counter = 0.f;
    memset(grad, 0x00, sizeof(float) * m_nParamSize); 
    float target; //y_t * w^T * x_t
    
    // Accumulate cost and grad
    for (int i=0; i < m_nMinibatchSize; i++)
    {
	offset = i * m_nParamSize;	
	predict = 0.f; //w^T * x^t
	for (int j=0; j < m_nParamSize; j++)
	{
	    predict += data[offset+j] * params[j];
	}
	//printf("predict : %f \n", predict);
	f_label = static_cast<float>(label[i]);
	target = f_label * predict;
	if (target > 0) correct_counter++;
	//std::cout << "target" << target << std::endl;
	if (target > 1)
	{
	    for (int j=0; j < m_nParamSize; j++) grad[j] += svm_lambda * params[j];
	}
	else
	{
	    for (int j=0; j < m_nParamSize; j++) 
	    {
		grad[j] += svm_lambda * params[j] - f_label * data[offset+j];
	    }
	}
	//printf("Iteration %d", i);
	//for (int j=0; j < m_nParamSize; j++) printf("%f,",grad[j]);
	//printf("\n");
	
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
    for (int j=0; j < m_nParamSize; j++) 
    {
    	grad[j] /= f_minibatchSize;
    }
    //printf("Correct Number: %d \n", static_cast<int> (correct_counter));
    //std::cout << "Correct rate: " << correct_counter/f_minibatchSize << std::endl;
    std::cout << correct_counter/f_minibatchSize << std::endl;
    return cost;
}

void modelSVM::initParams (float *params)
{
    for (int i=0; i<m_nParamSize; i++)
    {
	params[i] = static_cast<float>(rand())/RAND_MAX; 
    }
}
