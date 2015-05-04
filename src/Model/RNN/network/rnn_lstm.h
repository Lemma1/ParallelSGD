#ifndef __RNN_LSTM_H__
#define __RNN_LSTM_H__

#include "recurrent_nn.h"

using namespace std;

class RNN_LSTM: public RecurrentNN 
{

public:
	RNN_LSTM(ConfReader *confReader, int minibatchSize, string prefix= "");
	~RNN_LSTM();

	/* data */
	string m_taskType;

	/* method */
	float computeGrad (float *grad, float *params, float *data, float *label);
	void initParams (float *params);

	float computeError(float *sampleTarget, int inputSeqLen);

	void feedBackward(int inputSeqLen);
	void feedForward(int inputSeqLen);

	void bindWeights(float *params, float *grad);
	void resetStates(int inputSeqLen);
	
private:
	RecurrentLayer *initLayer (int layerIdx);
	RNNConnection *initConnection(int connIdx);	
};

#endif