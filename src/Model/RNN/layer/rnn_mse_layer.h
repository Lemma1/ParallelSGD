#ifndef __RNN_MSE_LAYER_H__
#define __RNN_MSE_LAYER_H__

#include "rnn_layer.h"

using namespace std;

class RNN_MSELayer: public RecurrentLayer
{
public:
	RNN_MSELayer(int numNeuron, int maxSeqLen) : RecurrentLayer(numNeuron, maxSeqLen, numNeuron){};
	~RNN_MSELayer() {};

	/* data */

	/* method */
	void feedForward(int inputSeqLen);
	void feedBackward(int inputSeqLen);
};

#endif