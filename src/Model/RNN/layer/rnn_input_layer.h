#ifndef __RNN_INPUT_LAYER_H__
#define __RNN_INPUT_LAYER_H__

#include "rnn_layer.h"

/****************************************************************
* Input Layer
****************************************************************/
class RNN_InputLayer : public RecurrentLayer
{
public:
	RNN_InputLayer(int numNeuron, int maxSeqLen) : RecurrentLayer(numNeuron, maxSeqLen, numNeuron) {};
	~RNN_InputLayer() {};

	/* data */
	void feedForward(int inputSeqLen);
	void feedBackward(int inputSeqLen);
};

#endif