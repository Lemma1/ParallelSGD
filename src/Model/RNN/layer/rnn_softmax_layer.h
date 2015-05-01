#ifndef __RNN_SOFTMAX_LAYER_H__
#define __RNN_SOFTMAX_LAYER_H__

#include "rnn_layer.h"

using namespace std;

class RNN_SoftmaxLayer: public RecurrentLayer
{
public:
	RNN_SoftmaxLayer(int numNeuron, int maxSeqLen) : RecurrentLayer(numNeuron, maxSeqLen, numNeuron){};
	~RNN_SoftmaxLayer() {};

	/* data */

	/* method */
	void feedForward(int inputSeqLen);
	void feedBackward(int inputSeqLen);
};

#endif