#ifndef __RNN_TRANSLATOR_H__
#define __RNN_TRANSLATOR_H__

#include "model.h"
#include "rnn_lstm.h"

using namespace std;

class RNNTranslator: public modelBase
{
public:
	RNNTranslator(ConfReader *confReader, int minibatchSize, string prefix="");
	~RNNTranslator();

	/* data */
	int m_reverseEncoder;

	float *m_encodingW;
	float *m_gradEncodingW;

	RNN_LSTM *m_encoder;
	RNN_LSTM *m_decoder;

	/* method */
	float computeGrad (float *grad, float *params, float *data, float *label);
	void initParams (float *params);

private:
	void bindWeights(float *params, float *grad);
};

#endif