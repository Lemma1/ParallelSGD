#include "rnn_connection.h"

using namespace std;

/****************************************************************
* Recurrent Connection Base
****************************************************************/

RNNConnection::RNNConnection(RecurrentLayer *preLayer, RecurrentLayer *postLayer) {
	m_preLayer = preLayer;
	m_postLayer = postLayer;
}

/****************************************************************
* Recurrent Full-Connection
****************************************************************/

RNNFullConnection::RNNFullConnection(RecurrentLayer *preLayer, RecurrentLayer *postLayer) : RNNConnection(preLayer, postLayer) {	
	// weights
	m_nParamSize = m_postLayer->m_numNeuron * m_preLayer->m_numNeuron;
	// bias
	m_nParamSize += m_postLayer->m_numNeuron;
}

void RNNFullConnection::initParams(float *params) {	
	float multiplier = 0.08;
	for (int i=0; i<m_nParamSize; ++i) {
		params[i] = multiplier * SYM_UNIFORM_RAND;
	}
}

void RNNFullConnection::bindWeights(float *params, float *grad) {
	float *paramsCursor = params;
	float *gradCursor = grad;
	// weights
	m_weights = paramsCursor;
	m_gradWeights = gradCursor;
	paramsCursor += m_postLayer->m_numNeuron * m_preLayer->m_numNeuron;
	gradCursor += m_postLayer->m_numNeuron * m_preLayer->m_numNeuron;
	// bias
	m_bias = paramsCursor;
	m_gradBias = gradCursor;
}

void RNNFullConnection::feedForward(int inputSeqLen) {
	int preNumNeuron = m_preLayer->m_numNeuron;
	int postNumNeuron = m_postLayer->m_numNeuron;
	for (int seqIdx=1; seqIdx<=inputSeqLen; ++seqIdx) {		
		// m_weights
		float *weights = m_weights;
		dot(m_postLayer->m_inputActs[seqIdx], weights, postNumNeuron, preNumNeuron, m_preLayer->m_outputActs[seqIdx], preNumNeuron, 1);
		// m_bias
		elem_accum(m_postLayer->m_inputActs[seqIdx], m_bias, postNumNeuron);
	}
}

void RNNFullConnection::feedBackward(int inputSeqLen) {
	int preNumNeuron = m_preLayer->m_numNeuron;
	int postNumNeuron = m_postLayer->m_numNeuron;	
	for (int seqIdx=1; seqIdx<=inputSeqLen; ++seqIdx) {
		// m_preLayer->m_outputErrs
		trans_dot(m_preLayer->m_outputErrs[seqIdx], m_weights, postNumNeuron, preNumNeuron, m_postLayer->m_inputErrs[seqIdx], postNumNeuron, 1);
		// m_gradWeights
		outer(m_gradWeights, m_postLayer->m_inputErrs[seqIdx], postNumNeuron, m_preLayer->m_outputActs[seqIdx], preNumNeuron);
		// m_gradBias
		elem_accum(m_gradBias, m_postLayer->m_inputErrs[seqIdx], postNumNeuron);
	}
}

/****************************************************************
* Recurrent LSTM-Connection
****************************************************************/

LSTMConnection::LSTMConnection(RecurrentLayer *preLayer, RecurrentLayer *postLayer) : RNNConnection(preLayer, postLayer) {
	m_nParamSize = 0;
}

void LSTMConnection::feedForward(int inputSeqLen) {
	// independent loop -> use OpenMP potentially
	int inputSize = m_preLayer->m_numNeuron;
	for (int seqIdx=1; seqIdx<=inputSeqLen; ++seqIdx) {
		float *preOutActs = m_preLayer->m_outputActs[seqIdx];
		float *postInActs = m_postLayer->m_inputActs[seqIdx];
		memcpy(postInActs, preOutActs, sizeof(float)*inputSize);
	}
}

void LSTMConnection::feedBackward(int inputSeqLen) {
	// independent loop -> use OpenMP potentially
	int errorSize = m_preLayer->m_numNeuron;
	for (int seqIdx=1; seqIdx<=inputSeqLen; ++seqIdx) {
		float *preOutErrs = m_preLayer->m_outputErrs[seqIdx];
		float *postInErrs = m_postLayer->m_inputErrs[seqIdx];
		memcpy(preOutErrs, postInErrs, sizeof(float)*errorSize);
	}
}