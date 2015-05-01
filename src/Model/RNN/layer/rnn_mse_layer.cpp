#include "rnn_mse_layer.h"

void RNN_MSELayer::feedForward(int inputSeqLen) {
	for (int seqIdx=1; seqIdx<=inputSeqLen; ++seqIdx) {		
		memcpy(m_outputActs[seqIdx], m_inputActs[seqIdx], sizeof(float) * m_numNeuron);
	}
}

void RNN_MSELayer::feedBackward(int inputSeqLen) {
	for (int seqIdx=1; seqIdx<=inputSeqLen; ++seqIdx) {
		elem_sub(m_inputErrs[seqIdx], m_outputActs[seqIdx], m_outputErrs[seqIdx], m_numNeuron);
	}
}

