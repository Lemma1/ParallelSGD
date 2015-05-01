#include "rnn_input_layer.h"

using namespace std;

void RNN_InputLayer::feedForward(int inputSeqLen) {		
	for (int seqIdx=1; seqIdx<=inputSeqLen; ++seqIdx) {		
		memcpy(m_outputActs[seqIdx], m_inputActs[seqIdx], sizeof(float) * m_numNeuron);
	}	
}

void RNN_InputLayer::feedBackward(int inputSeqLen) {
	// nothing to do
}