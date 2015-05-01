#include <sstream> 
#include "rnn_lstm.h"

using namespace std;

RNN_LSTM::RNN_LSTM(ConfReader *confReader, int minibatchSize, string prefix) {
	// set minibatchSize;
	m_nMinibatchSize = minibatchSize;

	// read basic conf 
	m_numLayer = confReader->getInt(prefix+"num_layer");
	m_maxSeqLen = confReader->getInt(prefix+"max_sequence_length");
	
	m_inputSize = confReader->getInt(prefix+"input_size");
	m_outputSize = confReader->getInt(prefix+"output_size");

	// allocate memory
	m_numNeuronList = new int[m_numLayer];
	m_layerTypeList = new string[m_numLayer];
	m_connTypeList = new string[m_numLayer-1];

	// read type and number of neurons of each layer from conf
	for (int layerIdx=0; layerIdx<m_numLayer; ++layerIdx) {
		stringstream ss;
		ss << layerIdx;
		m_numNeuronList[layerIdx] = confReader->getInt(prefix+"num_neuron_layer_" + ss.str());
		m_layerTypeList[layerIdx] = confReader->getString(prefix+"type_layer_" + ss.str());
	}
	
	// read type of each conectection from conf
	for (int connIdx=0; connIdx<m_numLayer-1; ++connIdx) {
		stringstream ss;
		ss << connIdx;
		m_connTypeList[connIdx] = confReader->getString(prefix+"type_connection_" + ss.str());
	}
	
	// m_nParamSize based on the m_nParamSize of layers and connections
	m_nParamSize = 0;

	// initialize layers
	for (int layerIdx=0; layerIdx<m_numLayer; layerIdx++) {
		RecurrentLayer *layer = initLayer(layerIdx);
		m_nParamSize += layer->m_nParamSize;
		m_vecLayers.push_back(layer);
	}	

	// initialize connections
	for (int connIdx=0; connIdx<m_numLayer-1; connIdx++) {
		RNNConnection *conn = initConnection(connIdx);
		m_nParamSize += conn->m_nParamSize;
		m_vecConnections.push_back(conn);
	}
}

RNN_LSTM::~RNN_LSTM() {
	if (m_numNeuronList != NULL) {delete [] m_numNeuronList;}
	if (m_layerTypeList != NULL) {delete [] m_layerTypeList;}
	if (m_connTypeList != NULL) {delete [] m_connTypeList;}
	
	for (int layerIdx=0; layerIdx<m_numLayer; ++layerIdx) {
		if (m_vecLayers[layerIdx] != NULL) {delete m_vecLayers[layerIdx];}
	}

	for (int connIdx=0; connIdx<m_numLayer-1; ++connIdx) {
		if (m_vecConnections[connIdx] != NULL) {delete m_vecConnections[connIdx];}
	}
}

RNNConnection *RNN_LSTM::initConnection(int connIdx) {
	string connType = m_connTypeList[connIdx];
	
	RecurrentLayer *preLayer = m_vecLayers[connIdx];
	RecurrentLayer *postLayer = m_vecLayers[connIdx+1];
	RNNConnection *conn;
	if (connType == "full_connection") {
		conn = new RNNFullConnection(preLayer, postLayer);
	} else if (connType == "lstm_connection") {
		conn = new LSTMConnection(preLayer, postLayer);
	} else {		
		exit(-1);
	}
	return conn;
}

RecurrentLayer *RNN_LSTM::initLayer(int layerIdx) {
	string layerType = m_layerTypeList[layerIdx];
	int numNeuron = m_numNeuronList[layerIdx];
	RecurrentLayer *layer;
	if (layerType == "input_layer") {
		layer = new RNN_InputLayer(numNeuron, m_maxSeqLen);
	} else if (layerType == "lstm_layer") {
		int inputSize;
		if (layerIdx == 0) {
			inputSize = m_inputSize;
		} else {
			inputSize = m_numNeuronList[layerIdx-1];
		}
		layer = new LSTMLayer(numNeuron, m_maxSeqLen, inputSize);
	} else if (layerType == "softmax_layer") {
		m_errorType = "cross_entropy_error";
		layer = new RNN_SoftmaxLayer(numNeuron, m_maxSeqLen);
	} else if (layerType == "mse_layer") {
		m_errorType = "mean_squared_error";
		layer = new RNN_MSELayer(numNeuron, m_maxSeqLen);
	} else {		
		exit(-1);
	}
	return layer;
}

void RNN_LSTM::initParams(float *params) {
	float *cursor = params;
	// layer part
	for (int layerIdx=0; layerIdx<m_numLayer; ++layerIdx) {
		m_vecLayers[layerIdx]->initParams(cursor);
		cursor += m_vecLayers[layerIdx]->m_nParamSize;
	}
	// connection part
	for (int connIdx=0; connIdx<m_numLayer-1; ++connIdx) {
		m_vecConnections[connIdx]->initParams(cursor);
		cursor += m_vecConnections[connIdx]->m_nParamSize;
	}
}

void RNN_LSTM::feedForward(int inputSeqLen) {
	/* feed forward through connections and layers */
	m_vecLayers[0]->feedForward(inputSeqLen);
	for (int connIdx=0; connIdx<m_numLayer-1; ++connIdx) {
		m_vecConnections[connIdx]->feedForward(inputSeqLen);
		m_vecLayers[connIdx+1]->feedForward(inputSeqLen);
	}
}

void RNN_LSTM::feedBackward(int inputSeqLen) {
	/* feed backward through connections and layers */
	m_vecLayers[m_numLayer-1]->feedBackward(inputSeqLen);
	for (int connIdx=m_numLayer-2; connIdx>=0; --connIdx) {
		m_vecConnections[connIdx]->feedBackward(inputSeqLen);
		m_vecLayers[connIdx]->feedBackward(inputSeqLen);
	}
}

float RNN_LSTM::computeError(float *sampleTarget, int inputSeqLen) {
	float sampleError = 0.f;
	float *targetCursor = sampleTarget;
	RecurrentLayer *curLayer = m_vecLayers[m_numLayer-1];
	for (int seqIdx=1; seqIdx<=inputSeqLen; ++seqIdx) {
		for (int i=0; i<m_outputSize; ++i) {
			if (m_errorType == "cross_entropy_error") {
				sampleError += targetCursor[i] * log(curLayer->m_outputActs[seqIdx][i]);
			} else if (m_errorType == "mean_squared_error") {
				float diff = targetCursor[i] - curLayer->m_outputActs[seqIdx][i];
				sampleError += diff * diff;
			}
		}
		targetCursor += m_outputSize;
	}
	return sampleError;
}

float RNN_LSTM::computeGrad(float *grad, float *params, float *data, float *target) {
	float error = 0.f;
	
	memset(grad, 0x00, sizeof(float)*m_nParamSize);
	bindWeights(params, grad);
	
	float *sampleData = data;
	float *sampleTarget = target;

	/*** feed forward and feed backward ***/
	for (int dataIdx=0; dataIdx<m_nMinibatchSize; ++dataIdx) {
		// TODO
		int inputSeqLen = m_maxSeqLen;
		
		/* reset internal states of LSTM layers */
		resetStates(inputSeqLen); // this is subject to change
		
		/* feedforward */
		float *dataCursor = sampleData;
		// bind input sequence to m_inputActs of the input layer 
		RecurrentLayer *RNN_InputLayer = m_vecLayers[0];
		for (int seqIdx=1; seqIdx<=inputSeqLen; ++seqIdx) {
			memcpy(RNN_InputLayer->m_inputActs[seqIdx], dataCursor, sizeof(float)*m_inputSize);
			dataCursor += m_inputSize;
		}
		// feedForward through connections and layers
		feedForward(inputSeqLen);

		/* compute error */
		error += computeError(sampleTarget, inputSeqLen);

		/* feedbackword */
		float *targetCursor = sampleTarget;
		// bind target sequence to m_outputErrs of the output layer
		RecurrentLayer *outputLayer = m_vecLayers[m_numLayer-1];
		for (int seqIdx=1; seqIdx<=inputSeqLen; ++seqIdx) {
			memcpy(outputLayer->m_outputErrs[seqIdx], targetCursor, sizeof(float)*m_outputSize);
			targetCursor += m_outputSize;
		}
		// feedback through connections and layers
		feedBackward(inputSeqLen);

		sampleData += m_inputSize * inputSeqLen;
		sampleTarget += m_outputSize * inputSeqLen;
	}

	// normalization by number of input sequences and clip gradients to [-1, 1]
	float normFactor = 1.f / (float) m_nMinibatchSize;
	for (int dim=0; dim<m_nParamSize; ++dim) {
		grad[dim] *= normFactor;
		if (grad[dim] < -1.f) {
			grad[dim] = -1.f;
		} else if (grad[dim] > 1.f) {
			grad[dim] = 1.f;
		}
	}
	error *= normFactor;

	return error;
}

void RNN_LSTM::resetStates(int inputSeqLen) {
	for (int layerIdx=0; layerIdx<m_numLayer; ++layerIdx) {
		m_vecLayers[layerIdx]->resetStates(inputSeqLen);
	}
}

// Sequential Part
void RNN_LSTM::bindWeights(float *params, float *grad) {
	// define cursors
	float *paramsCursor = params;
	float *gradCursor = grad;
	// layer part
	for (int layerIdx=0; layerIdx<m_numLayer; ++layerIdx) {
		m_vecLayers[layerIdx]->bindWeights(paramsCursor, gradCursor);
		paramsCursor += m_vecLayers[layerIdx]->m_nParamSize;
		gradCursor += m_vecLayers[layerIdx]->m_nParamSize;
	}
	// connection part
	for (int connIdx=0; connIdx<m_numLayer-1; ++connIdx) {
		m_vecConnections[connIdx]->bindWeights(paramsCursor, gradCursor);
		paramsCursor += m_vecConnections[connIdx]->m_nParamSize;
		gradCursor += m_vecConnections[connIdx]->m_nParamSize;
	}
}