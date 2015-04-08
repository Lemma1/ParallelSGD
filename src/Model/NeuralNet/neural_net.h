#ifndef __NEURAL_NET_H__
#define __NEURAL_NET_H__

#include <vector>
#include "model.h"
#include "layer.h"

class feedForwardNN: public modelBase
{
public:
	feedForwardNN (ConfReader *confReader, int minibatchSize);
	~feedForwardNN ();

	/* data */
	int m_numLayer;
	int *m_numNeuronList;
	int *m_layerTypeList;
	
	layerBase *m_softmaxLayer;
	std::vector<layerBase *> m_vecLayers;

	std::vector<float *> m_vecForwardInfo;
	std::vector<float *> m_vecBackpropInfo;

	std::vector<float *> m_vecWeights;
	std::vector<float *> m_vecWeightsGrad;

	/* method */
	float computeGrad (float *grad, float *params, float *data, float *label);
	void initParams (float *params);

private:
	layerBase *initLayer (int numNeuron, int layerType);
	void initWeights (float *weights, int fanIn, int fanOut, int type=0);
	void feedForward (float *input);
	void backProp (float *target);
};

#endif