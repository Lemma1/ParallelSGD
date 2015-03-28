#ifndef __LAYER_H__
#define __LAYER_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>



class layerBase
{
public:
	layerBase(){};
	~layerBase(){};

	/* data */
	int m_numNeuron;
	float *m_activation;
	float *m_delta;

	/* method */
	void virtual activateFunc (float *) {};
	void virtual computeDelta (float *) {};
};

class linearLayer: public layerBase
{
public:
	linearLayer (int numNeuron);
	~linearLayer ();

	/* data */

	/* method */
	void activateFunc (float *input);
	void computeDelta (float *error);
};

class softmaxLayer: public layerBase
{
public:
	softmaxLayer (int numNeuron);
	~softmaxLayer ();

	/* data */

	/* method */
	void activateFunc (float *input);
	void computeDelta (float *target);
};

class sigmoidLayer: public layerBase
{
public:
	sigmoidLayer (int m_numNeuron);
	~sigmoidLayer ();

	/* data */

	/* method */
	void activateFunc (float *input);
	void computeDelta (float *error);
};

#endif