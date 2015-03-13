#ifndef __MASTER_H__
#define __MASTER_H__

#include <mpi.h>
#include <stdio.h>
#include "sgd.h"
#include "MasterConfig.h"

#define ROOT 0

#define WORKTAG 1
#define STOPTAG 2

struct masterConfInfo {
	int paramSize;
	int nIterMax;
	int solverType;

	float learningRate;
	float initRange;
};

void masterFunc ();

void loadConf (masterConfInfo &confInfo);

void initParams (masterConfInfo confInfo, float *params);

sgdBase * initSgdSolver (masterConfInfo confInfo);

#endif