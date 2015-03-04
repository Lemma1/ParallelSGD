#include <stdio.h>
#include <mpi.h>
#include <math.h>
#include <algorithm>

#include "master.h"
#include "sgd.h"
#include "MasterConfig.h"

void loadConf (masterConfInfo &confInfo) {
    // int related
    confInfo.paramSize = getMasterIntConf("parameter size");
    confInfo.nIterMax  = getMasterIntConf("max iteration number");

    // float related
    confInfo.learningRate = getMasterFloatConf("global learning rate");
    confInfo.initRange = getMasterFloatConf("parameter init range");

    // string related
}

void initParams (masterConfInfo confInfo, float *params) {
    float initMin, initMax;
    initMin = -confInfo.initRange;
    initMax = confInfo.initRange;
    float initWidth = initMax - initMin;
    for (int i=0; i<confInfo.paramSize; i++) {
        params[i] = initMin + initWidth * static_cast<float>(rand()) / RAND_MAX;
    }
}

void masterFunc () {
    /****************************************************************
    * Step 1: Setup and Initialization
    * Load conf, allocate mem, init params, init solver
    ****************************************************************/
    // Step 1.1: Load configuration
    masterConfInfo confInfo;
    loadConf(confInfo);

    int paramSize = confInfo.paramSize;
    float learningRate = confInfo.learningRate;
	
    // Broadcast paramSize to all slaves
    MPI_Bcast(&paramSize, 1, MPI_INT, ROOT, MPI_COMM_WORLD);

    // Step 1.2: Get basic MPI info
    int nProc, nSlave;
    MPI_Comm_size(MPI_COMM_WORLD, &nProc);
    nSlave = nProc - 1;

    // Step 1.3: Allocate master memory
    float *params = new float[paramSize];
    float *grad = new float[paramSize];

    // Step 1.4: Initialize params
    initParams(confInfo, params);

    // Step 1.5: Initialize SGD Solver
    sgdBase *sgdSolver = new sgdBasic(paramSize, learningRate);

    // Step 1.6: Load cross-validation data
    // loadData();

	/****************************************************************
    * Step 2: Seed the slaves
	* Send the same initial params with WORKTAG to all slaves
	****************************************************************/
    int nSend = 0;
    int nRecv = 0;
    for (int rank = 1; rank < nProc; ++rank) {
        MPI_Send(params, paramSize, MPI_FLOAT, rank, WORKTAG, MPI_COMM_WORLD);
        nSend++;
    }

    /****************************************************************
    * Step 3: Paralleled training
	* Receive mini-batch grad from *ANY* slave
    * Update params based received grad
	* Re-send params to slave to process next mini-batch
	****************************************************************/
	
    MPI_Status status;
    int nSendMax = confInfo.nIterMax;
    
    // TEMP while loop condition
    while (nSend < nSendMax) {
        MPI_Recv(grad, paramSize, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        nRecv++;

    	// Call solver to update params
    	sgdSolver->updateParams(params, grad);

        // Check recv tag (eg. local new epoch info)
        // if (status.MPI_TAG == SOME_TAG) {}

        // Cross-validation (need a model tester)
        // TODO

        // Quit when certain condition meets (cross-validation, status)
        // TODO
    	if (0) {
        	break;
        }
        
        // Send updated params to corresponding slave
        MPI_Send(params, paramSize, MPI_FLOAT, status.MPI_SOURCE, WORKTAG, MPI_COMM_WORLD);
        nSend++;
    }

    /****************************************************************
	* Step 4: Stop the slaves
	****************************************************************/
	
    // Step 4.1: Receive all dispatched but irreceived grad result
    while (nRecv < nSend) {
        MPI_Recv(grad, paramSize, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    }

    // Step 4.2: Send STOPTAG to all slaves
    for (int rank = 1; rank < nProc; ++rank) {
        MPI_Send(&rank, 1, MPI_INT, rank, STOPTAG, MPI_COMM_WORLD);
    }

    /****************************************************************
    * Step 5: deallocate mem and clear things
    ****************************************************************/
    delete sgdSolver;

    delete [] params;
    delete [] grad;
}