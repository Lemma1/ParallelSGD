#include <stdio.h>
#include <mpi.h>
#include <math.h>
#include <algorithm>

#include "master.h"

//#define DEBUG

void loadConf (masterConfInfo &confInfo) {
    // int related
    confInfo.paramSize = getMasterIntConf("parameter size");
    confInfo.nIterMax  = getMasterIntConf("max iteration number");
    confInfo.solverType  = getMasterIntConf("solver type");

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
        // params[i] = 0.f;
    }
}

sgdBase * initSgdSolver (masterConfInfo confInfo) {
    int solverType = confInfo.solverType;
    sgdBase *sgdSolver;
    switch (solverType) {
        // sgdBasic
        case 0: {
            printf("Init basic sgd solver.\n");
            sgdSolver = new sgdBasic(confInfo.paramSize, confInfo.learningRate);
            break;
        }
        // adagrad
        case 1: { 
            printf("Init adagrad solver.\n");
            sgdSolver = new adagrad(confInfo.paramSize, confInfo.learningRate);
            break;
        }
        // adadelta
        case 2: {
            float decayFactor = getMasterFloatConf("adadelta decay factor");
            float stableConst = getMasterFloatConf("adadelta stable const");
            sgdSolver = new adadelta(confInfo.paramSize, decayFactor, stableConst);
            printf("Init adadelta solver.\n");
            break;
        }
        // rmsprop
        case 3: { 
            float decayFactor = getMasterFloatConf("rmsprop decay factor");
            sgdSolver = new rmsprop(confInfo.paramSize, decayFactor);
            printf("Init rmsprop solver.\n");
            break;
        }
        default: {
            printf("Error solver type.\n");
            exit(-1);
        }
    }
    return sgdSolver;
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
    printf("MASTER: check trained params\n");
    for (int i = 0; i < paramSize; i++) {
        printf("%f\t", params[i]);
    }
    printf("\n");

    // Step 1.5: Initialize SGD Solver
    sgdBase *sgdSolver = initSgdSolver(confInfo);
    #ifdef DEBUG
    printf("MASTER: finish step 1\n");
    #endif

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
    #ifdef DEBUG
    printf("MASTER: finish step 2\n");
    #endif
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
        // printf("MASTER: check recv grad\n");
        // for (int i = 0; i < paramSize; i++) {
        //     printf("%f\t", grad[i]);
        // }
        // printf("\n");
    	// Call solver to update params
        // printf("MASTER: check grad\n");
        // for (int i = 0; i < paramSize; i++) {
        //     printf("%f\t", grad[i]);
        // }
        // printf("\n");
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
    #ifdef DEBUG
    printf("MASTER: finish step 3\n");
    #endif
    /****************************************************************
	* Step 4: Stop the slaves
	****************************************************************/
	
    // Step 4.1: Receive all dispatched but irreceived grad result
    while (nRecv < nSend) {
        // printf("Master, nSend:%d, nRecv:%d\n", nSend, nRecv);
        MPI_Recv(grad, paramSize, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        sgdSolver->updateParams(params, grad);
        nRecv++;
    }
    // Step 4.2: Send STOPTAG to all slaves
    for (int rank = 1; rank < nProc; ++rank) {
        MPI_Send(&rank, 1, MPI_INT, rank, STOPTAG, MPI_COMM_WORLD);
    }

    #ifdef DEBUG
    printf("MASTER: finish step 4\n");
    #endif
    /****************************************************************
    * Step 5: deallocate mem and clear things
    ****************************************************************/
    printf("MASTER: check trained params\n");
    for (int i = 0; i < paramSize; i++) {
        printf("%f\t", params[i]);
    }
    printf("\n");

    delete sgdSolver;

    delete [] params;
    delete [] grad;
}