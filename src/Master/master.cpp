#include <stdio.h>
#include <mpi.h>
#include <math.h>
#include <algorithm>

#include "master.h"
#include "confreader.h"
#include "model.h"
#include "svm.h"
#include "neural_net.h"

// #define DEBUG_MASTER

modelBase * initModelMaster (ConfReader *modelConf, int validBatchSize) {
    int modelType = modelConf->getInt("model type");
    modelBase *model;
    switch (modelType) {
        // linear regression
        case 0: {
            printf("Model: Init linear regression.\n");
            model = new linearReg(modelConf, validBatchSize);
            break;
        }
        // softmax regression
        case 1: { 
            printf("Model: Init softmax regression.\n");
            model = new softmax(modelConf, validBatchSize);
            break;
        }
        // SVM
        case 2: {
            printf("Model: Init support vector machine.\n");
            model = new modelSVM(modelConf, validBatchSize);
            break;
        }
        // feedforward NN
        case 3: {
            printf("Model: Init feedforward neural network.\n");
            model = new feedForwardNN(modelConf, validBatchSize);
            break;
        }
        default: {
            printf("Error model type.\n");
            exit(-1);
        }
    }
    return model;
}

sgdBase * initSgdSolver (ConfReader *confReader, int paramSize) {
    int solverType = confReader->getInt("solver type");
    sgdBase *sgdSolver;
    switch (solverType) {
        // sgdBasic
        case 0: {
            printf("Init basic sgd solver.\n");
            sgdSolver = new sgdBasic(confReader, paramSize);
            break;
        }
        // adagrad
        case 1: { 
            printf("Init adagrad solver.\n");
            sgdSolver = new adagrad(confReader, paramSize);
            break;
        }
        // adadelta
        case 2: {            
            sgdSolver = new adadelta(confReader, paramSize);
            printf("Init adadelta solver.\n");
            break;
        }
        // rmsprop
        case 3: {
            sgdSolver = new rmsprop(confReader, paramSize);
            printf("Init rmsprop solver.\n");
            break;
        }
        // kernel adadelta
        case 4: {
            sgdSolver = new kernelAdadelta(confReader, paramSize);
            printf("Init kernel adadelta solver.\n");
            break;
        }
        // delayed adagrad
        case 5: {
            sgdSolver = new delayedAdagrad(confReader, paramSize);
            printf("Init delayed adagrad solver.\n");
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
    * Load conf, init model, allocate mem, init params, init solver
    * Load cross-validation data
    ****************************************************************/

    // Step 1.1: Load configuration
    ConfReader *masterConf = new ConfReader("config.conf", "Master");
    int validBatchSize = masterConf->getInt("validation batch size");
    #ifdef DEBUG_MASTER
    printf("validBatchSize: %d\n", validBatchSize);
    #endif

    // Step 1.2 Initialize model
    ConfReader *modelConf = new ConfReader("config.conf", "Model");
    modelBase *model = initModelMaster(modelConf, validBatchSize);
    int paramSize = model->m_nParamSize;
    printf("paramSize: %d\n", paramSize);    

    // Step 1.3: Allocate master memory
    float *params = new float[paramSize];
    float *grad = new float[paramSize];

    // Step 1.4: Initialize params
    model->initParams(params);
    #ifdef DEBUG_MASTER
    printf("MASTER: check initialized params\n");
    for (int i = 0; i < paramSize; i++) {
        printf("%f\t", params[i]);
    }
    printf("\n");
    #endif
	
    // Step 1.5: Initialize SGD Solver
    sgdBase *sgdSolver = initSgdSolver(masterConf, paramSize);
    printf("MASTER: finish step 1\n");

    // Step 1.6: Load cross-validation data
    // loadData();

    /****************************************************************
    * Step 2: Seed the slaves
    * (1) Broadcast paramSize to all slaves
    * (2) Send the same initial params with WORKTAG to all slaves
    ****************************************************************/
    int nProc;
    MPI_Comm_size(MPI_COMM_WORLD, &nProc);
    int nSlave = nProc - 1;

    MPI_Bcast(&paramSize, 1, MPI_INT, ROOT, MPI_COMM_WORLD);    
	
    int nSend = 0;
    int nRecv = 0;
    for (int rank = 1; rank < nProc; ++rank) {
        MPI_Send(params, paramSize, MPI_FLOAT, rank, WORKTAG, MPI_COMM_WORLD);
        nSend++;
    }
    printf("MASTER: finish step 2\n");

    /****************************************************************
    * Step 3: Paralleled training
	* Receive mini-batch grad from *ANY* slave
    * Update params based received grad
	* Re-send params to slave to process next mini-batch
	****************************************************************/
	
    MPI_Status status;
    int nSendMax = masterConf->getInt("max iteration number");
    
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
    	sgdSolver->updateParams(params, grad, status.MPI_SOURCE);

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
    printf("MASTER: finish step 3\n");
    
    /****************************************************************
	* Step 4: Stop the slaves
	****************************************************************/
	
    // Step 4.1: Receive all dispatched but irreceived grad result
    while (nRecv < nSend) {
        // printf("Master, nSend:%d, nRecv:%d\n", nSend, nRecv);
        MPI_Recv(grad, paramSize, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        sgdSolver->updateParams(params, grad, status.MPI_SOURCE);
        nRecv++;
    }
    // Step 4.2: Send STOPTAG to all slaves
    for (int rank = 1; rank < nProc; ++rank) {
        MPI_Send(&rank, 1, MPI_INT, rank, STOPTAG, MPI_COMM_WORLD);
    }    
    printf("MASTER: finish step 4\n");
    
    /****************************************************************
    * Step 5: deallocate mem and clear things
    ****************************************************************/
    #ifdef DEBUG_MASTER
    printf("MASTER: check trained params\n");
    for (int i = 0; i < paramSize; i++) {
        printf("%f\t", params[i]);
    }
    printf("\n");
    #endif

    delete sgdSolver;

    delete [] params;
    delete [] grad;
}