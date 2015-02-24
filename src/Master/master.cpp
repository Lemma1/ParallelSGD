#include <mpi.h>
#include <stdio.h>

#include "master.h"
#include "sgd.h"

void loadConf () {
    fprintf(stderr, "Load conf failed in file %s at line %d\n", __FILE__, __LINE__);
}

void masterFunc () {
    /****************************************************************
    * Step 1: Setup and Initialization
    * Load conf, allocate mem, init params, init solver
    ****************************************************************/
    // Step 1.1: Load configuration
    loadConf(); // TODO
	
    // Step 1.2: Get basic MPI info
    int nProc, nSlave;
    MPI_Comm_size(MPI_COMM_WORLD, &nProc);
    nSlave = nProc - 1;

    // Step 1.3: Allocate master memory
    float *params = new float[paramSize];
    float *grad = new float[paramSize];

    // Step 1.4: Initialize params
    initParams(params); // TODO

    // Step 1.5: Initialize SGD Solver
    // TODO

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
    // TEMP CODE
    int nSendMax = 10000; 
    while (nSend < nSendMax) {
        MPI_Recv(grad, paramSize, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        nRecv++;

    	// Call solver to update params
    	// SGDSolver.updateParams(params, grad); // TODO
        // TEMP CODE
        for (int i = 0; i < paramSize; i++) {
            params[i] += 0.01 * grad[i];
        }

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
}