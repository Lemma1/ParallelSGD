#include <mpi.h>
#include <stdio.h>
#include <algorithm>

#include "slave.h"
#include "model.h"
#include "svm.h"
#include "neural_net.h"
#include "TestData.h"
#include "DataFactory.h"
#include "confreader.h"
#include "Mnist.h"
#include "binary.h"
/*
#include "../Model/model.h"
#include "../Data/TestData.h"
#include "../Data/DataFactory.h"
*/

//#define DEBUG_SLAVE

modelBase * initModelSlave (ConfReader *modelConf, int batchSize) {
    int modelType = modelConf->getInt("model type");
    modelBase *model;
    switch (modelType) {
        // linear regression
        case 0: {
            printf("Slave Model: Init linear regression.\n");
            model = new linearReg(modelConf, batchSize);
            break;
        }
        // softmax regression
        case 1: { 
            printf("Slave Model: Init softmax regression.\n");
            model = new softmax(modelConf, batchSize);
            break;
        }
        // SVM
        case 2: {
            printf("Slave Model: Init support vector machine.\n");
            model = new modelSVM(modelConf, batchSize);
            break;
        }
        // feedforward NN
        case 3: {
            printf("Slave Model: Init feedforward neural network.\n");
            model = new feedForwardNN(modelConf, batchSize);
            break;
        }
        default: {
            printf("Error model type.\n");
            exit(-1);
        }
    }
    return model;
}

//Wei MA
//Use init function to initialize the datafactory
DataFactory* initDataFactory()
{
    int dataIndex = modelConf->getInt("data index");
    DataFactory* data;
    switch(dataIndex)
    {
	//linear data
	case 1: 
	    {	
		printf("Slave Model: Init Linear Data.\n");
		data = new TestData();
	    }
	case 2: 
	    {	
		printf("Slave Model: Init Minst Data.\n");
		data = new Minst();
	    }
	case 3: 
	    {	
		printf("Slave Model: Init Binary Data.\n");
		data = new BinaryData();
	    }
	default: 
	    {
		printf("Error, no Data Index");
		exit(-1);
	    }
	return data;

    }

}
//random pick the data 
//the main function of slaves


void slaveDo(){
    //step 0:init the data in local memory
    // DataFactory *dataset = new TestData();
    // DataFactory *dataset = new BinaryData();
    DataFactory *dataset = new Mnist(1);
    
    int dbSize = dataset->getNumberOfData();// define in slave.h or ?
    
    //dataInit(&dbSize,&batchSize);//TODO
    ConfReader *slaveConf = new ConfReader("config.conf", "Slave");
    int batchSize = slaveConf->getInt("training batch size");
    #ifdef DEBUG_SLAVE
    printf("batchSize: %d\n", batchSize);
    #endif

    MPI_Status status;
	//step 1:: configulation
    //slaveConfinfo sconfig;
    /*if(~slaveLoad(&sconfig))
        return -1;*/	
    int paramSize;

    //step 1.5:receive some pre-parameters 
    MPI_Bcast(&paramSize,1,MPI_INT,ROOT,MPI_COMM_WORLD);
    float *param = new float[paramSize]; 
    float *grad  = new float[paramSize];
    float *data  = new float[batchSize*paramSize];
    float *label = new float[batchSize];
    int   *index = new int[dbSize];
    int   *pickIndex = new int[batchSize];

    ConfReader *modelConf = new ConfReader("config.conf", "Model");
    modelBase *model = initModelSlave(modelConf, batchSize);
    for (int i=0;i<dbSize;i++){
        index[i]=i;
    }

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int count = 0;
    int indexI = 0;
    printf("Slave[%d] go into loop\n", rank);
	//main loop
    while(1){
		/*step 2:receive from master*/
		MPI_Recv(param,paramSize,MPI_FLOAT,ROOT,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
        count++;
        // printf("%d:%d\n", rank, count);
        
		/*step 3: check whether ends*/
		if(status.MPI_TAG == STOPTAG){
            break;
        } 
        
        /*step 4: request for data*/
        if (indexI+batchSize >= dbSize){
            std::random_shuffle(index,index+dbSize);
            indexI = 0;
        }
        
        for(int i=0;i<batchSize;i++){
            pickIndex[i] = index[indexI];
            indexI++;
        }        
        dataset->getDataBatch(label, data, pickIndex, batchSize);        
        //dataset->printOutData();

        /*step 5: calculate the grad*/        
        float cost = model->computeGrad(grad, param, data, label);
        // printf("MASTER: check grad\n");
        // for (int i = 0; i < paramSize; i++) {
        //     printf("%f\t", grad[i]);
        // }
        // printf("\n");
        // printf("SLAVE[%d]: %f\n", rank, cost);
        
        /*step 6: return to master*/
        MPI_Send(grad, paramSize, MPI_FLOAT, ROOT, rank, MPI_COMM_WORLD);
	}


    delete [] param;
    delete [] grad;
    delete [] label;
    delete [] data;
    delete [] index;
    delete dataset;
}
