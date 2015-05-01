#include <mpi.h>
#include <stdio.h>
#include <algorithm>

#include "slave.h"
#include "model.h"
#include "svm.h"
#include "neural_net.h"
#include "rnn_translator.h"
#include "TestData.h"
#include "DataFactory.h"
#include "confreader.h"
#include "Mnist.h"
#include "binary.h"
#include "sequence_data.h"

#include <time.h>

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
            model = new softmaxReg(modelConf, batchSize);
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
        // LSTM Translator
        case 4: {
            printf("Slave Model: Init LSTM Translator.\n");
            model = new RNNTranslator(modelConf, batchSize);
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
DataFactory* initDataFactory(ConfReader *slaveConf)
{
    int dataIndex = slaveConf->getInt("data index");
    DataFactory* data;
    switch(dataIndex) {
        case 0: {
            printf("Slave Data: Init Sequence Data.\n");
            data = new SequenceData(slaveConf);
            break;
        }
    	//linear data
    	case 1: {
    		printf("Slave Data: Init Linear Data.\n");
    		data = new TestData();
            break;
	    }
    	case 2: {
    		printf("Slave Data: Init Minst Data.\n");
    		data = new Mnist(1);
            break;
	    }
    	case 3: {
    		printf("Slave Data: Init Binary Data.\n");
    		data = new BinaryData();
            break;
	    }
    	default:  {
    		printf("Error, no Data Index");
    		exit(-1);
	    }
    }
    return data;
}
//random pick the data 
//the main function of slaves


void slaveDo(){ 
    openblas_set_num_threads(1);
    //step 0:init the data in local memory    
    ConfReader *slaveConf = new ConfReader("config.conf", "Slave");
    int batchSize = slaveConf->getInt("training batch size");
    printf("training batchSize: %d\n", batchSize);

    DataFactory *dataset = initDataFactory(slaveConf);
    int dbSize = dataset->getNumberOfData();// define in slave.h or ?
    
    int dataSize = dataset->getDataSize();
    int labelSize = dataset->getLabelSize();

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
    float *data  = new float[batchSize*dataSize];
    float *label = new float[batchSize*labelSize];
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
    srand(time(NULL) * rank);
    std::random_shuffle(index,index+dbSize);
    printf("Slave[%d] go into loop\n", rank);
	//main loop
    while(1){
		/*step 2:receive from master*/
		MPI_Recv(param,paramSize,MPI_FLOAT,ROOT,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
        count++;
        //printf("%d:%d\n", rank, count);
        
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
        // printf("Slave[%d] cost: %f\n", rank, cost);

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
