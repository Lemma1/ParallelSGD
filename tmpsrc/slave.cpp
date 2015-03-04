#include<mpi.h>
#include<stdio.h>
#include<algorithm>

#include "slave.h"
#include "SlaveConfig.h"
#include "model.h"
#include "TestData.h"
#include "DataFactory.h"
/*
#include "../Model/model.h"
#include "../Data/TestData.h"
#include "../Data/DataFactory.h"
*/



//random pick the data 
//the main function of slaves


void slaveDo(){
    //step 0:init the data in local memory
    DataFactory *dataset = new TestData();
    
    int batchSize = 5;//TODO
    int dbSize = dataset->getNumberOfDate();// define in slave.h or ?
    
  //  dataInit(&dbSize,&batchSize);//TODO


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
    linearReg model = linearReg(paramSize,batchSize);
    for (int i=0;i<dbSize;i++){
        index[i]=i;
    }
  
	//main loop
    while(1){
		/*step 2:receive from master*/
		MPI_Recv(param,paramSize,MPI_FLOAT,ROOT,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
		/*step 3: check whether ends*/
		if(status.tag == STOPTAG){
        break;
        } 
        /*step 4: request for data*/
        random_shuffle(index,index+dbSize);
        for(int i=0;i<batchSize;i++){
            pickIndex[i] = index[i];
        }
        /*step 5: calculate the grad*/
        model.computeGrad(grad,param,data,label);
        /*step 6: return to master*/
        MPI_Send(grad,paramSize,MPI_FLOAT,ROOT,MPI_ANY_TAG,MPI_COMM_WORLD);
	}
    delete [] params;
    delete [] grad;
    delete [] label;
    delete [] data;
    delete [] index;
    delete data;
}
