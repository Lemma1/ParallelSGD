#include<mpi.h>
#include<stdio.h>

#include "slave.h"
#include "SlaveConfig.h"

int slaveLoad(slaveConfinfo *sconfig)
{
    sconfig->paramSize = getSlaveIntConf("parameter size");
    sconfig->algorithmType = getSlaveIntConf("algorithm");
    //sconfig->
	return 1;
}

//the main function of slaves
void slaveDo(){
    MPI_Status status;
	//step 1:: configulation
    slaveConfinfo sconfig;

    if(~slaveLoad(&sconfig))
        return -1;	
    float *param = new float[sconfig.paramSize]; 
    int paramSize;
    //step 1.5:receive some 
    MPI_Bcast(&paramSize,1,MPI_INT,ROOT,MPI_COMM_WORLD);

	//main loo
    while(1){

		/*step 2:receive from master*/
		MPI_Recv(param,sconfig.paramSize,MPI_FLOAT,ROOT,MPI_ANY_TAG,MPI_COMM_WORLD,&status);

		/*step 3: check whether ends*/
		if(status.tag == STOPTAG){
        break;
        } 
		
        /*step 4: calculation*/
        /*TODO*/

        /*step 5: return to master*/
        MPI_Send(&param,sconfig.paramSize,MPI_FLOAT,ROOT,MPI_ANY_TAG,MPI_COMM_WORLD);

	}
    delete [] params;
}
