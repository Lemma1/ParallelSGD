#include<mpi.h>
#include<stdio.h>


#define WORKTAG 1
#define STOPTAG 2
#define ROOT 0


struct slaveConfinfo{
    int paramSize;
    int algorithmType;
    //TODO
    
};
void slaveDo();
int slaveLoad(*slaveConfinfo);//return 0 if fail




