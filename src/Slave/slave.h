#ifndef SLAVE_h
#define SLAVE_h

#include<mpi.h>
#include<stdio.h>


#define WORKTAG 1
#define STOPTAG 2
#define ROOT 0


struct slaveConfinfo{
    int paramSize;
    int algorithmType;
};

void slaveDo();
#endif

