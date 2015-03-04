#ifndef __MASTER_H__
#define __MASTER_H__

#include<mpi.h>
#include<stdio.h>

#define ROOT 0

#define WORKTAG 1
#define STOPTAG 2

struct masterConfInfo {
	int paramSize;
	int nIterMax;
	float learningRate;
	float initRange;
};

void masterFunc ();

void loadConf ();

#endif