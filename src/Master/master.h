#include<mpi.h>
#include<stdio.h>

#define ROOT 0

#define WORKTAG 1
#define STOPTAG 2

void masterFunc();

void loadConf();