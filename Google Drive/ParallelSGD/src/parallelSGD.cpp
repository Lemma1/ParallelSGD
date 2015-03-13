#include "master.h"
#include "slave.h"

int main(int argc, char ** argv) {
	MPI_Init(&argc, &argv);
	int worldSize, worldRank;
	MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
	MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

	if (worldRank == ROOT) {
		masterFunc();
	} else {
		slaveDo();
	}

	MPI_Finalize();
	return 0;
}