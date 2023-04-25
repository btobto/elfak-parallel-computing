#include <mpi.h>
#include <math.h>
#include <stdio.h>

int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);

	int rank, size;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int n = sqrt(size);
	int* members = new int[n];

	for (int i = 0; i < n; i++) {
		members[i] = i * (n + 1);
	}

	MPI_Group world_group, diag_group;
	MPI_Comm diag_comm;

	MPI_Comm_group(MPI_COMM_WORLD, &world_group);
	MPI_Group_incl(world_group, n, members, &diag_group);
	MPI_Comm_create(MPI_COMM_WORLD, diag_group, &diag_comm);

	int diag_rank;
	MPI_Group_rank(diag_group, &diag_rank);

	if (diag_rank != MPI_UNDEFINED) {
		int value;

		if (diag_rank == 0) {
			value = 5;
			for (int i = 1; i < n; i++) {
				MPI_Send(&value, 1, MPI_INT, i, 0, diag_comm);
			}
		}
		else {
			MPI_Recv(&value, 1, MPI_INT, 0, 0, diag_comm, MPI_STATUS_IGNORE);
		}

		printf("Old rank: [%d]\tNew rank: [%d]\tValue: [%d]\n", rank, diag_rank, value);
	}
		
	delete[] members;
	MPI_Finalize();
}