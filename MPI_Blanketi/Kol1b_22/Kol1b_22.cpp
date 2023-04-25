#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int main(int argc, char** argv)
{
	int rank, size;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int q = sqrt(size);
	if (q * q != size) exit(1);
	
	int row = rank / q, col = rank % q;
	bool isDiagonal = row == col || row + col == q - 1;

	MPI_Comm diag_comm;
	MPI_Comm_split(MPI_COMM_WORLD, isDiagonal, 0, &diag_comm);

	if (isDiagonal) {
		int diag_rank, data;
		MPI_Comm_rank(diag_comm, &diag_rank);

		if (diag_rank == 0) {
			data = 5;
			int members_count = (q % 2 == 0) ? (2 * q) : (2 * q - 1);
			for (int i = 1; i < members_count; i++) {
				MPI_Send(&data, 1, MPI_INT, i, 0, diag_comm);
			}
		}
		else {
			MPI_Recv(&data, 1, MPI_INT, 0, 0, diag_comm, MPI_STATUS_IGNORE);
		}

		printf("RANK: %d\tDIAG RANK: %d\tVAL: %d\n", rank, diag_rank, data);
	}

	MPI_Finalize();
}