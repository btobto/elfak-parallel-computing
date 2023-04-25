#include <mpi.h>
#include <stdio.h>
#include <iostream>

int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);

	int rank, size;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	constexpr int n = 4, m = 3, dim_count = 2;
	int dims[] = { 4, 3 }, periods[] = { 0, 0 }, cart_coords[dim_count];
	int cart_rank, col_rank;
	MPI_Comm cart_comm, col_comm;

	MPI_Cart_create(MPI_COMM_WORLD, dim_count, dims, periods, 1, &cart_comm);
	MPI_Comm_rank(cart_comm, &cart_rank);
	MPI_Cart_coords(cart_comm, cart_rank, dim_count, cart_coords);

	MPI_Comm_split(cart_comm, cart_coords[0], cart_coords[1], &col_comm);
	MPI_Comm_rank(col_comm, &col_rank);

	if (col_rank == m - 1) {
		//MPI_Send(&cart_coords, 2, MPI_INT, 0, 0, MPI_COMM_WORLD);
	}


	if (rank == 0) {
		int coords[2];
		for (int i = 0; i < n; i++) {
			//MPI_Recv(&coords, 2, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			//printf("Coords: [%d][%d]\n", coords[0], coords[1]);
		}
	}
	printf("RANK: %d, COL RANK: %d, COMM %d", rank, col_rank, col_comm);

	MPI_Finalize();
}