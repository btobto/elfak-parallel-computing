#include <mpi.h>
#include <stdio.h>
#include <iostream>

int main(int argc, char** argv)
{
	MPI_Init(&argc, &argv);

	int rank, size;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	constexpr int n = 3, dim_count = 2;
	int dims[] = { n, n }, periods[] = { 0, 0 }, cart_coords[n];

	int cart_comm, cart_rank;
	MPI_Cart_create(MPI_COMM_WORLD, dim_count, dims, periods, 1, &cart_comm);
	MPI_Comm_rank(cart_comm, &cart_rank);
	MPI_Cart_coords(cart_comm, cart_rank, dim_count, cart_coords);

	MPI_Comm triangle_comm, triangle_rank;
	MPI_Comm_split(MPI_COMM_WORLD, cart_coords[0] >= cart_coords[1], rank, &triangle_comm);
	MPI_Comm_rank(triangle_comm, &triangle_rank);

	int upper_sum, partial_sum;
	MPI_Reduce(&cart_rank, &partial_sum, 1, MPI_INT, MPI_SUM, 0, triangle_comm);

	if (triangle_rank == 0 && rank != 0) {
		MPI_Send(&partial_sum, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
	}

	if (rank == 0) {
		MPI_Recv(&upper_sum, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		printf("Upper sum: %d\tLower sum: %d\n", upper_sum, partial_sum);
	}

	MPI_Finalize();
}