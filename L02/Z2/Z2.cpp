#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);

	int rank, size;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	constexpr int n = 4, m = 3, dim_count = 2, displacement = 2;
	int dims[] = { n, m }, periods[] = { 0, 1 }, coords[dim_count];
	int cart_rank, left_rank, right_rank, left_coords[dim_count], right_coords[dim_count];

	MPI_Comm cart_comm;

	MPI_Cart_create(MPI_COMM_WORLD, dim_count, dims, periods, 1, &cart_comm);
	MPI_Comm_rank(cart_comm, &cart_rank);
	MPI_Cart_coords(cart_comm, rank, dim_count, coords);

	MPI_Cart_shift(cart_comm, 1, displacement, &left_rank, &right_rank);

	printf("Rank: [%d]\tCoords: [%d][%d]\tLeft rank: [%d]\tRight rank: [%d]\n", cart_rank, coords[0], coords[1], left_rank, right_rank);

	MPI_Finalize();
}