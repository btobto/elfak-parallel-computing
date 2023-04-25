#include <mpi.h>
#include <stdio.h>
#include <iostream>

int main(int argc, char** argv)
{
	MPI_Init(&argc, &argv);

	int rank, size;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	constexpr int n = 4, m = 3, dim_count = 2;
	int dims[] = { n, m }, periods[] = { 1, 0 };
	int upper, lower, sum;

	MPI_Comm cart_comm;
	MPI_Cart_create(MPI_COMM_WORLD, dim_count, dims, periods, 1, &cart_comm);
	MPI_Cart_shift(cart_comm, 0, 1, &upper, &lower);

	int partial_sum = upper + lower;
	MPI_Reduce(&partial_sum, &sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

	if (rank == 0) {
		printf("Sum: %d", sum);
	}

	MPI_Finalize();
}