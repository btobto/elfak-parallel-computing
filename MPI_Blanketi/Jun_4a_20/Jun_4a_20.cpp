#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
	int rank, size;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int dims[] = { 3, 4 }, periods[] = { 0, 1 }, coords[2];
	
	MPI_Comm cart_comm;
	MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Cart_coords(cart_comm, rank, 2, coords);

	int x = coords[0] + coords[1], left, right;
	int old_x = x;

	MPI_Cart_shift(cart_comm, 1, coords[0], &left, &right);
	MPI_Sendrecv_replace(&x, 1, MPI_INT, right, 0, left, 0, cart_comm, MPI_STATUS_IGNORE);

	printf("Rank: %d\tCoords: %d, %d\tOld X: %d\tNew X: %d", rank, coords[0], coords[1], old_x, x);

	MPI_Finalize();
}