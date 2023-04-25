#include <stdio.h>
#include <mpi.h>

int main(int argc, char** argv)
{
	MPI_Status stat;
	MPI_Request req;

	int rank, size;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	constexpr int n = 8, p = 4;
	int A[n][n], B[n][n];

	if (rank == 0) {
		printf("Matrix A:\n");
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				A[i][j] = i * n + j;
				printf("%d\t", A[i][j]);
			}
			printf("\n");
		}
		printf("\n\n");
	}

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			B[i][j] = 0;
		}
	}

	if (rank == 0) {
		for (int i = 0; i < p; i++) {
			MPI_Datatype vec_type;
			MPI_Type_vector(n - i, 1, n + 1, MPI_INT, &vec_type);
			MPI_Type_commit(&vec_type);

			MPI_Isend(&A[0][i], 1, vec_type, i, 0, MPI_COMM_WORLD, &req);
			MPI_Isend(&A[i][0], 1, vec_type, i, 0, MPI_COMM_WORLD, &req);
		}
	}

	MPI_Recv(&B[0][0], n - rank, MPI_INT, 0, 0, MPI_COMM_WORLD, &stat);
	MPI_Recv(&B[1][0], n - rank, MPI_INT, 0, 0, MPI_COMM_WORLD, &stat);

	printf("P[%d]\n", rank);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			printf("%d\t", B[i][j]);
		}
		printf("\n");
	}
	printf("\n\n");

	MPI_Finalize();
}