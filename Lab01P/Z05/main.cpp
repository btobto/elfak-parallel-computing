#include <stdio.h>
#include <mpi.h>

int main(int argc, char** argv) {
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	constexpr int n = 8, m = 4, k = 12, p = 4;
	int A[m][n], B[n][k], C[m][k], tmp[m][k / p];

	if (rank == 0) {
		printf("Matrix A:\n");
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				A[i][j] = i * m + j;
				printf("%d\t", A[i][j]);
			}
			printf("\n");
		}
		printf("\n\n");

		printf("Matrix B:\n");
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < k; j++) {
				B[i][j] = i * n + j;
				printf("%d\t", B[i][j]);
			}
			printf("\n");
		}
		printf("\n\n");
	}


}