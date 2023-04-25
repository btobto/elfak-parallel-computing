#include <mpi.h>
#include <stdio.h>
#include <iostream>

int main(int argc, char** argv)
{
	MPI_Init(&argc, &argv);

	int rank, size;
	
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	MPI_Comm comm1;
	MPI_Comm_split(MPI_COMM_WORLD, rank % 3, rank, &comm1);

	constexpr int v = 150;

	if (rank % 3 == 0) {
		int rank_comm1, size_comm1;
		MPI_Comm_rank(comm1, &rank_comm1);
		MPI_Comm_size(comm1, &size_comm1);

		int matrix_size = size_comm1 * size_comm1;

		int* A = new int[matrix_size];
		int* row_A = new int[size_comm1];

		if (rank == 0) {
			for (int i = 0; i < matrix_size; i++) {
				A[i] = i;
			}

			printf("\nMatrica A:\n");
			for (int i = 0; i < size_comm1; i++) {
				for (int j = 0; j < size_comm1; j++) {
					printf("%d\t", A[i * size_comm1 + j]);
				}
				printf("\n");
			}
			printf("\n");
		}

		MPI_Scatter(A, size_comm1, MPI_INT, row_A, size_comm1, MPI_INT, 0, comm1);

		int sum = 0;
		for (int i = 0; i < size_comm1; i++) {
			sum += row_A[i];
		}

		if (sum < v) {
			printf("Rank: [%d]\tNew rank: [%d]\tSum: [%d]\n", rank, rank_comm1, sum);
		}

		delete[] row_A;
		delete[] A;
	}

	MPI_Finalize();
}
