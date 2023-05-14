#include <stdio.h>
#include <mpi.h>

int main(int argc, char** argv)
{
	int rank, size;
	constexpr int n = 16, p = 4;
	int A[n][n], T[n / p][n / 2];

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (rank == 0) {
		printf("P[%d], Matrix A:\n", rank);
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				A[i][j] = i * n + j;
				printf("%d\t", A[i][j]);
			}
			printf("\n");
		}
		printf("\n\n");
	}

	MPI_Datatype vec_type, vec_extended_type;
	MPI_Type_vector(n / 2, 1, 2, MPI_INT, &vec_type);
	MPI_Type_create_resized(vec_type, 0, n * sizeof(MPI_INT), &vec_extended_type);
	MPI_Type_commit(&vec_extended_type);
	
	MPI_Scatter(&A[0][0], n / p, vec_extended_type, &T[0][0], (n / p) * (n / 2), MPI_INT, 0, MPI_COMM_WORLD);

	printf("P[%d]\n", rank);
	for (int i = 0; i < n / p; i++) {
		for (int j = 0; j < n / 2; j++) {
			printf("%d\t", T[i][j]);
		}
		printf("\n");
	}
	printf("\n\n");

	struct {
		int min;
		int rank;
	} local_min, min;

	local_min = { T[0][0], rank };
	for (int i = 0; i < n / p; i++) {
		for (int j = 0; j < n / 2; j++) {
			local_min.min = local_min.min > T[i][j] ? T[i][j] : local_min.min;
		}
	}

	MPI_Reduce(&local_min, &min, 1, MPI_2INT, MPI_MINLOC, 0, MPI_COMM_WORLD);

	if (rank == 0) {
		printf("P[0]: min: %d, proc: %d", min.min, min.rank);
	}

	MPI_Finalize();
}