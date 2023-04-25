#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define MASTER 0

void init_matrix(int* A, int n, int m)
{
	for (int i = 0; i < n; i++)
		for (int j = 0; j < m; j++)
			A[i * m + j] = i * m + j;
}

bool check_result(int* A, int* B, int* C, int n, int m, int k)
{
	bool ret = true;

	for (int i = 0; ret && i < n; i++)
	{
		for (int j = 0; ret && j < k; j++)
		{
			int temp = 0;
			for (int p = 0; p < m; p++)
				temp += A[i * m + p] * B[p * k + j];
			ret = temp == C[i * k + j];
		}
	}

	return ret;
}

void print_matrix(const char* lbl, int* A, int n, int m)
{
	printf("\n\n%s", lbl);
	for (int i = 0; i < n; i++)
	{
		printf("|\t");
		for (int j = 0; j < m; j++)
			printf("%d\t", A[i * m + j]);
		printf("|\n");
	}
}

int main(int argc, char** argv) {
	int rank, p;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	constexpr int k = 64, n = 32, m = 16;
	int A[k][n], B[n][m], C[k][m];
	int* local_A = new int[k / p * n], * local_C = new int[k / p * m];

	if (k % p != 0) exit(1);

	if (rank == MASTER) {
		init_matrix(&A[0][0], k, n);
		init_matrix(&B[0][0], n, m);
	}

	MPI_Datatype tmp_vec, send_type, recv_type;
	MPI_Type_vector(k / p, n, n * p, MPI_INT, &tmp_vec);
	MPI_Type_create_resized(tmp_vec, 0, n * sizeof(int), &send_type);
	MPI_Type_commit(&send_type);

	MPI_Scatter(&A[0][0], 1, send_type, local_A, k / p * n, MPI_INT, MASTER, MPI_COMM_WORLD);
	MPI_Bcast(&B[0][0], n * m, MPI_INT, MASTER, MPI_COMM_WORLD);

	for (int i = 0; i < k / p; i++) {
		for (int j = 0; j < m; j++) {
			local_C[i * m + j] = 0;
			for (int k = 0; k < n; k++) {
				local_C[i * m + j] += local_A[i * n + k] * B[k][j];
			}
		}
	}

	MPI_Type_vector(k / p, m, m * p, MPI_INT, &tmp_vec);
	MPI_Type_create_resized(tmp_vec, 0, m * sizeof(int), &recv_type);
	MPI_Type_commit(&recv_type);

	MPI_Gather(local_C, k / p * m, MPI_INT, &C[0][0], 1, recv_type, MASTER, MPI_COMM_WORLD);

	if (rank == 0) {
		if (check_result(&A[0][0], &B[0][0], &C[0][0], k, n, m)) {
			printf("\n\nCORRECT\n\n");
		}
		else {
			printf("\n\nINCORRECT\n\n");
		}
	}

	delete[] local_A;
	delete[] local_C;

	MPI_Finalize();
}