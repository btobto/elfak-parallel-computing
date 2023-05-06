#include <iostream>
#include <mpi.h>

int main(int argc, char** argv) {
	int rank, size;
	MPI_Status stat;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	constexpr int m = 4, n = 4, k = 8, l = 4;
	int A[m][n], B[n][k], C[m][k], tmp_B[n][k / l], tmp_C[m][k / l];

	if (rank == 0) {
		std::cout << "A:\n";
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				A[i][j] = i * n + j;
				std::cout << A[i][j] << "\t";
			}
			std::cout << "\n";
		}
		std::cout << "\n\n";

		std::cout << "B:\n";
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < k; j++) {
				B[i][j] = j;
				std::cout << B[i][j] << "\t";
			}
			std::cout << "\n";
		}
		std::cout << "\n\n";
	}

	MPI_Bcast(&A[0][0], m * n, MPI_INT, 0, MPI_COMM_WORLD);

	MPI_Datatype vec_type, vec_send_type;
	MPI_Type_vector(n, k / l, k, MPI_INT, &vec_type);
	MPI_Type_create_resized(vec_type, 0, (k / l) * sizeof(MPI_INT), &vec_send_type);
	MPI_Type_commit(&vec_send_type);

	MPI_Scatter(&B[0][0], 1, vec_send_type, &tmp_B, n * (k / l), MPI_INT, 0, MPI_COMM_WORLD);

	for (int i = 0; i < m; i++) {
		for (int j = 0; j < k / l; j++) {
			tmp_C[i][j] = 0;
			for (int z = 0; z < n; z++) {
				tmp_C[i][j] += A[i][z] * tmp_B[z][j];
			}
		}
	}

	MPI_Datatype vec_recv_type;
	MPI_Type_vector(m, k / l, k, MPI_INT, &vec_type);
	MPI_Type_create_resized(vec_type, 0, (k / l) * sizeof(MPI_INT), &vec_recv_type);
	MPI_Type_commit(&vec_recv_type);

	MPI_Gather(&tmp_C[0][0], m * (k / l), MPI_INT, &C[0][0], 1, vec_recv_type, 0, MPI_COMM_WORLD);

	if (rank == 0)
	{
		std::cout << "C:\n";
		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < k; j++) {
				std::cout << C[i][j] << "\t";
			}
			std::cout << "\n";
		}
		std::cout << "\n\n";
	}

	MPI_Finalize();
}