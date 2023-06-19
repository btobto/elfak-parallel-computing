#include <iostream>
#include <mpi.h>

void print_matrix(int* mat, int n, int m) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			std::cout << mat[i * m + j] << "\t";
		}
		std::cout << "\n";
	}
	std::cout << "\n";
}

bool check_result(int* a, int* b, int* res, int n) {
	int* local_res = new int[n * n];
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			local_res[i * n + j] = 0;
			for (int k = 0; k < n; k++) {
				local_res[i * n + j] += a[i * n + k] * b[k * n + j];
			}
		}
	}

	bool match = true;
	for (int i = 0; i < n * n; i++) {
		if (local_res[i] != res[i]) {
			match = false;
			break;
		}
	}

	delete[] local_res;
	return match;
}

int main(int argc, char** argv)
{
	int rank, size;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	constexpr int root = 0;
	constexpr int n = 16;
	int q = sqrt(size);
	int k = n / q;

	if (q * q != size) exit(1);

	int* a = new int[n * n];
	int* b = new int[n * n];
	int* c = new int[n * n];
	int* tile_a = new int[k * k];
	int* tile_b = new int[k * k];
	int* rows_a = new int[k * n];
	int* cols_b = new int[n * k];
	int* partial_c = new int[k * k];
	int* rows_c = new int[k * n];

	MPI_Datatype tile_type;
	MPI_Type_vector(k, k, n, MPI_INT, &tile_type);
	MPI_Type_commit(&tile_type);

	if (rank == root) {
		for (int i = 0; i < n * n; i++) {
			a[i] = b[i] = i;
		}

		//std::cout << "A, B:\n";
		//print_matrix(a, n, n);

		int row = (root / q) * k, col = (root % q) * k;
		for (int i = 0; i < k; i++) {
			for (int j = 0; j < k; j++) {
				tile_a[i * k + j] = a[(row + i) * n + (col + j)];
				tile_b[i * k + j] = b[(row + i) * n + (col + j)];
			}
		}

		//print_matrix(tile_b, k, k);

		for (int i = 0; i < q; i++) {
			for (int j = 0; j < q; j++) {
				if (i + j != root) {
					MPI_Send(&a[i * k * n + j * k], 1, tile_type, i * q + j, 0, MPI_COMM_WORLD);
					MPI_Send(&b[i * k * n + j * k], 1, tile_type, i * q + j, 1, MPI_COMM_WORLD);
				}
			}
		}
	}
	else {
		MPI_Recv(tile_a, k * k, MPI_INT, root, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(tile_b, k * k, MPI_INT, root, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		//std::cout << "Rank " << rank << ", tile A, B\n";
		//print_matrix(tile_a, k, k);
	}

	MPI_Comm comm_col, comm_row;
	int rank_col, rank_row;
	MPI_Comm_split(MPI_COMM_WORLD, rank / q, rank % q, &comm_row);
	MPI_Comm_split(MPI_COMM_WORLD, rank % q, rank / q, &comm_col);
	MPI_Comm_rank(comm_row, &rank_row);
	MPI_Comm_rank(comm_col, &rank_col);

	MPI_Datatype resized_tile_type;
	MPI_Type_create_resized(tile_type, 0, k * sizeof(int), &resized_tile_type);
	MPI_Type_commit(&resized_tile_type);

	MPI_Gather(tile_a, k * k, MPI_INT, rows_a, 1, resized_tile_type, root, comm_row);
	MPI_Bcast(rows_a, k * n, MPI_INT, root, comm_row);

	//if (rank == 5) {
	//	print_matrix(rows_a, k, n);
	//}

	MPI_Gather(tile_b, k * k, MPI_INT, cols_b, k * k, MPI_INT, root, comm_col);
	MPI_Bcast(cols_b, n * k, MPI_INT, root, comm_col);

	//if (rank == 1) {
	//	print_matrix(cols_b, n, k);
	//}

	for (int i = 0; i < k; i++) {
		for (int j = 0; j < k; j++) {
			partial_c[i * k + j] = 0;
			for (int l = 0; l < n; l++) {
				partial_c[i * k + j] += rows_a[i * n + l] * cols_b[l * k + j];
			}
		}
	}

	MPI_Gather(partial_c, k * k, MPI_INT, rows_c, 1, resized_tile_type, root, comm_row);
	if (rank_row == root) {
		MPI_Gather(rows_c, k * n, MPI_INT, c, k * n, MPI_INT, root, comm_col);
	}

	if (rank == root) {
		std::cout << "C:\n";
		print_matrix(c, n, n);
		std::cout << (check_result(a, b, c, n) ? "CORRECT" : "INCORRECT") << "\n";
	}

	delete[] a;
	delete[] b;
	delete[] c;
	delete[] tile_a;
	delete[] tile_b;
	delete[] rows_a;
	delete[] cols_b;
	delete[] partial_c;
	delete[] rows_c;

	MPI_Finalize();
}