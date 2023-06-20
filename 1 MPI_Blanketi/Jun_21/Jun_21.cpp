#include <mpi.h>
#include <iostream>
#include <limits>

#define POD_A true

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
	for (int i = 0; i < n; i++) {
		int el = 0;
		for (int j = 0; j < n; j++) {
			el += a[i * n + j] * b[j];
		}

		if (res[i] != el) return false;
	}

	return true;
}

int main(int argc, char** argv)
{
	int rank, size;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	constexpr int root = 0;
	constexpr int k = 8;
	int q = sqrt(size);
	int s = k / q;

	if (q * q != size || s * q != k) exit(1);

	int* a = new int[k * k];
	int* b = new int[k];
	int* c = new int[k * k];
	int* tile_a = new int[s * s];
	int* partial_b = new int[s];
	int* partial_c = new int[s];
	int* partial_c_sum = new int[s];

	if (rank == root) {
		for (int i = 0; i < k * k; i++) {
			a[i] = i;
		}

		for (int i = 0; i < k; i++) {
			b[i] = i;
		}

		//std::cout << "A:\n";
		//print_matrix(a, k, k);
		//
		//std::cout << "B:\n";
		//print_matrix(b, 1, k);
	}

	MPI_Datatype vec_type;
	MPI_Type_vector(s, s, q * k, MPI_INT, &vec_type);

#if POD_A
	MPI_Type_commit(&vec_type);

	if (rank == root) {
		int row = (rank / q) * s, col = (rank % q) * s;
		for (int i = 0; i < s; i++) {
			for (int j = 0; j < s; j++) {
				tile_a[i * s + j] = a[(row + i * q) * k + (col + j)];
			}
		}

		for (int i = 1; i < size; i++) {
			MPI_Send(&a[i * s], 1, vec_type, i, 0, MPI_COMM_WORLD);
		}
	}
	else {
		MPI_Recv(tile_a, s * s, MPI_INT, root, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
#else
	MPI_Datatype vec_extended_type;
	MPI_Type_create_resized(vec_type, 0, s * sizeof(int), &vec_extended_type);
	MPI_Type_commit(&vec_extended_type);

	MPI_Scatter(a, 1, vec_extended_type, a_tile, s * s, MPI_INT, root, MPI_COMM_WORLD);
#endif

	//std::cout << "Rank: " << rank << "\n";
	//print_matrix(a_tile, s, s);

	MPI_Comm comm_row, comm_col;
	int rank_row, rank_col;
	MPI_Comm_split(MPI_COMM_WORLD, rank / q, rank % q, &comm_row);
	MPI_Comm_split(MPI_COMM_WORLD, rank % q, rank / q, &comm_col);
	MPI_Comm_rank(comm_row, &rank_row);
	MPI_Comm_rank(comm_col, &rank_col);

	if (rank_col == 0) {
		MPI_Scatter(b, s, MPI_INT, partial_b, s, MPI_INT, root, comm_row);
	}
	MPI_Bcast(partial_b, s, MPI_INT, root, comm_col);

	struct {
		int val;
		int rank;
	} local_min{ tile_a[0], rank }, global_min;

	for (int i = 1; i < s * s; i++) {
		if (tile_a[i] < local_min.val) local_min.val = tile_a[i];
	}

	for (int i = 0; i < s; i++) {
		partial_c[i] = 0;
		for (int j = 0; j < s; j++) {
			partial_c[i] += tile_a[i * s + j] * partial_b[j];
		}
	}

	MPI_Reduce(&local_min, &global_min, 1, MPI_2INT, MPI_MINLOC, root, MPI_COMM_WORLD);
	MPI_Bcast(&global_min, 1, MPI_2INT, root, MPI_COMM_WORLD);

	//if (rank == 8) {
	//	std::cout << "Rank: " << global_min.rank << ", Value: " << global_min.val << "\n";
	//}

	MPI_Reduce(partial_c, partial_c_sum, s, MPI_INT, MPI_SUM, global_min.rank % q, comm_row);

	if (rank_row == global_min.rank % q) {
		MPI_Datatype recv_type, recv_type_extended;
		MPI_Type_vector(s, 1, q, MPI_INT, &recv_type);
		MPI_Type_create_resized(recv_type, 0, sizeof(int), &recv_type_extended);
		MPI_Type_commit(&recv_type_extended);

		MPI_Gather(partial_c_sum, s, MPI_INT, c, 1, recv_type_extended, global_min.rank / q, comm_col);
	}

	if (rank == global_min.rank) {
		std::cout << "C:\n";
		print_matrix(c, 1, k);
		std::cout << (check_result(a, b, c, k) ? "CORRECT" : "INCORRECT") << "\n";
	}

	delete[] a;
	delete[] b;
	delete[] c;
	delete[] tile_a;
	delete[] partial_b;
	delete[] partial_c;
	delete[] partial_c_sum;

	MPI_Finalize();
}