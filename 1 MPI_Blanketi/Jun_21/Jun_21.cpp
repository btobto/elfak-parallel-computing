#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

/*
	Ne radi
*/

#define POD_A true
#define MASTER 0

bool check_result(int* matrix, int* vector, int* result, int n) {
	for (int i = 0; i < n; i++) {
		int val = 0;
		for (int j = 0; j < n; j++) {
			val += matrix[i * n + j] * vector[j];
			if (val != result[i]) return false;
		}
	}
	return true;
}

int main(int argc, char** argv) {
	int rank, size;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int q = sqrt(size);
	int k = q * 2;
	int s = k / q; // 2

	if (q * q != size) exit(1);

	int* A = new int[k * k], * b = new int[k], * c = new int[k];
	int* local_A = new int[s * s], * local_b = new int[s], * local_c = new int[s];

	if (rank == MASTER) {
		for (int i = 0; i < k * k; i++) {
			A[i] = i;
		}

		printf("MATRIX A:\n");
		for (int i = 0; i < k; i++) {
			for (int j = 0; j < k; j++) {
				printf("%d\t", A[i * k + j]);
			}
			printf("\n");
		}
		printf("\n");

		printf("VECTOR B:\n");
		for (int i = 0; i < k; i++) {
			b[i] = i;
			printf("%d\t", b[i]);
		}
		printf("\n\n");
	}

	MPI_Datatype vec_type;
	MPI_Type_vector(s, s, (q - 1) * k, MPI_INT, &vec_type);

	#if POD_A
		MPI_Type_commit(&vec_type);

		if (rank == MASTER) {
			MPI_Request req;

			for (int i = 0; i < size; i++) {
				MPI_Isend(&A[i * s], 1, vec_type, i, 0, MPI_COMM_WORLD, &req);
			}
		}

		MPI_Recv(local_A, s * s, MPI_INT, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	#else 
		MPI_Datatype vec_extended_type;
		MPI_Type_create_resized(vec_type, 0, s * sizeof(int), &vec_extended_type);
		MPI_Type_commit(&vec_extended_type);

		MPI_Scatter(A, 1, vec_extended_type, local_A, s * s, MPI_INT, MASTER, MPI_COMM_WORLD);
	#endif

	MPI_Comm col_comm, row_comm;
	int col_rank, row_rank;
	MPI_Comm_split(MPI_COMM_WORLD, rank / q, 0, &row_comm);
	MPI_Comm_split(MPI_COMM_WORLD, rank % q, 0, &col_comm);
	MPI_Comm_rank(row_comm, &row_rank);
	MPI_Comm_rank(col_comm, &col_rank);

	if (col_rank == 0) {
		MPI_Scatter(b, s, MPI_INT, local_b, s, MPI_INT, 0, row_comm);
	}

	MPI_Bcast(local_b, s, MPI_INT, 0, col_comm);

	struct {
		int min;
		int rank;
	} min_local{ local_A[0], rank }, min_global{};

	min_local.min = local_A[0];
	for (int i = 0; i < s; i++) {
		local_c[i] = 0;
		for (int j = 0; j < s; j++) {
			int index = i * s + j;
			local_c[i] += local_A[index] * local_b[j];

			if (local_A[index] < min_local.min) min_local.min = local_A[index];
		}
	}

	MPI_Reduce(&min_local, &min_global, 1, MPI_2INT, MPI_MINLOC, MASTER, MPI_COMM_WORLD);
	MPI_Bcast(&min_global, 1, MPI_2INT, MASTER, MPI_COMM_WORLD);

	int* partial_res = new int[s], * res = new int[k];
	MPI_Reduce(local_c, partial_res, s, MPI_INT, MPI_SUM, min_global.rank % q, row_comm);

	if (min_global.rank % q == col_rank) {
		MPI_Datatype tmp_vec, rearranged_vec_type;
		MPI_Type_vector(q, 1, s, MPI_INT, &tmp_vec);
		MPI_Type_create_resized(tmp_vec, 0, sizeof(int), &rearranged_vec_type);
		MPI_Type_commit(&rearranged_vec_type);

		MPI_Gather(partial_res, s, MPI_INT, res, 1, rearranged_vec_type, min_global.rank / q, col_comm);
	}

	if (rank == min_global.rank) {
		MPI_Send(res, k, MPI_INT, MASTER, 0, MPI_COMM_WORLD);
	}

	if (rank == MASTER) {
		MPI_Recv(res, k, MPI_INT, min_global.rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		if (check_result(A, b, res, k)) {
			printf("\n\nCORRECT\n\n");
		}
		else {
			printf("\n\nINCORRECT\n\n");
		}
	}

	delete[] A;
	delete[] b;
	delete[] c;

	MPI_Finalize();
}