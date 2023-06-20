#include <iostream>
#include <string>
#include <mpi.h>

int main(int argc, char** argv)
{
	int rank, size;
	MPI_File fh;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	//constexpr int n = 10;
	//char text[n];

	//for (int i = 0; i < n; i++) {
	//	text[i] = rank + i + '0';
	//}

	auto file1 = "file1.txt";
	auto file2 = "file2.txt";
	constexpr int n = 4;
	char text[n];
	for (int i = 0; i < n; i++) {
		text[i] = rank + '0';
	}

	// 1.
	MPI_File_open(MPI_COMM_WORLD, file1, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
	MPI_File_write_at_all(fh, rank * n * sizeof(char), text, n, MPI_CHAR, MPI_STATUS_IGNORE);
	MPI_File_close(&fh);

	// 2.
	MPI_File_open(MPI_COMM_WORLD, file1, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
	MPI_File_read_shared(fh, text, n, MPI_CHAR, MPI_STATUS_IGNORE);
	MPI_File_close(&fh);

	//printf("Rank: %d\n", rank);
	//for (int i = 0; i < n; i++) {
	//	printf("%c", text[i]);
	//}

	// 3.
	printf("Rank: %d\n", rank);

	char buf[2 * n];
	for (int i = 0; i < n; i++) {
		buf[i] = buf[2 * i] = text[i];
		printf("%c", buf[i]);
	}
	printf("\n\n");

	MPI_Datatype vec_type;
	MPI_Type_vector(n, 2, 2 * n, MPI_CHAR, &vec_type);
	MPI_Type_commit(&vec_type);

	MPI_File_open(MPI_COMM_WORLD, file2, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
	MPI_File_set_view(fh, 2 * rank * sizeof(char), MPI_CHAR, vec_type, "native", MPI_INFO_NULL);
	MPI_File_write_all(fh, buf, 2 * n, MPI_CHAR, MPI_STATUS_IGNORE);
	MPI_File_close(&fh);

	MPI_Finalize();
}