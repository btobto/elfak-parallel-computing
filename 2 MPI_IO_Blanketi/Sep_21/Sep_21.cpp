#include <iostream>
#include <mpi.h>
#include <math.h>

int main(int argc, char** argv)
{
	int rank, size;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	MPI_File fh;
	auto file1 = "file1.dat";
	auto file2 = "file2.dat";

	constexpr int count = 9;
	int buf1[count];

	for (int i = 0; i < count; i++) {
		buf1[i] = rank;
	}

	// 1.
	MPI_Offset offset = (size - 1 - rank) * count * sizeof(int);
	MPI_File_open(MPI_COMM_WORLD, file1, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
	MPI_File_seek(fh, offset, MPI_SEEK_SET);
	MPI_File_write_all(fh, buf1, count, MPI_INT, MPI_STATUS_IGNORE);
	MPI_File_close(&fh);

	// 2.
	int buf2[count];

	MPI_File_open(MPI_COMM_WORLD, file1, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
	MPI_File_read_at_all(fh, offset, buf2, count, MPI_INT, MPI_STATUS_IGNORE);
	MPI_File_close(&fh);

	bool match = true;
	for (int i = 0; i < count; i++) {
		if (buf1[i] != buf2[i]) {
			match = false;
			break;
		}
	}
	printf("Rank %d: %s\n", rank, match ? "Correct" : "Incorrect");

	// 3.
	constexpr int m = 6, n = 9;
	int sub_dim = sqrt(count);
	int gsizes[] = { m, n },
		distribs[] = { MPI_DISTRIBUTE_BLOCK, MPI_DISTRIBUTE_BLOCK },
		dargs[] = { MPI_DISTRIBUTE_DFLT_DARG, MPI_DISTRIBUTE_DFLT_DARG },
		psizes[] = { m / sub_dim, n / sub_dim };

	MPI_Datatype file_type;
	MPI_Type_create_darray(size, rank, 2, gsizes, distribs, dargs, psizes, MPI_ORDER_C, MPI_INT, &file_type);
	MPI_Type_commit(&file_type);

	MPI_File_open(MPI_COMM_WORLD, file2, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
	MPI_File_set_view(fh, 0, MPI_INT, file_type, "native", MPI_INFO_NULL);
	MPI_File_write_all(fh, buf2, count, MPI_INT, MPI_STATUS_IGNORE);
	MPI_File_close(&fh);

	MPI_Finalize();
	return 0;
}