#include <iostream>
#include <mpi.h>
#include <vector>

int main(int argc, char** argv)
{
	int rank, size;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	constexpr int n = 105;
	int buf1[n];
	auto file1 = "file1.dat";
	auto file2 = "file2.dat";
	MPI_File fh;

	for (int i = 0; i < n; i++) {
		buf1[i] = rank;
	}

	// 1.
	MPI_Offset offset = (size - 1 - rank) * n * sizeof(int);
	MPI_File_open(MPI_COMM_WORLD, file1, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
	MPI_File_write_at_all(fh, offset, buf1, n, MPI_INT, MPI_STATUS_IGNORE);
	MPI_File_close(&fh);

	// 2.
	int buf2[n];

	MPI_File_open(MPI_COMM_WORLD, file1, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
	MPI_File_read_at_all(fh, offset, buf2, n, MPI_INT, MPI_STATUS_IGNORE);
	MPI_File_close(&fh);

	bool match = true;
	for (int i = 0; i < n; i++) {
		if (buf1[i] != buf2[i]) {
			match = false;
			break;
		}
	}
	printf("Rank %d: %s\n", rank, match ? "Correct" : "Incorrect");

	// 3.
	std::vector<int> block_lengths;
	std::vector<int> displacements;
	for (
		int d = 0, l = 1;
		d < n; 
		d = displacements.back() + rank + l * size, l++
	) {
		block_lengths.push_back(l);
		displacements.push_back(d);
	}
	int count = block_lengths.size();

	if (rank == 0)
		for (int i = 0; i < count; i++) {
			printf("D: %d BL:%d\n", displacements[i], block_lengths[i]);
		}

	MPI_Datatype file_type;
	MPI_Type_indexed(count, block_lengths.data(), displacements.data(), MPI_INT, &file_type);
	MPI_Type_commit(&file_type);

	MPI_File_open(MPI_COMM_WORLD, file2, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
	MPI_File_set_view(fh, rank * sizeof(int), MPI_INT, file_type, "native", MPI_INFO_NULL);
	MPI_File_write_all(fh, buf2, n, MPI_INT, MPI_STATUS_IGNORE);
	MPI_File_close(&fh);

	MPI_Finalize();
}