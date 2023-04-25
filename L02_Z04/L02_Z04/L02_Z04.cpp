#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <set>
#include <vector>
#include <algorithm>
#include <iterator>

int main(int argc, char** argv)
{
	MPI_Init(&argc, &argv);

	int rank, size;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	int n = sqrt(size);

	std::set<int> members_set;
	for (int i = 0; i < n; i++) {
		members_set.insert(i * (n + 1));
		members_set.insert((i + 1) * (n - 1));
	}

	std::vector<int> members;
	members.reserve(members_set.size());
	std::copy(members_set.begin(), members_set.end(), std::back_inserter(members));

	MPI_Group world_group, diag_group;
	MPI_Comm diag_comm;

	MPI_Comm_group(MPI_COMM_WORLD, &world_group);
	MPI_Group_incl(world_group, members.size(), members.data(), &diag_group);
	MPI_Comm_create(MPI_COMM_WORLD, diag_group, &diag_comm);

	int diag_rank;
	MPI_Group_rank(diag_group, &diag_rank);

	if (diag_rank != MPI_UNDEFINED) {
		int value;

		if (diag_rank == 0) {
			value = 10;
			for (int i = 1; i < members.size(); i++) {
				MPI_Send(&value, 1, MPI_INT, i, 0, diag_comm);
			}
		}
		else {
			MPI_Recv(&value, 1, MPI_INT, 0, 0, diag_comm, MPI_STATUS_IGNORE);
		}

		printf("RANK: [%d]\tDIAG RANK: [%d]\tVAL: [%d]\n", rank, diag_rank, value);
	}

	MPI_Finalize();

}