#include <iostream>
#include <omp.h>
#include <vector>

void init_vec(std::vector<int>& vec) {
	for (int i = 0; i < vec.capacity(); i++) {
		vec.push_back(i + 1);
	}
}

int main() {
	int num_threads = omp_get_num_procs();
	omp_set_num_threads(num_threads);
	std::cout << "Number of threads: " << num_threads << "\n\n";

	std::cout << std::fixed;

	constexpr int n = 512, m = 128;
	size_t size = n * m;
	
	std::vector<int> mat_serial, mat_parallel;
	mat_serial.reserve(size);
	mat_parallel.reserve(size);
	init_vec(mat_serial);
	init_vec(mat_parallel);

	double start, end;

	start = omp_get_wtime();
	{
	}
	end = omp_get_wtime();
	std::cout << "Serial: " << end - start << "s\n";

	start = omp_get_wtime();
	{
	}
	end = omp_get_wtime();
	std::cout << "Parallel: " << end - start << "s\n\n";

}