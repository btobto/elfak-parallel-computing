//%%cu
#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <limits.h>

#ifndef max
	#define max(a, b) (((a) > (b)) ? (a) : (b))
#endif
#ifndef min
	#define min(a, b) (((a) < (b)) ? (a) : (b))
#endif

#define N 128
#define M 32

__host__ void init_vec(std::vector<int> &vec)
{
	for (int i = 0; i < vec.size(); i++)
	{
		vec[i] = i + 1;
	}
}

__host__ bool check_result(int *mat, int *max, int *min)
{
	#if N < 20 && M < 20
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < M; j++)
			{
				std::cout << mat[i * M + j] << "\t";
			}
			std::cout << "\n";
		}

		std::cout << "MAX: ";
		for (int i = 0; i < M; i++)
		{
			std::cout << max[i] << " ";
		}

		std::cout << "\nMIN: ";
		for (int i = 0; i < M; i++)
		{
			std::cout << min[i] << " ";
		}
		std::cout << "\n";
	#endif

	for (int j = 0; j < M; j++)
	{
		int max_el = mat[j];
		int min_el = mat[j];
		for (int i = 1; i < N; i++)
		{
			int index = i * M + j;
			if (mat[index] > max_el)
				max_el = mat[index];
			if (mat[index] < min_el)
				min_el = mat[index];
		}
		if (max_el != max[j] || min_el != min[j])
			return false;
	}
	return true;
}

__global__ void kernel(int *mat, int *max, int *min)
{
	constexpr int dim_block = N / 2;

	__shared__ int sh_min[N / 2];
	__shared__ int sh_max[N / 2];

	// row = threadIdx.x, col = blockIdx.x
	int index = threadIdx.x * M + blockIdx.x;
	int first = mat[index];
	int second = mat[index + dim_block * M];

	sh_min[threadIdx.x] = min(first, second);
	sh_max[threadIdx.x] = max(first, second);

	__syncthreads();

	for (int s = dim_block / 2; s > 0; s /= 2)
	{
		if (threadIdx.x < s)
		{
			sh_min[threadIdx.x] = min(sh_min[threadIdx.x], sh_min[threadIdx.x + s]);
			sh_max[threadIdx.x] = max(sh_max[threadIdx.x], sh_max[threadIdx.x + s]);
		}

		__syncthreads();
	}

	if (threadIdx.x == 0)
	{
		min[blockIdx.x] = sh_min[0];
		max[blockIdx.x] = sh_max[0];
	}
}

int main()
{
	std::vector<int> a(N * M);
	std::vector<int> max_arr(M);
	std::vector<int> min_arr(M);

	init_vec(a);

	size_t mat_size = N * M * sizeof(int);
	size_t vec_size = M * sizeof(int);

	int *d_a, *d_max, *d_min;

	cudaMalloc((void **)&d_a, mat_size);
	cudaMalloc((void **)&d_max, vec_size);
	cudaMalloc((void **)&d_min, vec_size);

	cudaMemcpy(d_a, a.data(), mat_size, cudaMemcpyHostToDevice);

	kernel<<<M, N / 2>>>(d_a, d_max, d_min);

	cudaMemcpy(max_arr.data(), d_max, vec_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(min_arr.data(), d_min, vec_size, cudaMemcpyDeviceToHost);

	std::cout << (check_result(a.data(), max_arr.data(), min_arr.data()) ? "CORRECT" : "INCORRECT") << "\n";

	cudaFree(d_a);
	cudaFree(d_max);
	cudaFree(d_min);
}