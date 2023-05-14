//%%cu
#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <vector>

// ne radi

#define BLOCK_SIZE 256
#define N 65536

__host__ int vec_sum(int *vec, int n)
{
	int loc_sum = 0;
	for (int i = 0; i < n; i++)
	{
		loc_sum += vec[i];
	}
	return loc_sum;
}

__host__ void init_vec(int *vec, int n)
{
	for (int i = 0; i < n; i++)
	{
		vec[i] = i;
	}
}

__global__ void reduce(int *vec, int *sum)
{
	__shared__ int sh_data[BLOCK_SIZE];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockDim.x * 2) + tid;
	unsigned int grid_size = blockDim.x * 2 * gridDim.x;

	sh_data[tid] = 0;

	while (i < N)
	{
		sh_data[tid] += vec[i] + vec[i + blockDim.x];
		i += grid_size;
	}

	__syncthreads();

	if (tid == 0)
	{
		sum[blockIdx.x] = sh_data[0];
	}
}

int main()
{
	std::vector<int> vec(N);
	std::vector<int> out(N);
	init_vec(vec.data(), vec.size());
	size_t size = N * sizeof(N);

	int *d_vec, *d_out;
	cudaMalloc((void **)&d_vec, size);
	cudaMalloc((void **)&d_out, size);

	cudaMemcpy(d_vec, vec.data(), size, cudaMemcpyHostToDevice);

	reduce<<<N / BLOCK_SIZE / 2, BLOCK_SIZE>>>(d_vec, d_out);

	cudaMemcpy(out.data(), d_vec, size, cudaMemcpyHostToDevice);

	std::cout << (vec_sum(vec.data(), N) == out[0] ? "CORRECT" : "INCORRECT") << "\n";

	cudaFree(d_vec);
	cudaFree(d_out);
}