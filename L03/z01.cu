// %%cu
#include <cuda.h>
#include <iostream>
#include <stdio.h>

#define BLOCK_SIZE 256

__host__ void init_vec(int *vec, int n)
{
	for (int i = 0; i < n; i++)
	{
		vec[i] = i + 1;
	}
}

__host__ bool check_result(int *a, float *b, int n)
{
	// bool match = true;
	for (int i = 0; i < n; i++)
	{
		float res = (3 * a[i] + 10 * a[i + 1] + 7 * a[i + 2]) / 20.f;
		if (res != b[i])
		{
			// std::cout << "i: " << i << " res[i]: " << res << " B[i]: " << b[i] << "\n";
			// match = false;
			return false;
		}
	}
	// return match;
	return true;
}

__global__ void kernel(int *a, float *b, int n)
{
	__shared__ int temp[BLOCK_SIZE + 2];

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < n)
	{
		temp[threadIdx.x] = a[tid];

		if (threadIdx.x == BLOCK_SIZE - 1 || tid == n - 1)
		{
			temp[threadIdx.x + 1] = a[tid + 1];
			temp[threadIdx.x + 2] = a[tid + 2];
		}

		__syncthreads();

		b[tid] = (3 * temp[threadIdx.x] + 10 * temp[threadIdx.x + 1] + 7 * temp[threadIdx.x + 2]) / 20.f;
	}
}

int main()
{
	int n = 512; // scanf

	int *A = new int[n + 2];
	float *B = new float[n];

	int size_A = sizeof(int) * (n + 2);
	int size_B = sizeof(float) * n;

	init_vec(A, n + 2);

	int *d_A;
	float *d_B;
	cudaMalloc((void **)&d_A, size_A);
	cudaMalloc((void **)&d_B, size_B);

	cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);

	kernel<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_A, d_B, n);

	cudaMemcpy(B, d_B, size_B, cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_B);

	std::cout << (check_result(A, B, n) ? "CORRECT" : "INCORRECT") << "\n";

	delete[] A;
	delete[] B;
}