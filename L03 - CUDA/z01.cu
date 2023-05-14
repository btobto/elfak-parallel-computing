//%%cu
#include <cuda.h>
#include <iostream>
#include <vector>

#define PRINT false

#define BLOCK_SIZE 256

__host__ void init_vec(std::vector<int> &vec)
{
	for (int i = 0; i < vec.size(); i++)
	{
		vec[i] = i;
	}
}

__host__ bool check_result(int *a, float *res, int n)
{
	#if PRINT
		bool match = true;
	#endif

	for (int i = 0; i < n; i++)
	{
		float el = (3 * a[i] + 10 * a[i + 1] + 7 * a[i + 2]) / 20.f;
		if (el != res[i]) {
			#if PRINT
				std::cout << "i: " << i << " res[i]: " << res << " B[i]: " << b[i] << "\n";
				match = false;
			#else
				return false;
			#endif
		}
	}

	#if PRINT
		return match;
	#else
		return true;
	#endif
}

__global__ void kernel(int *a, float *b, int n)
{
	__shared__ int temp[BLOCK_SIZE + 2];

	int tid = threadIdx.x + blockIdx.x * blockDim.x;

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
	int n = 65536;

	std::vector<int> A(n + 2);
	std::vector<float> B(n);

	init_vec(A);

	int *d_a;
	float *d_b;
	size_t a_size = (n + 2) * sizeof(int);
	size_t b_size = n * sizeof(float);

	cudaMalloc((void **)&d_a, a_size);
	cudaMalloc((void **)&d_b, b_size);
	cudaMemcpy(d_a, A.data(), a_size, cudaMemcpyHostToDevice);

	kernel<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_a, d_b, n);

	cudaMemcpy(B.data(), d_b, b_size, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);

	std::cout << (check_result(A.data(), B.data(), n) ? "CORRECT" : "INCORRECT") << "\n";
}