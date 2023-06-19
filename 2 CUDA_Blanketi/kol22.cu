#include <cuda.h>
#include <cuda_runtime.h>

#define p 0.3f
#define BLOCK_NUM 256
#define BLOCK_SIZE 256

__global__ void kernel(int *a, int *b, int *c, int n)
{
	__shared__ int sh_a[BLOCK_SIZE];
	__shared__ int sh_b[BLOCK_SIZE];

	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	while (tid < n)
	{
		sh_a[threadIdx.x] = a[tid];
		sh_b[threadIdx.x] = b[tid];

		__syncthreads();

		if (tid < n - 2)
		{
			float res;

			if (threadIdx.x < blockDim.x - 2)
			{
				res = (sh_a[threadIdx.x] + sh_a[threadIdx.x + 1] + sh_a[threadIdx.x + 2]) * p + (sb_b[threadIdx.x]) * (p - 1);
			}
			else if (threadIdx.x < blockDim.x - 1)
			{
				// res = ...
			}
			else
			{
				// res = ...
			}

			c[tid] = res;
		}

		__syncthreads();

		tid += blockDim.x * gridDim.x;
	}
}

int main()
{
	int *a, *b;
	int *d_a, *d_b;
	float *c, d_c;

	int n = 255;

	a = new int[n];
	b = new int[n];
	c = new float[n - 2];

	cudaMalloc((void **)&d_a, n * sizeof(int));
	cudaMalloc((void **)&d_b, n * sizeof(int));
	cudaMalloc((void **)&d_c, (n - 2) * sizeof(float));

	for (int i = 0; i < n; i++)
	{
		a[i] = i + 1;
		b[i] = i + 1;
	}

	cudaMemcpy(d_a, a, n * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, n * sizeof(int), cudaMemcpyHostToDevice);

	kernel<<<BLOCK_NUM, BLOCK_SIZE>>>(d_a, d_b, d_c, n);

	cudaMemcpy(c, d_c, (n - 2) * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	delete[] a;
	delete[] b;
	delete[] c;
}