// %%cu
#include <cuda.h>
#include <vector>
#include <iostream>
#include <stdio.h>

#ifndef max
#define max(a, b) (((a) > (b)) ? (a) : (b))
#endif
#ifndef min
#define min(a, b) (((a) < (b)) ? (a) : (b))
#endif

#define BLOCK_SIZE 256
#define BLOCK_DIM_SIZE 16

#define N 156
#define M 128

void print_result(const std::string &header, float ms, bool result)
{
	std::cout << header << ": " << ms << "ms\t" << (result ? "CORRECT" : "INCORRECT") << "\n";
}

void init_matrix(int *mat)
{
	for (int i = 0; i < N * M; i++)
	{
		mat[i] = i + 1;
	}
}

template <typename T>
void create_matrix(T *&h_out, T *&d_out, size_t length)
{
	h_out = new T[length];
	cudaMalloc((void **)&d_out, length * sizeof(T));
}

template <typename T>
void cleanup(std::vector<T *> device_vec, std::vector<T *> host_vec)
{
	for (auto &d_arr : device_vec)
	{
		cudaFree(d_arr);
	}

	for (auto &h_arr : host_vec)
	{
		delete[] h_arr;
	}
}

bool check_result(int *mat1, int *mat2, float *res)
{
	for (int j = 0; j < M; j++)
	{
		float avg = 0;
		for (int i = 0; i < M; i++)
		{
			avg += mat1[i * M + j] + mat2[i * M + j];
		}
		avg /= N;

		if (avg != res[j])
			return false;
	}

	return true;
}

bool check_addition(int *a, int *b, int *res, bool print = false, int thresh = 20)
{
	if (print)
	{
		bool match = true;

		for (int i = 0; i < N * M; i++)
		{
			if (res[i] != a[i] + b[i])
			{
				match = false;
				if (i < thresh)
				{
					printf("i = %d\t a[i] = %d\tb[i] = %d\tres[i] = %d\n", i, a[i], b[i], res[i]);
				}
			}
		}

		return match;
	}
	else
	{
		for (int i = 0; i < N * M; i++)
		{
			if (res[i] != a[i] + b[i])
				return false;
		}

		return true;
	}
}

bool check_col_avg(int *mat, float *res, bool print = false, int thresh = 20)
{
	if (print)
	{
		bool match = true;

		for (int j = 0; j < M; j++)
		{
			float avg = 0;
			for (int i = 0; i < N; i++)
			{
				avg += mat[i * M + j];
			}
			avg /= N;

			if (avg != res[j])
			{
				match = false;
				if (j < thresh)
				{
					printf("i = %d\t local[i] = %f\tres[i] = %f\n", j, avg, res[j]);
				}
			}
		}

		return match;
	}
	else
	{
		for (int j = 0; j < M; j++)
		{
			float avg = 0;
			for (int i = 0; i < N; i++)
			{
				avg += mat[i * M + j];
			}
			avg /= N;

			if (avg != res[j])
				return false;
		}

		return true;
	}
}

__global__ void add_by_row(int *a, int *b, int *out)
{
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;

	for (int i = row; i < N; i += gridDim.y * blockDim.y)
	{
		for (int j = col; j < M; j += gridDim.x * blockDim.x)
		{
			out[i * M + j] = a[i * M + j] + b[i * M + j];
		}
	}
}

__global__ void add_by_col(int *a, int *b, int *out)
{
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;

	for (int j = col; j < M; j += gridDim.x * blockDim.x)
	{
		for (int i = row; i < N; i += gridDim.y * blockDim.y)
		{
			out[i * M + j] = a[i * M + j] + b[i * M + j];
		}
	}
}

__global__ void add_1d(int *a, int *b, int *out)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	while (tid < N * M)
	{
		out[tid] = a[tid] + b[tid];
		tid += gridDim.x * blockDim.x;
	}
}

void compare_addition()
{
	constexpr size_t mat_len = N * M;
	constexpr size_t mat_size = mat_len * sizeof(int);

	int *h_a, *h_b, *h_out;
	int *d_a, *d_b, *d_out;
	create_matrix(h_a, d_a, mat_len);
	create_matrix(h_b, d_b, mat_len);
	create_matrix(h_out, d_out, mat_len);
	init_matrix(h_a);
	init_matrix(h_b);
	cudaMemcpy(d_a, h_a, mat_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, mat_size, cudaMemcpyHostToDevice);

	dim3 blockSize(BLOCK_DIM_SIZE, BLOCK_DIM_SIZE);
	dim3 gridSize(min(256, (M + blockSize.y - 1) / blockSize.y), min(256, (N + blockSize.x - 1) / blockSize.x));

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float ms = 0;

	// add by row
	cudaEventRecord(start);
	// add_by_row<<<gridSize, blockSize>>>(d_a, d_b, d_out);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms, start, stop);
	cudaMemcpy(h_out, d_out, mat_size, cudaMemcpyDeviceToHost);
	print_result("Add by row", ms, check_addition(h_a, h_b, h_out));

	// add by col
	ms = 0;
	cudaMemset(d_out, 0, mat_size);
	cudaEventRecord(start);
	// add_by_col<<<gridSize, blockSize>>>(d_a, d_b, d_out);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms, start, stop);
	cudaMemcpy(h_out, d_out, mat_size, cudaMemcpyDeviceToHost);
	print_result("Add by col", ms, check_addition(h_a, h_b, h_out));

	// add 1D
	ms = 0;
	cudaMemset(d_out, 0, mat_size);
	cudaEventRecord(start);
	// add_1d<<<(N * M) / BLOCK_SIZE, BLOCK_SIZE>>>(d_a, d_b, d_out);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms, start, stop);
	cudaMemcpy(h_out, d_out, mat_size, cudaMemcpyDeviceToHost);
	print_result("Add 1D", ms, check_addition(h_a, h_b, h_out));

	cleanup<int>({d_a, d_b, d_out}, {h_a, h_b, h_out});
}

__global__ void col_avg_1d(int *matrix, float *out)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	while (tid < M)
	{
		int sum = 0;
		for (int i = 0; i < N; i++)
		{
			sum += matrix[i * M + tid];
		}
		out[tid] = sum / N;

		tid += gridDim.x * blockDim.x;
	}
}

__global__ void col_avg_reduction(int *matrix, int *out)
{
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;

	__shared__ int sh_mem[BLOCK_DIM_SIZE][BLOCK_DIM_SIZE];

	for (int j = col; j < M; j += blockDim.x * gridDim.x)
	{
		sh_mem[threadIdx.y][threadIdx.x] = 0;
		for (int i = row; i < N; i += blockDim.y * gridDim.y)
		{
			sh_mem[threadIdx.y][threadIdx.x] += matrix[i * M + j];
		}

		__syncthreads();

		for (int s = blockDim.y >> 2; s > 0; s >>= 1)
		{
			if (threadIdx.y < s)
			{
				sh_mem[threadIdx.y][threadIdx.x] += sh_mem[threadIdx.y + s][threadIdx.x];
			}

			__syncthreads();
		}
	}
}

void compare_col_avg()
{
	constexpr size_t vec_size = M * sizeof(float);
	constexpr size_t mat_size = N * M * sizeof(int);

	int *h_in, *d_in;
	float *h_out, *d_out;
	create_matrix(h_in, d_in, N * M);
	create_matrix(h_out, d_out, M);
	init_matrix(h_in);
	cudaMemcpy(d_in, h_in, mat_size, cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float ms = 0;

	// 1D
	cudaEventRecord(start);
	// col_avg_1d<<<1, M>>>(d_in, d_out);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms, start, stop);
	cudaMemcpy(h_out, d_out, vec_size, cudaMemcpyDeviceToHost);
	print_result("Avg 1D", ms, check_col_avg(h_in, h_out));

	// reduction
	ms = 0;
	cudaMemset(d_out, 0, vec_size);
	cudaEventRecord(start);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms, start, stop);
	cudaMemcpy(h_out, d_out, vec_size, cudaMemcpyDeviceToHost);
	print_result("Avg reduction", ms, check_col_avg(h_in, h_out));

	cleanup<int>({d_in}, {h_in});
	cleanup<float>({d_out}, {h_out});
}

int main()
{
	compare_addition();
	compare_col_avg();
}