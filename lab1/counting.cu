#include "counting.h"
#include <cstdio>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__global__ void init(const char *, int *, int);
__global__ void kernel(const char *, int *, int, int);

void CountPosition(const char *text, int *pos, int text_size)//part1
{
	const int k = 500;
	const int blocksize = 512;
	int nblock = text_size / blocksize + (text_size % blocksize == 0?0:1);
	init<<<nblock, blocksize>>>(text, pos, text_size);
	cudaDeviceSynchronize();
    for (int i = 2; i <= k; i++) 
	{
      nblock = (text_size / i + 1) / blocksize + (text_size % blocksize == 0?0:1);
      kernel<<<nblock, blocksize>>>(text, pos, text_size, i);
      cudaDeviceSynchronize();
    }
}

 __global__ void init(const char * text, int * pos, int n) //using time complexity nlogk algorithm
 {
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx >= n) //out of boundry
   {
     return;
   }
   else
   {
			if (text[idx] == '\n') //white space output 0
			{
				pos[idx] = 0;
			} 
			else //not white space output 1
			{
				pos[idx] = 1;
			}
   }
 }
__global__ void kernel(const char * text, int * pos, int n, int k) 
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   idx = idx * k;
   if (idx >= n) {
     return;
   }
   if (pos[idx] == 0) {
     return;
   }
   if (idx > 0 && pos[idx-1] == k-1) {
     pos[idx] = k;
     return;
   }
   int i = pos[idx];
   int j = idx + (k - i - 1);
   if (j+1 < n && pos[j] == k-1 && text[j+1] != '\n') 
   {
     pos[j+1] = k;
   }
 }


 
struct is_One
{
	__host__ __device__
	bool operator()(const int x)
	{
		return x == 1;
	}
};

int ExtractHead(const int *pos, int *head, int text_size)//part2
{
	int *buffer;
	int nhead;
	cudaMalloc(&buffer, sizeof(int)*text_size*2); // this is enough
	thrust::device_ptr<const int> pos_d(pos);
	thrust::device_ptr<int> head_d(head), flag_d(buffer), cumsum_d(buffer+text_size);

	// TODO
	auto head_end_d = thrust::copy_if(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(text_size), pos_d, head_d, is_One());
	nhead = head_end_d - head_d;
	cudaFree(buffer);
	return nhead;
}

//Convert all characters to be capital
__global__ void Transform_1(char *input_gpu, int fsize) 
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < fsize and 97 <= input_gpu[idx] and input_gpu[idx] <= 122) 
	{
		input_gpu[idx] = input_gpu[idx] - 32;
	}
}

//Swap all pairs in all words
__global__ void Transform_2(char *input_gpu, int fsize) 
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	char temp;
	if (idx < fsize and input_gpu[idx] != '\n' and input_gpu[idx+1] != '\n') 
	{
		temp = input_gpu[idx];
		input_gpu[idx] = input_gpu[idx+1];
		input_gpu[idx+1] = temp;
		idx = idx + 2;
	}
	else if(idx < fsize and input_gpu[idx] != '\n' and input_gpu[idx+1] == '\n')
	{
		input_gpu[idx] = input_gpu[idx];
	}
}

void Part3(char *text, int *pos, int *head, int text_size, int n_head)
{
	Transform_1<<<8, 32>>>(text, text_size);
	Transform_2<<<8, 32>>>(text, text_size);
}
