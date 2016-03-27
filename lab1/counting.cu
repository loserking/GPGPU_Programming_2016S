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

__global__ void part1_init(const char *, int *, int);
__global__ void part1_kernel(const char *, int *, int, int);
void CountPosition(const char *text, int *pos, int text_size)
{
	const int maxK = 500;
	const int threadCount = 512;
	int blockCount = text_size / threadCount + 1;
	part1_init<<<blockCount, threadCount>>>(text, pos, text_size);
	cudaDeviceSynchronize();
    for (int i = 2; i <= maxK; ++i) {
      blockCount = (text_size / i + 1) / threadCount + 1;
      part1_kernel<<<blockCount, threadCount>>>(text, pos, text_size, i);
      cudaDeviceSynchronize();
    }
}

__global__ void part1_kernel(const char * text, int * pos, int n, int k) {
   int id = blockIdx.x * blockDim.x + threadIdx.x;
   id = id * k;
   /* should be careful here as id may int overflow */
   if (id >= n) {
     return;
   }
   if (pos[id] == 0) {
     return;
   }
   if (id > 0 && pos[id-1] == k-1) {
     pos[id] = k;
     return;
   }
   int i = pos[id];
   int j = id + (k - i - 1);
   if (j+1 < n && pos[j] == k-1 && text[j+1] != '\n') {
     pos[j+1] = k;
   }
 }

 __global__ void part1_init(const char * text, int * pos, int n) {
   int id = blockIdx.x * blockDim.x + threadIdx.x;
   if (id >= n) {
     return;
  }
   if (text[id] == '\n') {
     pos[id] = 0;
   } else {
     pos[id] = 1;
   }
 
class isOne
{
	public:
	__device__ bool operator()(int x)
	{
		return x == 1;
	}
}

int ExtractHead(const int *pos, int *head, int text_size)
{
	int *buffer;
	int nhead;
	cudaMalloc(&buffer, sizeof(int)*text_size*2); // this is enough
	thrust::device_ptr<const int> pos_d(pos);
	thrust::device_ptr<int> head_d(head), flag_d(buffer), cumsum_d(buffer+text_size);

	// TODO
	auto head_end_d = thrust::copy_if(thrust::counting_iterator<int>(0), thrust::countung_iterator<int>(text_size), pos_d, head_d, isOne());
	nhead = head_end_d - head_d;
	cudaFree(buffer);
	return nhead;
}

void Part3(char *text, int *pos, int *head, int text_size, int n_head)
{
}
