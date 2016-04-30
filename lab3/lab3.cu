#include "lab3.h"
#include <cstdio>
#include <iostream>

using namespace std;

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }


__global__ void CalculateFixed(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox	
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;//coordinate
	if (yt < ht and xt < wt and mask[curt] > 127.0f) {//to simple clone
		return;
	}
	const int dir[4][2]{{1,1}, {1,-1}, {-1,1}, {-1,-1}};//definition direction
	float temp[3];
	temp[0] = 4*target[3*curt+0];
	temp[1] = 4*target[3*curt+1];
	temp[2] = 4*target[3*curt+2];
	for(int i = 0;i < 4;i++)
	{	//declare neighbor coordinate
		const int nx = idx + dir[i][0];
		const int ny = idy + dir[i][0];
		const int curn = wt*ny + nx;
		if (nx >= 0 && ny >= 0 && nx < wt && ny < ht) {//inside boundry
       temp[0] -= target[3*curn + 0];
       temp[1] -= target[3*curn + 1];
       temp[2] -= target[3*curn + 2];
     } else {
       temp[0] -= 255.0f;
       temp[1] -= 255.0f;
       temp[2] -= 255.0f;
     }
     if ((nx < 0 || ny < 0 || nx >= wt || ny >= ht) || mask[curn] < 127.0f) {//at edge
		const int bx = nx + ox;
		const int by = ny + oy;
		const int curb = (wb*by + bx)*3;
		temp[0] += background[curb+0];
		temp[1] += background[curb+1];
		temp[2] += background[curb+2];
     }
   }
	output[3*curt+0] = temp[0];
	output[3*curt+1] = temp[1];
	output[3*curt+2] = temp[2];
}

__global__ void PoissonImageCloningIteration(
	const float *fixed,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	const int idy = blockDim.y * blockIdx.y + threadIdx.y;
	const int curt = wt * idy + idx;
	if (idx >= wt || idy >= ht || mask[curt] < 127.0f) {
		return;
	}
	float temp[3];
   temp[0] = fixed[3*curt+0];
   temp[1] = fixed[3*curt+1];
   temp[2] = fixed[3*curt+2];
   const int dir[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
   for (int i = 0; i < 4; ++i) {
     const int nx = idx + dir[i][0];
     const int ny = idy + dir[i][1];
     const int curn = wt*ny + nx;
     if (nx >= 0 && nx < wt && ny >= 0 && ny < ht && mask[curn] > 127.0f) {
       temp[0] += target[3*curn+0];
       temp[1] += target[3*curn+1];
       temp[2] += target[3*curn+2];
     }
   }
   output[3*curt+0] = temp[0] / 4;
   output[3*curt+1] = temp[1] / 4;
   output[3*curt+2] = temp[2] / 4;
}

__global__ void SimpleClone(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	if (yt < ht and xt < wt and mask[curt] > 127.0f) {
		const int yb = oy+yt, xb = ox+xt;
		const int curb = wb*yb+xb;
		if (0 <= yb and yb < hb and 0 <= xb and xb < wb) {
			output[curb*3+0] = target[curt*3+0];
			output[curb*3+1] = target[curt*3+1];
			output[curb*3+2] = target[curt*3+2];
		}
	}
}

void PoissonImageCloning(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	// set up
	float *fixed, *buf1, *buf2;
	cudaMalloc(&fixed, 3*wt*ht*sizeof(float));
	cudaMalloc(&buf1, 3*wt*ht*sizeof(float));
	cudaMalloc(&buf2, 3*wt*ht*sizeof(float));
	
	// initialize the iteration
	dim3 gdim(CeilDiv(wt,32), CeilDiv(ht,16)), bdim(32,16);
	CalculateFixed<<<gdim, bdim>>>(
	background, target, mask, fixed,
	wb, hb, wt, ht, oy, ox
	);
	cudaMemcpy(buf1, target, sizeof(float)*3*wt*ht, cudaMemcpyDeviceToDevice);
	
	// iterate
	for (int i = 0; i < 10000; ++i) {
	PoissonImageCloningIteration<<<gdim, bdim>>>(
	fixed, mask, buf1, buf2, wt, ht
	);
	PoissonImageCloningIteration<<<gdim, bdim>>>(
	fixed, mask, buf2, buf1, wt, ht
	);
	}
	
	// copy the image back
	cudaMemcpy(output, background, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);
	SimpleClone<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
		background, target, mask, output,
		wb, hb, wt, ht, oy, ox
	);
	
	// clean up
	cudaFree(fixed);
	cudaFree(buf1);
	cudaFree(buf2);
}
