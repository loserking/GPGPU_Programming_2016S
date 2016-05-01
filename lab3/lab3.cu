#include "lab3.h"
#include <cstdio>
#include <Timer.h>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }


__global__ void CalculateFixed(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, 
	const int wt, const int ht,
	const int oy, const int ox	
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;

	//target array index for the current pixel location
	const int curt = wt*yt + xt;//coordinate
	
	if(0 <= xt && xt < wt && 0 <= yt && yt < ht && mask[curt] > 127.0f)
	{
		float Nb, Wb, Sb, Eb, Nt, Wt, St, Et;
		const int yb = oy+yt, xb = ox+xt;
		// background array index for the current pixel location
		const int curb = wb*yb+xb;
		if (0 <= yb && yb < hb && 0 <= xb && xb < wb) 
		{
			for(int i = 0; i < 3; i++) 
			{
				Nb = (yb > 0) ? ((yt == 0 || mask[curt-wt] < 127.0f) ? background[(curb-wb)*3+i]:0) :background[curb*3+i];
				Wb = (xb > 0) ? ((xt == 0 || mask[curt-1] < 127.0f) ? background[(curb-1)*3+i]:0) :background[curb*3+i];
				Sb = (yb < hb-1) ? ((yt == ht-1 || mask[curt+wt] < 127.0f) ? background[(curb+wb)*3+i]:0) :background[curb*3+i];
				Eb = (xb < wb-1) ? ((xt == wt-1 || mask[curt+1] < 127.0f) ? background[(curb+1)*3+i]:0) :background[curb*3+i];
				Nt = (yt > 0) ? target[(curt-wt)*3+i] :target[curt*3+i];
				Wt = (xt > 0) ? target[(curt-1)*3+i] :target[curt*3+i];
				St = (yt < ht-1) ? target[(curt+wt)*3+i] :target[curt*3+i];
				Et = (xt < wt-1) ? target[(curt+1)*3+i] :target[curt*3+i];
		
				output[curt*3+i] = 4*target[curt*3+i] - (Nt + Wt + St + Et) + (Nb + Wb + Sb+ Eb);
			}
		}
	}
	else
	{
		output[curt*3] = 0.0f;
		output[curt*3+1] = 0.0f;
		output[curt*3+2] = 0.0f;
	}	
}

__global__ void PoissonImageCloningIteration(
	const float *fixed,
	const float *mask,
	const float *target,
	float *output,
	const int wt, 
	const int ht
)
{
	const int xt = blockDim.x * blockIdx.x + threadIdx.x;
	const int yt = blockDim.y * blockIdx.y + threadIdx.y;
	const int curt = wt*yt + xt;
	float Nb, Wb, Sb, Eb;
	
	if(0 <= xt && xt < wt && 0 <= yt && yt < ht && mask[curt] > 127.0f)//inside boundry
	{
		for(int i = 0; i < 3; i++) 
		{
			Nb = (yt == 0 || mask[curt-wt] < 127.0f) ?0 :target[(curt-wt)*3+i];
			Wb = (xt == 0 || mask[curt-1] < 127.0f) ?0 :target[(curt-1)*3+i];
			Sb = (yt == ht-1 || mask[curt+wt] < 127.0f) ?0 :target[(curt+wt)*3+i];
			Eb = (xt == wt-1 || mask[curt+1] < 127.0f) ?0 :target[(curt+1)*3+i];
			output[curt*3+i] = (fixed[curt*3+i] + Nb + Wb + Sb + Eb)/4;
		}
	}
	else
	{
		output[curt*3] = target[curt*3];
		output[curt*3+1] = target[curt*3+1];
		output[curt*3+2] = target[curt*3+2];
	}
}

//Implement  successive over-relaxation acceleration
__global__ void sor(
	float w,
	float *cur,
	float *nxt,
	const int wt,
	const int ht 
)
{
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int curt = wt*yt + xt;
	
	nxt[curt*3] = nxt[curt*3] + (1-w)*cur[curt*3];
	nxt[curt*3+1] = nxt[curt*3+1] + (1-w)*cur[curt*3+1];
	nxt[curt*3+2] = nxt[curt*3+2] + (1-w)*cur[curt*3+1];
}

__global__ void SimpleClone(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, 
	const int wt, const int ht,
	const int oy, const int ox
)
{
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int curt = wt*yt + xt;

	if (yt < ht and xt < wt and mask[curt] > 127.0f) {
		const int yb = oy + yt, xb = ox + xt;
		const int curb = wb*yb + xb;
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

	//timer start	
	cudaDeviceSynchronize();
	Timer timer;
	timer.Start();

	// initialize the iteration
	//Gridsize: (wt/32 + wt%32) * (ht/16 + ht%16) blocks; Blocksize: 32*16 threads
	//dim3 is a 3D structure, flatten to 2D for simpler usage here
	dim3 gdim(CeilDiv(wt,32), CeilDiv(ht,16)), bdim(32,16);
	CalculateFixed<<<gdim, bdim>>>(
	background, target, mask, fixed,
	wb, hb, wt, ht, oy, ox
	);
	cudaDeviceSynchronize();
	cudaMemcpy(buf1, target, sizeof(float)*3*wt*ht, cudaMemcpyDeviceToDevice);
	for (int i = 0; i < 10; i++)
	{
		// use buf1 as reference and write to buf2
		PoissonImageCloningIteration<<<gdim, bdim>>>(
			fixed, mask, buf1, buf2, wt, ht
		);
		cudaDeviceSynchronize();
		//run SOR
		sor<<<gdim, bdim>>>(w, buf1, buf2, wt, ht);
		cudaDeviceSynchronize();
		
		// use buf2 as reference and write back to buf1
		PoissonImageCloningIteration<<<gdim, bdim>>>(
			fixed, mask, buf2, buf1, wt, ht
		);
		cudaDeviceSynchronize();
		
		//run SOR
		sor<<<gdim, bdim>>>(w, buf2, buf1, wt, ht);
		cudaDeviceSynchronize();
	}
	
	// iterate
	for (int i = 0; i < 5000; ++i) {
	//use buf1 as reference and write to buf2
	PoissonImageCloningIteration<<<gdim, bdim>>>(
	fixed, mask, buf1, buf2, wt, ht
	);
	cudaDeviceSynchronize();

	//use buf2 as reference and write to buf1
	PoissonImageCloningIteration<<<gdim, bdim>>>(
	fixed, mask, buf2, buf1, wt, ht
	);
	cudaDeviceSynchronize();
	}
	
	// copy the image back
	cudaMemcpy(output, background, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);
	SimpleClone<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
		background, target, mask, output,
		wb, hb, wt, ht, oy, ox
	);	
	cudaDeviceSynchronize();
	
	//print timer
	timer.Pause();
	printf_timer(timer);
	
	// clean up
	cudaFree(fixed);
	cudaFree(buf1);
	cudaFree(buf2);
}
