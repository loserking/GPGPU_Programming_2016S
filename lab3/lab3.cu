#include "lab3.h"
#include <cstdio>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__global__ void SimpleClone(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox)
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

__global__ void CalculateFixed(
	const float *background, 
	const float *target, 
	const float *mask, 
	float *fixed, 
	const int wb, const int hb, 
	const int wt, const int ht, 
	const int oy, const int ox)
{
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	
	// target array index for the current pixel locationh
	const int curt = wt*yt+xt;

	if(0 <= xt and xt < wt and 0 <= yt and yt < ht and mask[curt] > 127.0f)
	{
		float Nb, Wb, Sb, Eb, Nt, Wt, St, Et;
		const int yb = oy+yt, xb = ox+xt;
		// background array index for the current pixel locationh
		const int curb = wb*yb+xb;
 		// boundary	condition: (1) boundary of mask (2) boundary of background 	
		if (0 <= yb and yb < hb and 0 <= xb and xb < wb) 
		{
			for(int i = 0; i < 3; i++) 
			{
				Nb = (yb > 0) ?((yt == 0 || mask[curt-wt] < 127.0f) ?background[(curb-wb)*3+i] :0) :background[curb*3+i];
				Wb = (xb > 0) ?((xt == 0 || mask[curt-1] < 127.0f) ?background[(curb-1)*3+i] :0) :background[curb*3+i];
				Sb = (yb < hb-1) ?((yt == ht-1 || mask[curt+wt] < 127.0f) ?background[(curb+wb)*3+i] :0) :background[curb*3+i];
				Eb = (xb < wb-1) ?((xt == wt-1 || mask[curt+1] < 127.0f) ?background[(curb+1)*3+i] :0) :background[curb*3+i];
				Nt = (yt > 0) ?target[(curt-wt)*3+i] :target[curt*3+i];
				Wt = (xt > 0) ?target[(curt-1)*3+i] :target[curt*3+i];
				St = (yt < ht-1) ?target[(curt+wt)*3+i] :target[curt*3+i];
				Et = (xt < wt-1) ?target[(curt+1)*3+i] :target[curt*3+i];
		
				fixed[curt*3+i] = 4*target[curt*3+i] - (Nt + Wt + St + Et) + (Nb + Wb + Sb+ Eb);
			}
		}
	}
	else
			fixed[curt*3] = fixed[curt*3+1] = fixed[curt*3+2] = 0.0f;
}

__global__ void PoissonImageCloningIteration(float *fixed, const float *mask, float *ref, float *output, const int wt, const int ht)
{
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int curt = wt*yt+xt;
	float Nb, Wb, Sb, Eb;
	
	if(0 <= xt and xt < wt and 0 <= yt and yt < ht and mask[curt] > 127.0f)
	{
		// consider the mask and boundary conditions
		for(int i = 0; i < 3; i++) 
		{
			Nb = (yt == 0 || mask[curt-wt] < 127.0f) ?0 :ref[(curt-wt)*3+i];
			Wb = (xt == 0 || mask[curt-1] < 127.0f) ?0 :ref[(curt-1)*3+i];
			Sb = (yt == ht-1 || mask[curt+wt] < 127.0f) ?0 :ref[(curt+wt)*3+i];
			Eb = (xt == wt-1 || mask[curt+1] < 127.0f) ?0 :ref[(curt+1)*3+i];
			output[curt*3+i] = (fixed[curt*3+i] + Nb + Wb + Sb + Eb)/4;
		}
	}
	else
	{
		output[curt*3] = ref[curt*3];
		output[curt*3+1] = ref[curt*3+1];
		output[curt*3+2] = ref[curt*3+2];
	}
}

__global__ void sor_add(float w, float *base, float *next, const int wt, const int ht)
{
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int curt = wt*yt+xt;

	next[curt*3] = next[curt*3] * w + (1-w) * base[curt*3];
	next[curt*3+1] = next[curt*3+1] * w + (1-w) * base[curt*3+1];
	next[curt*3+2] = next[curt*3+2] * w + (1-w) * base[curt*3+2];
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

/* No addtional editing: paste the target to background, through mask.
	cudaMemcpy(output, background, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);
	SimpleClone<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
		background, target, mask, output,
		wb, hb, wt, ht, oy, ox
	);
*/
 
	// set up
	float *fixed, *buf1, *buf2;
	// store R,G,B pixel value in adjacent array indexes
	cudaMalloc((void **)&fixed, 3*wt*ht*sizeof(float));
	cudaMalloc((void **)&buf1, 3*wt*ht*sizeof(float));
	cudaMalloc((void **)&buf2, 3*wt*ht*sizeof(float));

	// initialize the iteration
	// Gridsize: (wt/32 + wt%32) * (ht/16 + ht%16) blocks; Blocksize: 32*16 threads
	// dim3 is a 3D structure, flatten to 2D for simpler usage here
	dim3 gdim(CeilDiv(wt, 32), CeilDiv(ht, 16)), bdim(32, 16);
	CalculateFixed<<<gdim, bdim>>>(
		background, target, mask, fixed,
		wb, hb, wt, ht, oy, ox
	);
	cudaDeviceSynchronize();
	cudaMemcpy(buf1, target, sizeof(float)*3*wt*ht, cudaMemcpyDeviceToDevice);


	// Implementing successive over-relaxation method	
	float w = 1.5;
	for (int i = 0; i < 10; i++)
	{
		// use buf1 as reference and write to buf2
		PoissonImageCloningIteration<<<gdim, bdim>>>(
			fixed, mask, buf1, buf2, wt, ht
		);
		cudaDeviceSynchronize();
		sor_add<<<gdim, bdim>>>(w, buf1, buf2, wt, ht);
		cudaDeviceSynchronize();
		// use buf2 as reference and write back to buf1
		PoissonImageCloningIteration<<<gdim, bdim>>>(
			fixed, mask, buf2, buf1, wt, ht
		);
		cudaDeviceSynchronize();
		sor_add<<<gdim, bdim>>>(w, buf2, buf1, wt, ht);
		cudaDeviceSynchronize();
	}
	// iterate
	for (int i = 0; i < 5000; ++i) {
		// use buf1 as reference and write to buf2
		PoissonImageCloningIteration<<<gdim, bdim>>>(
			fixed, mask, buf1, buf2, wt, ht
		);
		cudaDeviceSynchronize();
		// use buf2 as reference and write back to buf1
		PoissonImageCloningIteration<<<gdim, bdim>>>(
			fixed, mask, buf2, buf1, wt, ht
		);
		cudaDeviceSynchronize();
	}

	// copy the image back
	cudaMemcpy(output, background, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);
	SimpleClone<<<gdim, bdim>>>(
		background, buf1, mask, output,
		wb, hb, wt, ht, oy, ox
	);
	cudaDeviceSynchronize();
	// clean up
	cudaFree(fixed);
	cudaFree(buf1);
	cudaFree(buf2);
}
