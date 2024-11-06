#ifndef NEWSIMPIM_H
#define NEWSIMPIM_H
#include<cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include<thrust/copy.h>
#include "constants.h"
#define TILE_WIDTH 16
#define QUANT_BITS 4

namespace pim
{
    const int Numberofresults = (Numberofcolumns*NUM_BITS)/QUANT_BITS;
    thrust::device_vector<int> weightmemory(Numberofcrossbar*Numberofcolumns*Numberofrows);
    thrust::device_vector<float> d_smem(Numberofcrossbar*Numberofrows*Numberofcolumns);
    thrust::device_vector<float> gpu_mem(Numberofcrossbar*Numberofrows*Numberofcolumns*4);

    inline void cleardevicememory() //clears device memory
    {
        thrust::fill(gpu_mem.begin(), gpu_mem.end(), 0);
    }
    inline void clearcrossbarmemory() //Clears the crossbar outputs needs to be done everytime new vector matrix multiplication is done
    {
        thrust::fill(d_smem.begin(), d_smem.end(), 0);
    }
    inline void initialisecrossbar()
    {
       thrust::fill(weightmemory.begin(), weightmemory.end(), 0);
    }

    inline void load_values_xbar(int* values, int offset,int size)
    {
        thrust::copy(values, values + size, weightmemory.begin() + offset);
    }

    inline void load_values_GPU(int* values, int reg_start,int size)
    {
        thrust::copy(values, values + size, gpu_mem.begin() + reg_start);
    }
    inline void copyfromXbartoGPU(int Xstart,int Xend,int GPUstart)
    {
	thrust::copy(d_smem.begin() + Xstart*Numberofresults, d_smem.begin() + Xend*Numberofresults, gpu_mem.begin()+GPUstart);
    }
 
    __global__ void vectormultiplication(float *d_smem,int *vec, int *mat, const int vlen,int mnum_cols,int offset,int cols_per_data)
       {
         int tid=threadIdx.x+blockIdx.x*blockDim.x;
         int thread = threadIdx.x;
        __shared__ int out[Numberofcolumns];

        if(tid < mnum_cols)
         {
            int sum = 0;
	    out[thread] = 0;
            for(int i=0; i<vlen; i++)
                 sum += vec[i]*mat[(i*Numberofcolumns)+tid];
	    
            int product_idx = offset+tid/cols_per_data;	        
            int bitsToShift = (tid%cols_per_data)*NUM_BITS;
            out[thread] = sum<<bitsToShift;
            __syncthreads();

            if(bitsToShift == 0)
	    {
	      int shift_added_sum = 0;
	      for(int k = 0; k < cols_per_data && k < mnum_cols-tid; k++)
                  shift_added_sum+= out[thread+k];
              d_smem[product_idx] = shift_added_sum;	
                  
            }
	    
            __syncthreads();
            if(thread == 0 && bitsToShift!=0)
            {
                int shift_added_sum = 0;
                for(int k = 0; (cols_per_data)%(tid+k) != 0; k++)
	             shift_added_sum+=out[thread+k];
                
                d_smem[product_idx] = shift_added_sum;
                  
	    }
		
	}
      }

       
    __global__ void softmax(float *gpu_mem, int in_start,int outstart,int num_rows,int num_cols)
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_rows && col < num_cols) {
        // Find the maximum value in the row
        float max_val = gpu_mem[row * num_cols + in_start];
        for (int i = 1; i < num_cols; i++) {
            max_val = fmax(max_val, gpu_mem[row * num_cols + in_start + i]);
        }

        // Compute the numerator and denominator for Softmax
        float numerator = exp(gpu_mem[row * num_cols + col + in_start] - max_val);
        float denominator = 0.0f;
        for (int i = 0; i < num_cols; i++) {
            denominator += exp(gpu_mem[row * num_cols + in_start + i] - max_val);
        }

        // Compute the Softmax value
        gpu_mem[outstart + row * num_cols + col] = numerator / denominator;

    } }

   __global__ void scalardiv(float* gpu_mem,int in_start,int outstart,int total_size,int scale)
   {
       int tid = threadIdx.x;
       if(tid < total_size)
          gpu_mem[outstart + tid] = gpu_mem[in_start + tid]/scale;
   }

  __global__ void matrixMultiplyShared(float* gpu_mem,int ASt,int BSt,int CSt, int N, int M) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Cvalue = 0.0;

    for (int t = 0; t < (M + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        if (Row < N && t * TILE_WIDTH + tx < M)
            As[ty][tx] = gpu_mem[ASt + Row * M + t * TILE_WIDTH + tx];
        else
            As[ty][tx] = 0.0;

        if (Col < N && t * TILE_WIDTH + ty < M)
            Bs[ty][tx] = gpu_mem[BSt + (t * TILE_WIDTH + ty) * N + Col];
        else
            Bs[ty][tx] = 0.0;

        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; ++i) {
            Cvalue += As[ty][i] * Bs[i][tx];
        }

        __syncthreads();
    }

    if (Row < N && Col < N) {
        gpu_mem[CSt + Row * N + Col] = Cvalue;
    }
}
       
}
#endif
