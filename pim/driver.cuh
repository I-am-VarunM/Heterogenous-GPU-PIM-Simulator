#include<cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include<thrust/copy.h>
#include "constants.h"
#include "newsimpim.cuh"
#define TILE_WIDTH 16
#define QUANT_BITS 4
namespace pim
{

    
    //extern  thrust::device_vector<int> weightmemory(Numberofcrossbar*Numberofcolumns*Numberofrows,2);
    //extern thrust::device_vector<float> d_smem(Numberofcrossbar*Numberofrows*Numberofcolumns);
    //extern thrust::device_vector<float> gpu_mem(Numberofcrossbar*Numberofrows*Numberofcolumns*4);

    void matrix_multiplication(int *h_inputmatrix, int num_cols, int num_rows,int Xstart,int Xend,char transfer_device,int out_start_ind)
    {  
        clearcrossbarmemory();
        
        cudaStream_t streams[num_rows];
        for (int i = 0; i < num_rows; ++i)
             cudaStreamCreate(&streams[i]); 

        thrust::device_vector<int> d_input(num_cols,0);
        int cols_per_data = QUANT_BITS/NUM_BITS;
        int Xnum_cols = (Xend - Xstart)*Numberofcolumns;
        int num_res = Numberofcolumns/cols_per_data;

        for(int i=0;i<num_cols;i++)
        {
            
            cudaMemcpyAsync(thrust::raw_pointer_cast(d_input.data()),h_inputmatrix+i*num_cols, d_input.size()*sizeof(int), cudaMemcpyHostToDevice, streams[i]);

	    vectormultiplication<<<256,256,Numberofcrossbar*Numberofcolumns*sizeof(float), streams[i]>>>(
             thrust::raw_pointer_cast(d_smem.data()),
             thrust::raw_pointer_cast(d_input.data()),
             thrust::raw_pointer_cast(weightmemory.data()),
             num_cols, Xnum_cols,Xstart*num_res,cols_per_data);
         

            std::vector<float> h_smem_new(Numberofcrossbar * Numberofrows * Numberofcolumns);
            cudaMemcpy(thrust::raw_pointer_cast(h_smem_new.data()), thrust::raw_pointer_cast(d_smem.data()), h_smem_new.size() * sizeof(float), cudaMemcpyDeviceToHost); 
            if(transfer_device == 'G')
                copyfromXbartoGPU(Xstart,Xend,out_start_ind+i*num_res);
          
        }
        
        for (int i = 0; i < num_rows; ++i) 
            cudaStreamSynchronize(streams[i]);
        
    
        for (int i = 0; i < num_rows; ++i) 
            cudaStreamDestroy(streams[i]);
         
   }

   void GPUmatmult(int mat1_start,int mat2_start,int res_start,int num_cols,int num_rows)
   {
       
        matrixMultiplyShared<<<dim3(1,1),dim3(16,16),2*TILE_WIDTH*TILE_WIDTH*sizeof(float)>>>(thrust::raw_pointer_cast(gpu_mem.data()),0,20,520, 4, 4);
  }
   
   void softmaxDr(int in_start_reg,int out_start_reg,int num_rows,int num_cols)
   {
        
     softmax<<<256,256>>>(thrust::raw_pointer_cast(gpu_mem.data()),in_start_reg,out_start_reg,num_rows,num_cols);
   }
   
  void scalardivDr(int in_start,int out_start,int num_rows,int num_cols,int scale)
{

   scalardiv<<<256,256>>>(thrust::raw_pointer_cast(gpu_mem.data()),in_start,out_start,num_rows*num_cols,scale);
}
}
