#include "driver.cuh"
//using namespace pim;
#include<iostream>
#include<vector>
#include<string>

namespace pim
{
void printNewRegMemory(int start_idx, int end_idx)
    {
        thrust::host_vector<float> h_smem(Numberofcrossbar*Numberofrows*Numberofcolumns);
        h_smem = gpu_mem;
        for(int i = start_idx; i < end_idx;i++)
           printf("%0.2f ",h_smem[i]);
        printf("\n------------\n");
    }

void printDevMemory(int start_idx, int end_idx)
    {
        thrust::host_vector<int> h_smem(Numberofcrossbar*Numberofrows*Numberofcolumns);
        h_smem = d_smem;
        for(int i = start_idx; i < end_idx;i++)
           printf("%d ",h_smem[i]);
        printf("\n------------\n");
    }
}
int main(int argc, char **argv)
{
	
	int M = 2,N = 2,total_size = 600;
        int* host = (int*)malloc(sizeof(int)*N*M);
        for(int i = 0;i < M*N;i++)
            host[i] = 1;
        int Xstart = 0, Xend = 1;

        int* weightmemory = (int*)malloc(sizeof(int)*total_size);
        for(int i = 0;i < total_size;i++)
            weightmemory[i] = 2;

        int* h_vector = (int*)malloc(sizeof(int)*M*N);
       // pim::loadweightstopim(weightmemory,0,total_size);
        pim::matrix_multiplication(host,M,N,Xstart,Xend,'G',0);
//        pim::copyfromXbartoGPU(0,1,0);
  //      pim::printNewRegMemory(0,256);
  //      printf("\n---------\n");
/*        for(int i = 0;i < M*N;i++)
            host[i] = 2;
        pim::matrix_multiplication(host,M,N,Xstart,Xend,'G',256);
        //pim::copyfromXbartoGPU(0,1,);
        pim::GPUmatmult(0,256,512,2,128);
        pim::printDevMemory(0,256);
        pim::scalardivDr(520,540,4,4,4);
        //pim::softmaxDr(530,550,4,4);
        pim::printNewRegMemory(520,570);
        //pim::GPUtoHost(40,50,h_vector);*/
        return 0;
}

