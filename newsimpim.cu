#include<cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include<thrust/copy.h>
#define NUM_STREAMS 4 // Has to be played around to find the optimal value
#define TILE_WIDTH 16
namespace pim
{
    
    const int Numberofcolumns  = 1024; // per crossbar
    const int Numberofrows = 4; //per crossbar
    const int Numberofcrossbar =2;
    const int bitspermemristor = 2;
    int leftcrossbar = Numberofcrossbar; //To check how many crossbars are not occupied
    thrust::device_vector<int> smem(Numberofcrossbar*Numberofcolumns); // This is the memory for the crossbar. The crossbar outputs are accumulated here where the shift and add takes place
    thrust::host_vector<int> h_inputmatrix(M*N, 0); // Host Input Matrix 
    thrust::device_vector<int> d_input(M,0); // Device Input
    thrust::device_vector<int> weightmemory(Numberofcrossbar*Numberofcolumns*Numberofrows,0);
    thrust::host_vector<int> weightmemhost(Numberofcrossbar*Numberofcolumns*Numberofrows,0);
    thrust::device_vector<float> d_smem(Numberofcrossbar*Numberofrows*Numberofcolumns,0);
    /*
    cols should basically be a total cap on the number of the columsn totally including all the crossbars
    xsize is the x dimension of each vector
    ysize is the number of columns of each matrix
    rows and cols should be multiples of crossbar dimensions
    Before sending the input matrix rotate the Input matrix by 90 degrees.
    */
    
    __global__ void vectormultiplication(int *smem, float *d_smem,int *vec, int  *mat, int *out, const int rows, const int cols, const int xsize, const int ysize, int row_number){
        int tid=threadIdx.x+blockIdx.x*blockDim.x;
            int sum=0;

        int sharedMemSize = (cols + 31) / 32; // Integer division with ceiling effect
        /*
        //Suppose you want to multiply v1 and v2 to M1 and M2 represent that in this format one proper vector with [v1] and Matrix to be represented as [M1 M2]
                                                                                                                   [v2]
        */                                                                                                         
        if(tid<cols && (i+(tid/ysize)*xsize) < rows){ 
            for(int i=0; i<rows; i++)
                sum += vec[i + (tid/ysize)*xsize]*mat[(i*cols)+tid];
            out[tid]= sum;
            smem[tid/32] |= out[tid]<<(30-(bitspermemristor*tid)); //Mimicking the shift and add
            __syncthreads();
        }
        if (threadIdx.x < sharedMemSize) { //Copying the multiplied values from crossbar to Main memory.
            d_smem[blockIdx.x * sharedMemSize + threadIdx.x + row_number* cols] += __int_as_float(smem[threadIdx.x]); // accordingly some converting function has to be used here __int_as_float converts to IEEE FP32
        }
    }
    void cleardevicememory(float *d_smem) //clears device memory
    {
        thrust::fill(d_smem.begin(), d_smem.end(), 0);
    }
    void clearcrossbarmemory(int *smem) //Clears the crossbar outputs needs to be done everytime new vector matrix multiplication is done
    {
        thrust::fill(smem.begin(), smem.end(), 0);
    }
    /*sending the input matrix to multiply
    if the weight matrix is already loaded one function and if the weight matrix is not loaded one function
    M and N are the dimensions of Matrix
    */
    void matrix_multiplication(int *h_inputmatrix, int M, int N, int rows, int cols,int xsize, int ysize)
    {  cudaStream_t streams[NUM_STREAMS];
        for (int i = 0; i < NUM_STREAMS; ++i) { cudaStreamCreate(&streams[i]); }

        clearcrossbarmemory(thrust::raw_pointer_cast(smem.data()));
        
        for(int i=0;i<N;i++)
        {
            const int index= i%NUM_STREAMS;
            //Copying the host input to GPU memory
            cudaMemcpyAsync(thrust::raw_pointer_cast(d_input.data()), thrust::raw_pointer_cast(h_inputmatrix.data())+i*M, d_input.size()*sizeof(mytype), cudaMemcpyHostToDevice, streams[index]);
            //launching the kernel
            vectormultiplication<<<activeCrossbars,Numberofcolumns,Numberofcrossbar*Numberofrows*Numberofcolumns*sizeof(float), streams[i]>>>(
             thrust::raw_pointer_cast(smem.data()),
             thrust::raw_pointer_cast(d_smem.data()),
             thrust::raw_pointer_cast(d_input.data()),
             thrust::raw_pointer_cast(weightmemory.data()),
            rows, cols,
            xsize, ysize,
            i
         );
         clearcrossbarmemory(thrust::raw_pointer_cast(smem.data())); // Clearing the crossbar outputs
        }
        for (int i = 0; i < NUM_STREAMS; ++i) {
            cudaStreamSynchronize(streams[i]);
        }
    
        for (int i = 0; i < NUM_STREAMS; ++i) {
            cudaStreamDestroy(streams[i]);
        } 
    }

    void loadweightstopim(int *weightmatrix, int* weightmemory, int rows, int cols, int crossbarnumber)
    {
        thrust::copy(weightmatrix, weightmatrix + (rows * cols), weightmemory.begin() + crossbarnumber * (rows*cols));
    }
    void copyfromdevice(int *d_smem, int *h_smem,int rows, int cols, int start_crossbar, int end_crossbar)
    {
        thrust::copy(d_smem.begin()+(start_crossbar*rows*cols), d_smem.begin()+(end_crossbar*rows*cols), h_smem.begin());
    }
    void clearweights(int *weightmemory, int start_crossbar, int end_crossbar, int rows, int cols)
    {
        thrust::fill(weightmemory.begin() + (start_crossbar*rows*cols), weightmemory.begin() + (end_crossbar*rows*cols), 0); 
    }
    __global__ void softmax(float *kqtranspose, float *softmaxout, int rows, int cols)
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        // Find the maximum value in the row
        float max_val = kqtranspose[row * cols + 0];
        for (int i = 1; i < cols; i++) {
            max_val = fmax(max_val, kqtranspose[row * cols + i]);
        }

        // Compute the numerator and denominator for Softmax
        float numerator = exp(kqtranspose[row * cols + col] - max_val);
        float denominator = 0.0f;
        for (int i = 0; i < cols; i++) {
            denominator += exp(kqtranspose[row * cols + i] - max_val);
        }

        // Compute the Softmax value
        softmaxout[row * cols + col] = numerator / denominator;

    } }
    void padmatrixandload(int *xbar, int *matrix, int M, int N, int Numberofcrossbar, int Numberofcolumns, int Numberofrows)
    {/*
        Explaining the function with an illustration
        suppose my matrix is 2x2 matrix but my crossbar size is 3x3
        [[1,2],[3,4]]
        This function is going to convert the matrix to [[1,2,0],[3,4,0],[0,0,0]]
        This fucntion is to maintain the uniformity of the crossbar
     */
     int x = (N+Numberofcolumns-1)/Numberofcolumns;
     int y = (M+Numberofrows-1)/Numberofrows;
     int reqxbar = x*y; //Required Number of crossbars for storing one matrix
     if(leftcrossbar > reqxbar)
     {
     for(int i=0;i<M;i++)
     {
        for(int j=0;j<N;j++)
        {
            xbar[(i)*Numberofcolumns + (j) + (Numberofcrossbar-leftcrossbar)*(Numberofcolumns*Numberofrows)] = matrix[i*N + j];
        }
     }
     //start = weightmemory.end(); //start index for loading the matrix
     //thrust::copy(xbarforparticularmatrix.begin(),xbarforparticularmatrix.end(), weightmemory.begin()+start);
     leftcrossbar -= reqxbar;
    }
}
//Matrix Multiplication (Tile Based considering the fact that Matrix sizes can be bigger than the maximum number of threads)
//GPU being used as GPU to perform matrix multiplication
__global__ void matrixMultiplyShared(float* A, float* B, float* C, int N, int M) {
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
            As[ty][tx] = A[Row * M + t * TILE_WIDTH + tx];
        else
            As[ty][tx] = 0.0;

        if (Col < N && t * TILE_WIDTH + ty < M)
            Bs[ty][tx] = B[(t * TILE_WIDTH + ty) * N + Col];
        else
            Bs[ty][tx] = 0.0;

        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; ++i) {
            Cvalue += As[ty][i] * Bs[i][tx];
        }

        __syncthreads();
    }

    if (Row < N && Col < N) {
        C[Row * N + Col] = Cvalue;
    }
}
    }