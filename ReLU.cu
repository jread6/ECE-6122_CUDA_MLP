#include "cuda_helper.h"

__global__ void ReLU_kernel(float *A, uint A_cols, uint A_rows) 
{
    uint x = threadIdx.x + blockIdx.x * blockDim.x; // column index in A
    uint y = threadIdx.y + blockIdx.y * blockDim.y; // row index in A

    if (x < A_cols && y < A_rows)
    {
        if (A[y*A_cols + x] < 0)
        {
            A[y*A_cols + x] = 0;
        }
    }
}

float *ReLU(float* h_A, uint A_rows, uint A_cols)
{
    // B is a [1 x A_cols] vector
    dim3 dimsA(A_cols, A_rows, 1); 
    dim3 dimsA_out(A_cols, A_rows, 1); 

    uint A_size = dimsA.x * dimsA.y;
    uint A_out_size = dimsA_out.x * dimsA_out.y;

    uint A_mem_size = sizeof(float)*A_size;
    uint A_out_mem_size = sizeof(float)*A_out_size;

    // declare the arrays
    float *d_A, *h_A_out;
    
    // allocate host memory
    checkCuda( cudaMallocHost((void**)&h_A_out, A_out_mem_size) );

    // allocate device memory
    checkCuda( cudaMalloc((void**)&d_A, A_mem_size) );

    // copy host memory to device
    checkCuda( cudaMemcpy(d_A, h_A, A_mem_size, cudaMemcpyHostToDevice) );

    // define block size and number of blocks (grid size)
    uint block_size = 32;
    dim3 threads(block_size, block_size);

    // calculate number of blocks
    dim3 grid((dimsA_out.x / threads.x)+1, (dimsA_out.y / threads.y)+1);

    // exectute the kernels
    ReLU_kernel<<<grid, threads>>>(d_A, dimsA.x, dimsA.y);

    // copy result from device to host
    checkCuda( cudaMemcpy(h_A_out, d_A, A_mem_size, cudaMemcpyDeviceToHost) );    

    checkCuda( cudaFree(d_A) );

    return h_A_out;
}