#include "cuda_helper.h"

__global__ void transpose_kernel(float *C, float *A, uint C_cols, uint C_rows) 
{
    uint x = threadIdx.x + blockIdx.x * blockDim.x; // column index in C
    uint y = threadIdx.y + blockIdx.y * blockDim.y; // row index in C

    if (x < C_cols && y < C_rows)
    {
        uint out_idx = y*C_cols + x;
        uint in_idx = x*C_rows + y;
        
        C[out_idx] = A[in_idx];
    }
}

float *transpose(float* h_A, uint A_rows, uint A_cols)
{
    dim3 dimsA(A_cols, A_rows, 1); 
    dim3 dimsC(A_rows, A_cols, 1);

    uint A_size = dimsA.x * dimsA.y;
    uint C_size = dimsC.x * dimsC.y;

    uint A_mem_size = sizeof(float)*A_size;
    uint C_mem_size = sizeof(float)*C_size;

    // declare the arrays
    float *h_C, *d_A, *d_C;

    // allocate host memory
    // checkCuda( cudaMallocHost((void**)&h_A, A_mem_size) );
    // checkCuda( cudaMallocHost((void**)&h_B, B_mem_size) );
    checkCuda( cudaMallocHost((void**)&h_C, C_mem_size) );

    // allocate device memory
    checkCuda( cudaMalloc((void**)&d_A, A_mem_size) );
    checkCuda( cudaMalloc((void**)&d_C, C_mem_size) );

    // copy host memory to device
    checkCuda( cudaMemcpy(d_A, h_A, A_mem_size, cudaMemcpyHostToDevice) );

    // define block size and number of blocks (grid size)
    uint block_size = 32;
    dim3 threads(block_size, block_size);

    // calculate number of blocks
    dim3 grid((dimsC.x / threads.x)+1, (dimsC.y / threads.y)+1); 

    // exectute the kernels
    transpose_kernel<<<grid, threads>>>(d_C, d_A, dimsC.x, dimsC.y);

    // copy result from device to host
    checkCuda( cudaMemcpy(h_C, d_C, C_mem_size, cudaMemcpyDeviceToHost) );    

    // write csv file
    // write_mat("transpose.csv", h_C, dimsC.y, dimsC.x);

    checkCuda( cudaFree(d_A) );
    checkCuda( cudaFree(d_C) );

    return h_C;
}