#include "cuda_helper.h"

__global__ void el_wise_mat_mul_kernel(float *C, float *A, float *B, uint cols, uint rows) 
{
    uint x = threadIdx.x + blockIdx.x * blockDim.x; // column index in C
    uint y = threadIdx.y + blockIdx.y * blockDim.y; // row index in C

    uint mat_idx = y*cols + x;

    if (x < cols && y < rows)
    {
        C[mat_idx] = A[mat_idx]*B[mat_idx];
    }
}

float *el_wise_mat_mul(float* h_A, float* h_B, uint A_rows, uint A_cols)
{
    dim3 dimsA(A_cols, A_rows, 1); 
    dim3 dimsB(A_cols, A_rows, 1); 
    dim3 dimsC(A_cols, A_rows, 1);

    uint A_size = dimsA.x * dimsA.y;
    uint B_size = dimsB.x * dimsB.y;
    uint C_size = dimsC.x * dimsC.y;

    uint A_mem_size = sizeof(float)*A_size;
    uint B_mem_size = sizeof(float)*B_size;
    uint C_mem_size = sizeof(float)*C_size;

    // declare the arrays
    float *h_C, *d_A, *d_B, *d_C;

    // allocate host memory
    // checkCuda( cudaMallocHost((void**)&h_A, A_mem_size) );
    // checkCuda( cudaMallocHost((void**)&h_B, B_mem_size) );
    checkCuda( cudaMallocHost((void**)&h_C, C_mem_size) );

    // allocate device memory
    checkCuda( cudaMalloc((void**)&d_A, A_mem_size) );
    checkCuda( cudaMalloc((void**)&d_B, B_mem_size) );
    checkCuda( cudaMalloc((void**)&d_C, C_mem_size) );

    // copy host memory to device
    checkCuda( cudaMemcpy(d_A, h_A, A_mem_size, cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(d_B, h_B, B_mem_size, cudaMemcpyHostToDevice) );

    // define block size and number of blocks (grid size)
    uint block_size = 32;
    dim3 threads(block_size, block_size);

    // calculate number of blocks
    dim3 grid((dimsC.x / threads.x)+1, (dimsC.y / threads.y)+1);

    // exectute the kernels
    el_wise_mat_mul_kernel<<<grid, threads>>>(d_C, d_A, d_B, dimsA.x, dimsA.y);

    // copy result from device to host
    checkCuda( cudaMemcpy(h_C, d_C, C_mem_size, cudaMemcpyDeviceToHost) );    

    checkCuda( cudaFree(d_A) );
    checkCuda( cudaFree(d_B) );
    checkCuda( cudaFree(d_C) );

    return h_C;
}