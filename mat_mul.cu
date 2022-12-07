#include "cuda_helper.h"

__global__ void mat_mul_kernel(float *C, float *A, float *B, uint C_rows, uint A_cols, uint B_cols) 
{
    uint x = threadIdx.x + blockIdx.x * blockDim.x; // column index in C
    uint y = threadIdx.y + blockIdx.y * blockDim.y; // row index in C

    if (x < B_cols && y < C_rows)
    {
        float Psum = 0;
        for (uint i = 0; i < A_cols; i++)
        {
            Psum += A[y*A_cols + i] * B[i*B_cols + x];
        }

        C[y*B_cols + x] = Psum;
    }
}

float *mat_mul(float* h_A, float* h_B, uint A_rows, uint A_cols, uint B_rows, uint B_cols)
{
    dim3 dimsA(A_cols, A_rows, 1); 
    dim3 dimsB(B_cols, B_rows, 1); 
    dim3 dimsC(B_cols, A_rows, 1);

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
    mat_mul_kernel<<<grid, threads>>>(d_C, d_A, d_B, dimsC.y, dimsA.x, dimsB.x);

    // copy result from device to host
    checkCuda( cudaMemcpy(h_C, d_C, C_mem_size, cudaMemcpyDeviceToHost) );    

    checkCuda( cudaFree(d_A) );
    checkCuda( cudaFree(d_B) );
    checkCuda( cudaFree(d_C) );

    return h_C;
}