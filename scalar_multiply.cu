#include "cuda_helper.h"

__global__ void scalar_multiply_kernel(float *A, double *scalar, uint cols, uint rows) 
{
    uint x = threadIdx.x + blockIdx.x * blockDim.x; // column index in C
    uint y = threadIdx.y + blockIdx.y * blockDim.y; // row index in C

    if (x < cols && y < rows)
    {
        uint mat_idx = y*cols + x;
        A[mat_idx] = A[mat_idx]*(*scalar);
    }
}

float *scalar_multiply(float* h_A, double *scalar, uint A_rows, uint A_cols)
{
    dim3 dimsA(A_cols, A_rows, 1); 

    uint A_size = dimsA.x * dimsA.y;

    uint A_mem_size = sizeof(float)*A_size;
    uint scalar_mem_size = sizeof(double);

    // declare the arrays
    float *d_A;

    double *d_scalar;

    // allocate host memory
    // checkCuda( cudaMallocHost((void**)&h_A, A_mem_size) );
    // checkCuda( cudaMallocHost((void**)&h_B, B_mem_size) );
    // checkCuda( cudaMallocHost((void**)&h_C, C_mem_size) );

    // allocate device memory
    checkCuda( cudaMalloc((void**)&d_A, A_mem_size) );
    checkCuda( cudaMalloc((void**)&d_scalar, scalar_mem_size) );

    // copy host memory to device
    checkCuda( cudaMemcpy(d_A, h_A, A_mem_size, cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(d_scalar, scalar, scalar_mem_size, cudaMemcpyHostToDevice) );

    // define block size and number of blocks (grid size)
    uint block_size = 32;
    dim3 threads(block_size, block_size);

    // calculate number of blocks
    dim3 grid((dimsA.x / threads.x)+1, (dimsA.y / threads.y)+1);  

    // exectute the kernels
    scalar_multiply_kernel<<<grid, threads>>>(d_A, d_scalar, dimsA.x, dimsA.y);

    // copy result from device to host
    checkCuda( cudaMemcpy(h_A, d_A, A_mem_size, cudaMemcpyDeviceToHost) );    

    checkCuda( cudaFree(d_A) );
    checkCuda( cudaFree(d_scalar) );

    return h_A;

}