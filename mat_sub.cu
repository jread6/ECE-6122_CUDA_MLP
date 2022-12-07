#include "cuda_helper.h"

__global__ void mat_sub_kernel(float *A, float *B, uint cols, uint rows) 
{
    uint x = threadIdx.x + blockIdx.x * blockDim.x; // column index in C
    uint y = threadIdx.y + blockIdx.y * blockDim.y; // row index in C

    uint mat_idx = y*cols + x;

    if (x < cols && y < rows)
    {
        A[mat_idx] = A[mat_idx] - B[mat_idx];
    }
}

float *mat_sub(float* h_A, float *h_B, uint A_rows, uint A_cols)
{
    dim3 dimsA(A_cols, A_rows, 1); 
    dim3 dimsB(A_cols, A_rows, 1); 

    uint A_size = dimsA.x * dimsA.y;
    uint B_size = dimsB.x * dimsB.y;

    uint A_mem_size = sizeof(float)*A_size;
    uint B_mem_size = sizeof(float)*B_size;

    // declare the arrays
    float *d_A, *d_B;

    // allocate host memory
    // checkCuda( cudaMallocHost((void**)&h_A, A_mem_size) );
    // checkCuda( cudaMallocHost((void**)&h_B, B_mem_size) );
    // checkCuda( cudaMallocHost((void**)&h_C, C_mem_size) );

    // allocate device memory
    checkCuda( cudaMalloc((void**)&d_A, A_mem_size) );
    checkCuda( cudaMalloc((void**)&d_B, B_mem_size) );

    // copy host memory to device
    checkCuda( cudaMemcpy(d_A, h_A, A_mem_size, cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(d_B, h_B, B_mem_size, cudaMemcpyHostToDevice) );

    // define block size and number of blocks (grid size)
    uint block_size = 32;
    dim3 threads(block_size, block_size);

    // calculate number of blocks
    dim3 grid((dimsA.x / threads.x)+1, (dimsA.y / threads.y)+1);

    // create timing events
    cudaEvent_t start, stop;
    checkCuda( cudaEventCreate(&start) );
    checkCuda( cudaEventCreate(&stop) );

    // start timer
    checkCuda( cudaEventRecord(start, NULL)) ;  

    // exectute the kernels
    mat_sub_kernel<<<grid, threads>>>(d_A, d_B, dimsA.x, dimsA.y);

    // copy result from device to host
    checkCuda( cudaMemcpy(h_A, d_A, A_mem_size, cudaMemcpyDeviceToHost) );    

    checkCuda( cudaFree(d_A) );
    checkCuda( cudaFree(d_B) );

    return h_A;

}