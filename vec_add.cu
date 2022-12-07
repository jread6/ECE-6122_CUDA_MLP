#include "cuda_helper.h"

__global__ void vec_add_kernel(float *C, float *A, float *B, uint A_cols, uint A_rows) 
{
    uint x = threadIdx.x + blockIdx.x * blockDim.x; // column index in A
    uint y = threadIdx.y + blockIdx.y * blockDim.y; // row index in A

    if (x < A_cols && y < A_rows)
    {
        C[y*A_cols + x] = A[y*A_cols + x] + B[x];
    }
}

float *vec_add(float* h_A, float* h_B, uint A_rows, uint A_cols)
{
    // B is a [1 x A_cols] vector
    dim3 dimsA(A_cols, A_rows, 1); 
    dim3 dimsB(A_cols, 1, 1); 
    dim3 dimsC(A_cols, A_rows, 1);

    uint A_size = dimsA.x * dimsA.y;
    uint B_size = dimsB.x * dimsB.y;

    uint A_mem_size = sizeof(float)*A_size;
    uint B_mem_size = sizeof(float)*B_size;

    // declare the arrays
    float *h_C, *d_A, *d_B, *d_C;

    // allocate host memory
    // checkCuda( cudaMallocHost((void**)&h_A, A_mem_size) );
    // checkCuda( cudaMallocHost((void**)&h_B, B_mem_size) );
    checkCuda( cudaMallocHost((void**)&h_C, A_mem_size) );

    // allocate device memory
    checkCuda( cudaMalloc((void**)&d_A, A_mem_size) );
    checkCuda( cudaMalloc((void**)&d_B, B_mem_size) );
    checkCuda( cudaMalloc((void**)&d_C, A_mem_size) );

    // copy host memory to device
    checkCuda( cudaMemcpy(d_A, h_A, A_mem_size, cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(d_B, h_B, B_mem_size, cudaMemcpyHostToDevice) );

    // define block size and number of blocks (grid size)
    uint block_size = 32;
    dim3 threads(block_size, block_size);

    // calculate number of blocks
    dim3 grid((dimsA.x / threads.x)+1, (dimsA.y / threads.y)+1);

    // create timing events
    // cudaEvent_t start, stop;
    // checkCuda( cudaEventCreate(&start) );
    // checkCuda( cudaEventCreate(&stop) );

    // start timer
    // checkCuda( cudaEventRecord(start, NULL)) ;  

    // exectute the kernels
    vec_add_kernel<<<grid, threads>>>(d_C, d_A, d_B, dimsA.x, dimsA.y);

    // stop the timer
    // checkCuda( cudaEventRecord(stop, NULL) );  
    // checkCuda( cudaEventSynchronize(stop) );

    // calculate time spent
    // float msecTotal = 0.0f;
    // checkCuda( cudaEventElapsedTime(&msecTotal, start, stop) );

    // printf("vector-matrix addition took %.3f milliseconds.\n", msecTotal);

    // copy result from device to host
    checkCuda( cudaMemcpy(h_C, d_C, A_mem_size, cudaMemcpyDeviceToHost) );    

    // write csv file
    // write_mat("vec_add.csv", h_C, dimsA.y, dimsA.x);

    checkCuda( cudaFree(d_A) );
    checkCuda( cudaFree(d_B) );
    checkCuda( cudaFree(d_C) );

    return h_C;
}