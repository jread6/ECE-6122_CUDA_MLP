#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <driver_types.h>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <cstdlib>
#include <math.h>
#include <memory>

#ifndef CUDA_HELPER
#define CUDA_HELPER

#define checkCuda(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int gpuDeviceInit(int devID);


struct network
{
    // number of training images
    int train_size = 60000;

    // number of test images
    int test_size = 10000;

    // learning rate for the network
    double learning_rate = 0.1;

    bool load_weights = false;

    // number of epochs to train for
    int num_epochs = 1;

    // dimensions of layers
    int *dims = NULL;

    // batch size
    int batch_size = 0;

    // layer 1 weights and biases
    float *l1 = NULL;
    float *b1 = NULL;

    // layer 2 weights and biases
    float *l2 = NULL;
    float *b2 = NULL;
};

struct out_cache
{
    float *u1 = NULL;
    float *u2 = NULL;

    float *o1 = NULL;
    float *o2 = NULL;
};

void write_mat(std::string filename, float *H, uint rows, uint cols);
void write_img(std::string filename, uint *H, uint rows, uint cols);
void read_mat(std::string filename, float *mat, uint rows, uint cols); 
void save_weights(network *net);

// mnist helper functions
int reverseInt (int i);
void load_mnist_labels(uint *labels_train, std::string filename);
void load_mnist_images(int *inputs_train, std::string filename);
void load_mnist(int *images, uint *labels, uint data_size, uint img_size, std::string filename);

// neural network helper functions
void init_nn(network *net, out_cache *cache, int num_HU, int batch_size, int num_epochs, bool load_weights);
void randomize(float *array, int array_size);
bool check_inputs(int argc, char *argv[]);
int select_batch(float *inputs, float *input_batch, uint *labels, uint *label_batch, uint batch_size, uint img_size, int img_idx, int label_idx, int data_size);
void append_loss(double loss);
double calc_loss(network *net, out_cache *cache, uint *labels);
int calc_num_correct(out_cache *cache, uint *labels, int batch_size, int num_classes);
float *normalize(int *inputs, uint input_size, float max_val);
void clip_gradients_cpu(float *mat, uint size);

// element-wise multiplication
float *el_wise_mat_mul(float* h_A, float* h_B, uint A_rows, uint A_cols);
__global__ void el_wise_mat_mul_kernel(float *C, float *A, float *B, uint cols, uint rows);

// derivate of the cross-entropy loss function
float *dLoss(float *h_A, uint *labels, uint A_rows, uint A_cols);
__global__ void dLoss_kernel(float *A, uint *labels, uint A_cols, uint A_rows); 

// matrix multiplication
float *mat_mul(float* h_A, float* h_B, uint A_rows, uint A_cols, uint B_rows, uint B_cols);
__global__ void mat_mul_kernel(float *C, float *A, float *B, uint C_rows, uint A_cols, uint B_cols); 

// vector-matrix addition
float *vec_add(float* h_A, float* h_B, uint A_rows, uint A_cols);
__global__ void vec_add_kernel(float *C, float *A, float *B, uint A_cols, uint A_rows); 

// derivative of ReLU
float *dReLU(float *h_A, uint A_rows, uint A_cols);
__global__ void dReLU_kernel(float *A, uint A_cols, uint A_rows); 

// ReLU
float *ReLU(float* h_A, uint A_rows, uint A_cols);
__global__ void ReLU_kernel(float *A, uint A_cols, uint A_rows);

// transpose
float *transpose(float* h_A, uint A_rows, uint A_cols);
__global__ void transpose_kernel(float *C, float *A, uint C_cols, uint C_rows);

// row sum
float *row_sum(float *h_A, uint A_rows, uint A_cols);
__global__ void row_sum_kernel(float *A_out, float *A, uint A_cols, uint A_rows);

// scalar multiply
float *scalar_multiply(float* h_A, double *scalar, uint A_rows, uint A_cols);
__global__ void scalar_multiply_kernel(float *A, double *scalar, uint cols, uint rows);

// matrix subtraction
float *mat_sub(float* h_A, float *h_B, uint A_rows, uint A_cols);
__global__ void mat_sub_kernel(float *A, float *B, uint cols, uint rows);

// softmax
float *softmax(float* h_A, uint A_rows, uint A_cols);
__global__ void softmax_kernel(float *A, uint A_cols, uint A_rows);

// derivative of softmax (dLoss/du2)
float *dsoftmax(float* h_A, uint *labels, uint A_rows, uint A_cols);
__global__ void dsoftmax_kernel(float *A, uint *labels, uint A_cols, uint A_rows);

__global__ void clip_gradients_kernel(float *A, uint A_cols, uint A_rows); 
float *clip_gradients(float* h_A, uint A_rows, uint A_cols);

#endif