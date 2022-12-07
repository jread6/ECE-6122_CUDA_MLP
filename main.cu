#include "cuda_helper.h"

// 2-layer neural network with user defined number of hidden units
// trained on the MNIST dataset to classify handwritten digits

void forward(network* net, out_cache *cache, float *inputs)
{
    // calculate dot product between each input and the first layer
    cache->u1 = mat_mul(inputs, net->l1, net->batch_size, net->dims[0], net->dims[0], net->dims[1]);
    
    // write csv file
    // std::cout << "writing u1...\n";
    // write_mat("temp/u1.csv", cache->u1, net->batch_size, net->dims[1]);

    // add the biases in place 
    cache->u1 = vec_add(cache->u1, net->b1, net->batch_size, net->dims[1]);
    // std::cout << "writing u1...\n";
    // write_mat("temp/u1_2.csv", cache->u1, net->batch_size, net->dims[1]);

    // make first 2 rows negative
    // for (int i = 0; i < net->dims[1]; i++) {
    //     cache->u1[i] = -cache->u1[i];
    //     cache->u1[300+i] = -cache->u1[300+i];
    // }

    // ReLU
    cache->o1 = ReLU(cache->u1, net->batch_size, net->dims[1]);
    // std::cout << "writing o1...\n";
    // write_mat("temp/o1.csv", cache->o1, net->batch_size, net->dims[1]);

    // return;

    // calculate dot product between each input and the second layer    
    cache->u2 = mat_mul(cache->o1, net->l2, net->batch_size, net->dims[1], net->dims[1], net->dims[2]);

    // write_mat("temp/u2.csv", cache->u2, net->batch_size, net->dims[2]);

    // add the biases in place 
    cache->u2 = vec_add(cache->u2, net->b2, net->batch_size, net->dims[2]);

    // write_mat("temp/u2_2.csv", cache->u2, net->batch_size, net->dims[2]);

    // ReLU
    cache->o2 = softmax(cache->u2, net->batch_size, net->dims[2]);

    // write_mat("temp/o2.csv", cache->o2, net->batch_size, net->dims[2]);

}

void backward(network *net, out_cache *cache, float *inputs, uint *labels)
{

    // calculate dLoss/do2 matrix (categorical cross-entropy loss)
    // float *dLoss_do2 = dLoss(cache->o2, labels, net->batch_size, net->dims[2]);

    // write_mat("temp/dLoss_do2.csv", dLoss_do2, net->batch_size, net->dims[2]);

    // calculate do2/du2 matrix
    // float *do2_du2 = dReLU(cache->o2, net->batch_size, net->dims[2]);

    // write_mat("temp/dReLU_do2.csv", do2_du2, net->batch_size, net->dims[2]);

    // multiply the two matrices element wise
    float *dLoss_du2 = dsoftmax(cache->o2, labels, net->batch_size, net->dims[2]);

    // write_mat("temp/dLoss_du2.csv", dLoss_du2, net->batch_size, net->dims[2]);

    // transpose o1
    float *du_dl2 = transpose(cache->o1, net->batch_size, net->dims[1]);

    // write_mat("temp/du_dl2.csv", du_dl2, net->dims[1], net->batch_size);

    // calculate dLoss/dl2
    float *dLoss_dl2 = mat_mul(du_dl2, dLoss_du2, net->dims[1], net->batch_size, net->batch_size, net->dims[2]);
    
    // write_mat("temp/dLoss_dl2.csv", dLoss_dl2, net->dims[1], net->dims[2]);

    // du2/db2 is an all ones matrix
    float *dLoss_db2 = row_sum(dLoss_du2, net->batch_size, net->dims[2]); // add rows together

    // write_mat("temp/dLoss_db2.csv", dLoss_db2, 1, net->dims[2]);

    // multiply weight update by the learning rate
    scalar_multiply(dLoss_dl2, &net->learning_rate, net->dims[1], net->dims[2]);

    // write_mat("temp/dLoss_dl2_scaled.csv", dLoss_dl2, net->dims[1], net->dims[2]);

    // multiply bias update by the learning rate
    scalar_multiply(dLoss_db2, &net->learning_rate, 1, net->dims[2]);

    // du2/do1 is equal to net->l2
    float *du2_do1 = transpose(net->l2, net->dims[1], net->dims[2]);

    float *dLoss_do1 = mat_mul(dLoss_du2, du2_do1, net->batch_size, net->dims[2], net->dims[2], net->dims[1]);

    float *do1_du1 = dReLU(cache->o1, net->batch_size, net->dims[1]);

    float *dLoss_du1 = el_wise_mat_mul(dLoss_do1, do1_du1, net->batch_size, net->dims[1]);

    float *du1_dl1 = transpose(inputs, net->batch_size, net->dims[0]);

    float *dLoss_dl1 = mat_mul(du1_dl1, dLoss_du1, net->dims[0], net->batch_size, net->batch_size, net->dims[1]);

    // du1/db1 is an all ones matrix
    float *dLoss_db1 = row_sum(dLoss_du1, net->batch_size, net->dims[1]); // add rows together

    // multiply by the learning rate
    scalar_multiply(dLoss_dl1, &net->learning_rate, net->dims[0], net->dims[1]);

    // multiply bias update by the learning rate
    scalar_multiply(dLoss_db1, &net->learning_rate, 1, net->dims[1]);

    // clip gradients
    clip_gradients(dLoss_dl1, net->dims[0], net->dims[1]);
    clip_gradients(dLoss_db1, 1, net->dims[1]);
    clip_gradients(dLoss_dl2, net->dims[1], net->dims[2]);
    clip_gradients(dLoss_db2, 1, net->dims[2]);


    // update layer 2
    mat_sub(net->l2, dLoss_dl2, net->dims[1], net->dims[2]);
    mat_sub(net->b2, dLoss_db2, 1, net->dims[2]);               // these functions are probably slower on cuda

    // write_mat("temp/l2_updated.csv", net->l2, net->dims[1], net->dims[2]);

    // update layer 1
    mat_sub(net->l1, dLoss_dl1, net->dims[0], net->dims[1]);
    mat_sub(net->b1, dLoss_db1, 1, net->dims[1]);               // these functions are probably slower on cuda

    if (std::isnan(net->l1[0]))
    {
        write_mat("broken/l1.csv", net->l1, net->dims[0], net->dims[1]);
        write_mat("broken/b1.csv", net->b1, 1, net->dims[1]);
        write_mat("broken/l2.csv", net->l2, net->dims[1], net->dims[2]);
        write_mat("broken/b2.csv", net->b2, 1, net->dims[2]);

        write_mat("broken/dLoss_dl1.csv", dLoss_dl1, net->dims[1], net->dims[2]);
        write_mat("broken/dLoss_db1.csv", dLoss_db1, 1, net->dims[2]);
        std::cout << "shit! gradients exploded\n";
        exit(1);
    }

    // checkCuda( cudaFreeHost(dLoss_do2) );
    // checkCuda( cudaFreeHost(do2_du2) );
    checkCuda( cudaFreeHost(dLoss_du2) );
    checkCuda( cudaFreeHost(du_dl2) );
    checkCuda( cudaFreeHost(dLoss_dl2) );
    checkCuda( cudaFreeHost(dLoss_db2) );
    checkCuda( cudaFreeHost(du2_do1) );
    checkCuda( cudaFreeHost(dLoss_do1) );
    checkCuda( cudaFreeHost(do1_du1) );
    checkCuda( cudaFreeHost(dLoss_du1) );
    checkCuda( cudaFreeHost(du1_dl1) );
    checkCuda( cudaFreeHost(dLoss_dl1) );
    checkCuda( cudaFreeHost(dLoss_db1) );

}

void test(network *net, out_cache *cache, float *inputs, uint *labels, uint test_size)
{

    // determine number of iterations necessary to classify all inputs
    int num_iters = test_size / net->batch_size;
    int remainder = test_size - (num_iters * net->batch_size);

    float *input_batch = new float[net->batch_size*net->dims[0]];
    uint *label_batch = new uint[net->batch_size];
    int num_correct = 0;

    // FOR TESTING:
    // num_iters = 1;

    // for indexing the inputs based on the batch number
    int img_idx = 0;
    int label_idx = 0;
    for (int i = 0; i < num_iters; i++)
    {
        img_idx = i*net->batch_size*net->dims[0];
        label_idx = i*net->batch_size;
        // printf("label_idx: %d\n", label_idx);
        select_batch(inputs, input_batch, labels, label_batch, net->batch_size, net->dims[0], img_idx, label_idx, test_size);

        // print the batch
        // write_mat("temp/first_batch.csv", input_batch, net->batch_size, net->dims[0]);

        // forward pass for the batch
        forward(net, cache, input_batch);
        if (i == 0) {
            //write outputs and labels
            write_mat("outputs/test_outputs.csv", cache->o2, net->batch_size, net->dims[2]);
            write_img("outputs/test_labels.csv", labels, net->batch_size, 1);
        }

        // calculate the number of outputs that are correct
        num_correct += calc_num_correct(cache, labels, net->batch_size, net->dims[2]);
    }

    delete[] input_batch;
    delete[] label_batch;

    uint old_batch_size;
    if (remainder > 0)
    {
        std::cout << "calculating remainder\n";
        img_idx += net->batch_size*net->dims[0];
        label_idx += net->batch_size;

        // calculate remaining images
        old_batch_size = net->batch_size;
        net->batch_size = remainder;
        float *input_remainder = new float[net->batch_size*net->dims[0]];
        uint *label_remainder = new uint[net->batch_size];
        select_batch(inputs, input_batch, labels, label_batch, net->batch_size, net->dims[0], img_idx, label_idx, test_size);

        forward(net, cache, input_remainder);

        // calculate the number of outputs that are correct
        num_correct += calc_num_correct(cache, labels, net->batch_size, net->dims[2]);

        delete[] input_remainder;
        delete[] label_remainder;
        net->batch_size = old_batch_size;
    }

    printf("%d / %d\n", num_correct, test_size);

}



void train(network *net, out_cache *cache, float *inputs, uint *labels)
{

    // write_mat("first_img.csv", inputs, 28, 28);

    // determine number of iterations necessary to classify all inputs
    int num_iters = net->train_size / net->batch_size;
    int remainder = net->train_size - (num_iters * net->batch_size);
    if (remainder > 0) { num_iters++; }

    float *input_batch = new float[net->batch_size*net->dims[0]];
    uint *label_batch = new uint[net->batch_size];

    // FOR TESTING:
    // num_iters = 100;
    double loss = 2.5;
    int offset = 0;
    int batch_offset = 0;
    for (int n = 0; n < net->num_epochs; n++)
    {
        // for indexing the inputs based on the batch number
        int img_idx = 0;
        int label_idx = 0;
        for (int i = 0; i < num_iters; i++)
        {
            img_idx = i*net->batch_size*net->dims[0] + batch_offset*net->dims[0];
            label_idx = i*net->batch_size + batch_offset;

            // printf("label_idx: %d\n", label_idx);

            // printf("img_idx: %d\n", img_idx);
            // printf("label_idx: %d\n", label_idx);
            // printf("image index: %d\n", img_idx);

            offset = select_batch(inputs, input_batch, labels, label_batch, net->batch_size, net->dims[0], img_idx, label_idx, net->train_size);

            // print the batch
            // write_mat("temp/first_batch.csv", input_batch, net->batch_size, net->dims[0]);

            // forward pass for the batch
            // std::cout << "forward pass\n";
            forward(net, cache, input_batch);        

            if ((n % 10 == 0) && i == 0) {
                //write outputs and labels
                write_mat("outputs/train_outputs.csv", cache->o2, net->batch_size, net->dims[2]);
                write_img("outputs/train_labels.csv", labels, net->batch_size, 1);
            }

            // backward pass
            backward(net, cache, input_batch, label_batch);

        }

        if (offset > 0) { 
            batch_offset = offset; 
            std::cout << "updated offset to: " << batch_offset << '\n';
        }

        // compute loss
        loss = calc_loss(net, cache, label_batch);  

        // track the loss
        append_loss(loss);


        // if (n % 10 == 0) 
        // {
        //     std::cout << "saving weights\n";
        //     save_weights(net);
        // }

    }

    // training accuracy
    std::cout << "training accuracy:\n";
    test(net, cache, inputs, labels, net->train_size);

    // save outputs
    std::cout << "saving outputs\n";
    write_mat("outputs/train_output.csv", cache->o2, net->batch_size, net->dims[2]);
    write_img("outputs/train_labels.csv", labels, net->batch_size, 1);

    std::cout << "saving weights\n";
    save_weights(net);

    
    delete[] input_batch;
    delete[] label_batch;


}

// NEED TO INVERT THE LOSS FUNCTION?
int main(int argc, char *argv[])
{
    // check inputs
    bool good = check_inputs(argc, argv);
    if (!good) { return 1; }

    // get inputs
    int num_HU = atoi(argv[1]);
    int batch_size = atoi(argv[2]);
    int num_epochs = atoi(argv[3]);
    bool load_weights = atoi(argv[4]);

    // set cuda device
    int devID = gpuDeviceInit(1);

    if (devID < 0) {
        printf("exiting...\n");
        exit(EXIT_FAILURE);
    }


    // initialize network
    network *net = new network;     // struct holding neural network parameters
    out_cache *cache = new out_cache;       // struct holding intermediate results

    init_nn(net, cache, num_HU, batch_size, num_epochs, load_weights);

    // write weights for testing
    // write_mat("temp/layer_1_weights.csv", net->l1, net->dims[0], net->dims[1]);
    // write_mat("temp/layer_2_weights.csv", net->l2, net->dims[1], net->dims[2]);
    if (net->num_epochs > 0) 
    { 
        // clear loss file
        // std::ofstream ofs;
        // ofs.open("loss.csv", std::ofstream::trunc);
        // ofs.close();

        // load training dataset and labels
        int *inputs_train_int = new int[net->dims[0]*net->train_size];
        uint *labels_train = new uint[net->train_size];

        load_mnist_images(inputs_train_int, "dataset/train-images-idx3-ubyte");
        load_mnist_labels(labels_train, "dataset/train-labels-idx1-ubyte");

        // write_img("outputs/train_labels_ini.csv", labels_train, net->batch_size, 1);


        // convert inputs to floats and normalize to [0,1]
        float *inputs_train = normalize(inputs_train_int, net->dims[0]*net->train_size, 255.0);

        delete[] inputs_train_int;

        // write_img("first_img_int.csv", inputs_train_int, 28, 28);
        // write_mat("first_img.csv", inputs_train, 28, 28);

        train(net, cache, inputs_train, labels_train);


        delete[] inputs_train;
        delete[] labels_train;
    }

    // load testing dataset and labels
    int *inputs_test_int = new int[net->dims[0]*net->test_size];
    uint *labels_test = new uint[net->test_size];

    load_mnist_images(inputs_test_int, "dataset/t10k-images-idx3-ubyte");
    load_mnist_labels(labels_test, "dataset/t10k-labels-idx1-ubyte");

    // load_mnist(inputs_test_int, labels_test, net->test_size, net->dims[0], "mnist_test.csv");

    // write_img("outputs/ini_test_labels.csv", labels_test, net->batch_size, 1);

    // normalize inputs
    float *inputs_test = normalize(inputs_test_int, net->dims[0]*net->test_size, 255.0);

    delete[] inputs_test_int;

    std::cout << "testing dataset accuracy:\n";
    test(net, cache, inputs_test, labels_test, net->test_size);


    // // save weights
    // std::cout << "saving weights\n";
    // save_weights(net);
 

    delete[] inputs_test;
    delete[] labels_test;

    delete net;
    delete cache;

    return 0;

}