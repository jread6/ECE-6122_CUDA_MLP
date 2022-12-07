#include "cuda_helper.h"

// modified from: https://stackoverflow.com/questions/8286668/how-to-read-mnist-data-in-c
int reverseInt (int i) 
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void load_mnist(int *images, uint *labels, uint data_size, uint img_size, std::string filename)
{
    std::ifstream file;
    file.open(filename, std::fstream::in);
    for (int i = 0; i < data_size; i++)
    {
        file >> labels[i];
        for (int j = 0; j < img_size; j++)
        {
            file >> images[i*(img_size+1) + j];
        }
    }
}

// modified from: https://stackoverflow.com/questions/8286668/how-to-read-mnist-data-in-c 
void load_mnist_labels(uint *labels_train, std::string filename)
{
    std::ifstream file;
    file.open(filename, std::ios::binary);

    if (file.is_open())
    {
        int magic_number=0;
        int number_of_labels=0;

        file.read((char*)&magic_number,sizeof(magic_number)); 
        magic_number= reverseInt(magic_number);
        if(magic_number != 2049) throw std::runtime_error("Invalid MNIST image file!");

        file.read((char*)&number_of_labels,sizeof(number_of_labels));
        number_of_labels = reverseInt(number_of_labels);

        for(int i=0;i<number_of_labels;++i)
        {
            file.read((char*)&labels_train[i], 1); // read one char at a time
        }
    }
    else {
        throw std::runtime_error("Unable to open file `" + filename + "`!");
    }
}

// modified from: https://stackoverflow.com/questions/8286668/how-to-read-mnist-data-in-c 
void load_mnist_images(int *inputs_train, std::string filename)
{
    std::ifstream file;
    file.open(filename, std::ios::binary);

    if (file.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;

        file.read((char*)&magic_number,sizeof(magic_number)); 
        magic_number= reverseInt(magic_number);
        if(magic_number != 2051) throw std::runtime_error("Invalid MNIST image file!");

        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);

        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows= reverseInt(n_rows);

        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols= reverseInt(n_cols);

        for(int i=0; i<number_of_images*n_rows*n_cols; i++)
        {
            file.read((char*)&inputs_train[i], 1); // read one char at a time            
        }
    }
    else {
        throw std::runtime_error("Unable to open file `" + filename + "`!");
    }

}


void write_mat(std::string filename, float *H, uint rows, uint cols) 
{
    std::fstream outFile;

    outFile.open(filename, std::fstream::out);

    for (uint i = 0; i < rows; i++) 
    {
        for (uint j = 0; j < cols; j++) 
        {
            outFile << H[i*cols + j];
            if (j < cols-1) outFile << ',';
        }
        outFile << '\n';
    }

    outFile.close();

    return;
}

void read_mat(std::string filename, float *mat, uint rows, uint cols) 
{
    std::ifstream inFile;

    inFile.open(filename, std::fstream::in);
    char garb;

    for (uint i = 0; i < rows; i++) 
    {
        for (uint j = 0; j < cols; j++) 
        {
            inFile >> mat[i*cols + j];
            inFile >> garb;
            // if (j < cols-1) outFile << ',';
        }
        // outFile << '\n';
    }

    inFile.close();

    return;
}

void write_img(std::string filename, uint *H, uint rows, uint cols) 
{
    std::fstream outFile;

    outFile.open(filename, std::fstream::out);

    for (uint i = 0; i < rows; i++) 
    {
        for (uint j = 0; j < cols; j++) 
        {
            outFile << H[i*cols + j] << ',';
        }
        outFile << '\n';
    }

    outFile.close();

    return;
}

void init_nn(network *net, out_cache *cache, int num_HU, int batch_size, int num_epochs, bool load_weights)
{
    net->dims = new int[3];
    net->dims[0] = 28*28;       // MNIST images are 28 x 28
    net->dims[1] = num_HU;
    net->dims[2] = 10;

    net->batch_size = batch_size;
    net->num_epochs = num_epochs;
    net->load_weights = load_weights;

    net->l1 = new float[net->dims[0]*net->dims[1]];
    net->b1 = new float[net->dims[1]]();        // initialize biases to 0

    // // testing
    // for (int i = 0; i < net->dims[1]; i++)
    // {
    //     net->b1[i] = (float)i;
    // }
    // write_mat("temp/b1.csv", net->b1, 1, 300);

    net->l2 = new float[net->dims[1]*net->dims[2]];
    net->b2 = new float[net->dims[2]]();        // initialize biases to 0

    // load weights
    if (net->load_weights == true)
    {
        std::cout << "loading previously saved weights\n";
        read_mat("current_weights/l1.csv", net->l1, net->dims[0], net->dims[1]);
        read_mat("current_weights/b1.csv", net->b1, 1, net->dims[1]);
        read_mat("current_weights/l2.csv", net->l2, net->dims[1], net->dims[2]);
        read_mat("current_weights/b2.csv", net->b2, 1, net->dims[2]);
        // write_mat("loaded_l1.csv", net->l1, net->dims[0], net->dims[1]);
        // write_mat("first_img.csv", inputs_test, 28, 28);
    }
    else
    {
        // randomly initialize weights
        randomize(net->l1, net->dims[0]*net->dims[1]);
        randomize(net->l2, net->dims[1]*net->dims[2]);
    }

    // write_mat("current_weights/check_l1.csv", net->l1, net->dims[0], net->dims[1]);



    return;

}

void randomize(float *array, int array_size)
{
    // srand(1); // initialize seed for testing
    for (int i = 0; i < array_size; i++)
    {
        // initialize between 0 and 1
        array[i] = (float)rand() / ((float)RAND_MAX*1000);
        // array[i] = rand();
        // std::cout << array[i];
    }
}

bool check_inputs(int argc, char *argv[])
{
    if (argc != 5) 
    {
        std::cout << "error! wrong number of inputs!\n";
        std::cout << "usage: [number of hidden units] [batch size] [train]\n";
        return false;
    }
    else if (!isdigit(*argv[1]) || !isdigit(*argv[2]) || !isdigit(*argv[3]))
    {
        std::cout << "error! number of hidden units and batch size";
        std::cout << " must be numbers\n";
        std::cout << "usage: [number of hidden units] [batch size] [train]\n";
        return false;
    }

    return true;
}

int select_batch(float *inputs, float *input_batch, uint *labels, uint *label_batch, uint batch_size, uint img_size, int img_idx, int label_idx, int data_size)
{
    int label_offset = 0;
    int input_offset = 0;

    // select a batch of images and labels
    for (int b = 0; b < batch_size; b++)
    {
        if (b+label_idx > data_size)
        {
            label_offset = (b+label_idx) - data_size;

            // if (b==0) printf("label offset: %d\n", label_offset);
            label_batch[b] = labels[label_offset];
        }
        else
        {
            label_batch[b] = labels[b+label_idx];
        }
    }

    for (int b = 0; b < batch_size*img_size; b++)
    {

        if (b+img_idx > data_size*img_size)
        {
            input_offset = (b+img_idx) - data_size*img_size;
            input_batch[b] = inputs[input_offset];
        }
        else
        {
            // std::cout << b << '\n';
            input_batch[b] = inputs[b+img_idx];
        }
    }

    return label_offset;
}

void append_loss(double loss) 
{
    std::fstream outFile;

    outFile.open("loss.csv", std::fstream::app);

    outFile << loss << '\n';

    outFile.close();

    return;
}

void save_weights(network *net)
{
    write_mat("current_weights/l1.csv", net->l1, net->dims[0], net->dims[1]);
    write_mat("current_weights/b1.csv", net->b1, 1, net->dims[1]);
    write_mat("current_weights/l2.csv", net->l2, net->dims[1], net->dims[2]);
    write_mat("current_weights/b2.csv", net->b2, 1, net->dims[2]); 
}

double calc_loss(network *net, out_cache *cache, uint *labels)
{
    double loss = 0;
    for (int b = 0; b < net->batch_size; b++)
    {
        // use categorical cross-entropy loss
        loss += -log(cache->o2[b*net->dims[2] + labels[b]]); // ex: o2[1] is the probability of predicting a '1'
    }

    return loss / net->batch_size;
}

int calc_num_correct(out_cache *cache, uint *labels, int batch_size, int num_classes)
{
    int num_correct = 0;
    for (int b = 0; b < batch_size; b++)
    {
        // find correct label
        // int label_idx = labels[b];

        // if (b < 20) { printf("%d\n", labels[b]); } 

        // find most probable class
        float max = 0;
        int pred = 0;
        for (int i = 0; i < num_classes; i++) 
        {
            float val = cache->o2[b*num_classes + i];
            if (val > max)
            {
                max = val;
                pred = i;
            }
        }

        if (pred == labels[b])
        {
            // std::cout << "image " << b << " was correct!\n";
            // std::cout << "label is: " << label_idx << '\n';
            num_correct++;
        }
            
    }

    return num_correct;
}


float *normalize(int *inputs, uint input_size, float max_val)
{
    float *inputs_norm = new float[input_size];
    for (uint i = 0; i < input_size; i++)
    {
        inputs_norm[i] = (float)inputs[i] / max_val;
    }

    return inputs_norm;
}

int gpuDeviceInit(int devID) {
  int device_count;
  checkCuda(cudaGetDeviceCount(&device_count));

  if (device_count == 0) {
    fprintf(stderr,
            "gpuDeviceInit() CUDA error: "
            "no devices supporting CUDA.\n");
    exit(EXIT_FAILURE);
  }

  if (devID < 0) {
    devID = 0;
  }

  if (devID > device_count - 1) {
    fprintf(stderr, "\n");
    fprintf(stderr, ">> %d CUDA capable GPU device(s) detected. <<\n",
            device_count);
    fprintf(stderr,
            ">> gpuDeviceInit (-device=%d) is not a valid"
            " GPU device. <<\n",
            devID);
    fprintf(stderr, "\n");
    return -devID;
  }

  cudaDeviceProp deviceProp;
  checkCuda(cudaGetDeviceProperties(&deviceProp, devID));

  if (deviceProp.computeMode == cudaComputeModeProhibited) {
    fprintf(stderr,
            "Error: device is running in <Compute Mode "
            "Prohibited>, no threads can use cudaSetDevice().\n");
    return -1;
  }

  if (deviceProp.major < 1) {
    fprintf(stderr, "gpuDeviceInit(): GPU device does not support CUDA.\n");
    exit(EXIT_FAILURE);
  }

  checkCuda(cudaSetDevice(devID));
  printf("gpuDeviceInit() CUDA Device [%d]: \"%s\n", devID, deviceProp.name);

  return devID;
}

void clip_gradients_cpu(float *mat, uint size)
{
    for (uint i = 0; i < size; i++)
    {
        if (mat[i] < -1)
        {
            mat[i] = -1;
            std::cout << "clipping negative gradients\n";
        }
        else if (mat[i] > 1)
        {
            mat[i] = 1;
            std::cout << "clipping positive gradients\n";

        }
    }
}