#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <random>
#include <string>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <ctime>
#include "NN.h"
using namespace std;

int reverse_int(int i)
{
    unsigned char c1 = i & 255;
    unsigned char c2 = (i >> 8) & 255;
    unsigned char c3 = (i >> 16) & 255;
    unsigned char c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

float** read_mnist(string full_path)
{
    typedef unsigned char uchar;
    ifstream file(full_path);
    if(file.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverse_int(magic_number);
        file.read((char*)&number_of_images, sizeof(number_of_images));
        number_of_images = reverse_int(number_of_images);
        file.read((char*)&n_rows, sizeof(n_rows));
        n_rows = reverse_int(n_rows);
        file.read((char*)&n_cols, sizeof(n_cols));
        n_cols = reverse_int(n_cols);
        int image_size = n_rows * n_cols;
        float** _dataset = new float*[number_of_images];
        for(int i = 0; i < number_of_images; i++)
        {
            _dataset[i] = new float[image_size];
            uchar* temp = new uchar[image_size];
            file.read((char*)temp, image_size);
            for(int j = 0; j < image_size; j++)
            {
                _dataset[i][j] = temp[j];
            }
        }
        return _dataset;
    }
    else
    {
        throw runtime_error("Cannot open file " + full_path);
    }
}

float** read_mnist_labels(string full_path)
{
    typedef unsigned char uchar;
    ifstream file(full_path);
    if(file.is_open())
    {
        int magic_number = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverse_int(magic_number);
        int number_of_labels = 0;
        file.read((char*)&number_of_labels, sizeof(number_of_labels));
        number_of_labels = reverse_int(number_of_labels);
        float** _dataset = new float*[number_of_labels];
        for(int i = 0; i < number_of_labels; i++)
        {
            _dataset[i] = new float[10];
            uchar temp = 0;
            file.read((char*)&temp, 1);
            for(int j = 0; j < 10; j++)
            {
                _dataset[i][j] = 0;
            }
            _dataset[i][temp] = 1.0;
        }
        return _dataset;
    }
    else
    {
        throw runtime_error("Cannot open file " + full_path);
    }
}



int main(int argc, char* argv[])
{
    int layers = argc - 1;
    int* number_neurons = new int[layers];
    for(int i = 0; i < layers; i++)
    {
        number_neurons[i] = atoi(argv[i+1]);
    }
    NN* myNN = new NN(layers, number_neurons);
    float** images = read_mnist("/home/fillan/Documents/NeuralNetwork/NeuralNetwork/bin/train-images.idx3-ubyte");
    float** labels = read_mnist_labels("/home/fillan/Documents/NeuralNetwork/NeuralNetwork/bin/train-labels.idx1-ubyte");

    float** inputs = new float*[3];
    float** outputs = new float*[3];
    for(int i = 0; i < 3; i++)
    {
        inputs[i] = new float[3];
        outputs[i] = new float[3];
        for(int j = 0; j < 3; j++)
        {
            if(i == j)
            {
                inputs[i][j] = 1;
                outputs[i][j] = 1;
            }
            else
            {
                inputs[i][j] = 0;
                outputs[i][j] = 0;
            }
        }
    }
    int number_to_test = 100;
    for(int i = 0; i < 1; i++)
    {
        myNN->train(3.0, images, labels, 60000, 60);
        cout << "Epoch: " << i << "  ";
        cout << myNN->correct(images, labels, 60000, number_to_test) << "/" << number_to_test << endl;
    }
    cout << "Cost: " << myNN->cost(images, labels, 10) << endl;
    return 0;
}
