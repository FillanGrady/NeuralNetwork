#ifndef NN_H
#define NN_H
#include <random>
#include <iostream>
#include <string>
#include <ctime>
#include <algorithm>

class NN
{
    public:
        NN(int, int*);
        virtual ~NN();
        float* execute(float*);
        void train(float, float**, float**, int, int);
        float cost(float**, float**, int);
        int correct(float**, float**, int, int);
        void print();
        void print_run(float*);
    protected:
    private:
        int layers;
        int* number_neurons;
        float*** weights;
        float** constants;
        float** last_run;
        float sigma(float);
        float*** calculate_weight_derivatives(float*, float**);
        float** calculate_constant_derivatives(float*);
};

#endif // NN_H
