#include "NN.h"


NN::NN(int l, int* n)
{
    layers = l;
    number_neurons = n;
    std::cout.precision(3);
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0, 1.0);
    weights = new float**[layers];
    constants = new float*[layers];
    for(int layer = 1; layer < layers; layer++)
    {
        const int neurons_previous_layer = number_neurons[layer - 1];
        const int neurons_next_layer = number_neurons[layer];
        weights[layer] = new float*[neurons_next_layer];
        for(int i = 0; i < neurons_next_layer; i++)
        {
            weights[layer][i] = new float[neurons_previous_layer];
            for(int j = 0; j < neurons_previous_layer; j++)
            {
                NN::weights[layer][i][j] = distribution(generator);
            }
        }
        constants[layer] = new float[neurons_next_layer];
        for(int i = 0; i < neurons_next_layer; i++)
        {
            constants[layer][i] = distribution(generator);
        }
    }
    last_run = new float*[layers];
    for(int layer = 0; layer < layers; layer++)
    {
        last_run[layer] = new float[number_neurons[layer]];
    }
}

float NN::sigma(float z){
    return 1.0 / (1.0 + exp(-1.0 * z));
}

float* NN::execute(float* inputs)
{
    float* input_layer = inputs;
    last_run[0] = input_layer;
    float* output_layer;
    for(int layer = 1; layer < layers; layer++)
    {
        int number_output_neurons = number_neurons[layer];
        int number_input_neurons = number_neurons[layer - 1];
        output_layer = new float[number_output_neurons];
        for(int i = 0; i < number_output_neurons; i++)
        {
            float z = constants[layer][i];
            for(int j = 0; j < number_input_neurons; j++)
            {
                z += weights[layer][i][j] * input_layer[j];
            }
            output_layer[i] = sigma(z);
        }
        input_layer = output_layer;
        last_run[layer] = output_layer;
    }
    return output_layer;
}

float*** NN::calculate_weight_derivatives(float* output, float** constant_derivatives)
{
    //Please call the execute function on inputs before you call this.
    //This function depends on last_pass being set correctly
    float*** weight_derivatives = new float**[layers];
    for(int modify_layer = 1; modify_layer < layers; modify_layer++)
    {
        const int number_input_neurons = number_neurons[modify_layer - 1];
        const int number_output_neurons = number_neurons[modify_layer];
        weight_derivatives[modify_layer] = new float*[number_output_neurons];
        for(int i = 0; i < number_output_neurons; i++)
        {
            weight_derivatives[modify_layer][i] = new float[number_input_neurons];
            for(int j = 0; j < number_input_neurons; j++)
            {
                weight_derivatives[modify_layer][i][j] = constant_derivatives[modify_layer][i] * last_run[modify_layer - 1][j];
            }
        }
    }
    return weight_derivatives;
}

float** NN::calculate_constant_derivatives(float* output)
{
    float** constant_derivatives = new float*[layers];
    for(int modify_layer = layers - 1; modify_layer >= 1; modify_layer--)
    {
        const int number_input_neurons = number_neurons[modify_layer];
        constant_derivatives[modify_layer] = new float[number_input_neurons];
        if(modify_layer == layers - 1)
        {
            for(int i = 0; i < number_input_neurons; i++)
            {
                constant_derivatives[modify_layer][i] = last_run[modify_layer][i] - output[i];
            }
        }
        else
        {
            const int number_output_neurons = number_neurons[modify_layer + 1];
            for(int i = 0; i < number_input_neurons; i++)
            {
                constant_derivatives[modify_layer][i] = 0;
                for(int j = 0; j < number_output_neurons; j++)
                {
                    constant_derivatives[modify_layer][i] += constant_derivatives[modify_layer + 1][j] * last_run[modify_layer + 1][j] * (1 - last_run[modify_layer + 1][j]) * weights[modify_layer + 1][j][i];
                }
            }
        }
    }
    return constant_derivatives;
}

float NN::cost(float** inputs, float** outputs, int number_data)
{
    float sum = 0;
    for(int n = 0; n < number_data; n++)
    {
        float* prediction = this->execute(inputs[n]);
        for(int i = 0; i < number_neurons[layers - 1]; i++)
        {
            sum += 0.5 * (prediction[i] - outputs[n][i]) * (prediction[i] - outputs[n][i]) / number_data;
        }
    }
    return sum;
}

void NN::train(float eta, float** inputs, float** outputs, int number_data, int mini_batch_size)
{
    eta /= mini_batch_size;
    auto seed = unsigned(std::time(0));
    std::srand(seed);
    std::random_shuffle(inputs, inputs + number_data);
    std::srand(seed);//Guarantess that it'll shuffle the same way.
    std::random_shuffle(outputs, outputs + number_data);
    for(int m = 0; m < number_data; m += mini_batch_size)
    {
        float*** aggregate_constant_derivatives = new float**[mini_batch_size];
        float**** aggregate_weight_derivatives = new float***[mini_batch_size];
        for(int n = 0; n < mini_batch_size; n++)
        {
            this->execute(inputs[n + m]);
            aggregate_constant_derivatives[n] = calculate_constant_derivatives(outputs[n + m]);
            aggregate_weight_derivatives[n] = calculate_weight_derivatives(outputs[n + m], aggregate_constant_derivatives[n]);
        }
        for(int n = 0; n < mini_batch_size; n++)
        {
            for(int layer = 1; layer < layers; layer++)
            {
                for(int i = 0; i < number_neurons[layer]; i++)
                {
                    for(int j = 0; j < number_neurons[layer - 1]; j++)
                    {
                        weights[layer][i][j] -= eta * aggregate_weight_derivatives[n][layer][i][j];
                    }
                    constants[layer][i] -= eta * aggregate_constant_derivatives[n][layer][i];
                }
            }
        }
    }
}

void NN::print()
{
    using namespace std;
    for(int layer = 0; layer < layers; layer++)
    {
        cout << "Layer " << layer << ": ";
        for(int j = 0; j < number_neurons[layer]; j++)
        {
            cout << "o";
        }
        cout << endl;
    }
    for(int layer = 1; layer < layers; layer++)
    {
        cout << "Layer " << layer << endl;
        cout << "   from ->" << endl;
        cout << "to" << endl;
        cout << "|" << endl;
        cout << "v" << endl;
        cout << "Weights" << endl;
        for(int i = 0; i < number_neurons[layer]; i++)
        {
            for(int j = 0; j < number_neurons[layer - 1]; j++)
            {
                cout << weights[layer][i][j];
                cout << " ";
            }
            cout << endl;
        }
        cout << "Constants";
        cout << endl;
        for(int i = 0; i < number_neurons[layer]; i++)
        {
            cout << constants[layer][i];
            cout << endl;
        }
    }
    cout << endl;
}

void NN::print_run(float* input)
{
    using namespace std;
    this -> execute(input);
    for(int layer = 0; layer < layers; layer++)
    {
        cout << "Layer " << layer << "    ";
        for(int i = 0; i < number_neurons[layer]; i++)
        {
            cout << last_run[layer][i] << "    ";
        }
        cout << endl;
    }
}

int NN::correct(float** inputs, float** outputs, int number_data, int number_to_test)
{
    auto seed = unsigned(std::time(0));
    std::srand(seed);
    std::random_shuffle(inputs, inputs + number_data);
    std::srand(seed);//Guarantess that it'll shuffle the same way.
    std::random_shuffle(outputs, outputs + number_data);
    int correct = 0;
    for(int n = 0; n < number_to_test; n++)
    {
        float* prediction = execute(inputs[n]);
        int max_index = 0;
        float m = 0;
        int c_max_index = 0;
        float c_m = 0;
        for(int i = 0; i < 10; i++)
        {
            if(prediction[i] > m)
            {
                m = prediction[i];
                max_index = i;
            }
            if(outputs[n][i] > c_m)
            {
                c_m = outputs[n][i];
                c_max_index = i;
            }
        }
        if(c_max_index == max_index)
        {
            correct++;
        }
    }
    return correct;
}

NN::~NN()
{
    //dtor
}
