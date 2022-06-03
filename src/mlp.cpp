#include <iostream>
#include <cmath>
#include <ctime>
#include <utility>
#include <vector>
#include <armadillo>
#include <string>

using namespace std;
using namespace arma;


class NeuralNetwork {
    public:
        int input_length;
        int n_hidden_layers;
        int output_length;
        field<colvec> bias;
        field<mat> weights;
        string activation_function="sigm";
public:

    NeuralNetwork(int input_length, int n_hidden_layers, vector<int> neurons_per_layer, int output_length) {
        this->input_length = input_length;
        this->n_hidden_layers = n_hidden_layers;
        this->output_length = output_length;
        
        bias = field<colvec>(n_hidden_layers+1);
        weights = field<mat>(n_hidden_layers+1);
        

        for (int i = 0; i <= n_hidden_layers; i++){
            if(i==0){
                weights(i) = mat(input_length, neurons_per_layer[i],fill::randu);
                bias(i) = colvec(neurons_per_layer[i], fill::randu);
            }
            else if(i<n_hidden_layers){
                weights(i) = mat(neurons_per_layer[i-1], neurons_per_layer[i],fill::randu);
                bias(i) = colvec(neurons_per_layer[i], fill::randu);
            }
            else{
                weights(i) = mat(neurons_per_layer[i-1], output_length,fill::randu);
                bias(i) = colvec(output_length, fill::randu);
            }    
        }
    }

    rowvec sigm(rowvec input){
        rowvec output = 1 / (1+exp(-input));
        return output;
    }

    rowvec tanh(rowvec z){
        return z;
    }

    rowvec relu(rowvec z){
        return z;
    }


    rowvec apply_activation_function(rowvec input) {
        if (this->activation_function == "sigm"){
            return sigm(input);
        }
        else if(this->activation_function == "tanh"){
            return tanh(input);
        }
        else{
            return relu(input);
        }

    }


    colvec forward_propagation(rowvec input) {
        colvec neths;
        for(int i = 0; i<=n_hidden_layers;i++){
            input = apply_activation_function(input*weights(i) + bias(i).t());
        }
        return input.t();
    }

    void propagate_backward() {

    }

    // void learn_image() {

    // }

};

int main(){

    arma_rng::set_seed(42);
    vector<int> capas{3, 2, 3, 2};
    NeuralNetwork mlp(5, 4, capas, 7);

    // for(int i=0; i<mlp.weights.n_elem;i++){
    //     cout<<"Weight"<<i<<":\n"<<mlp.weights(i);
    // }
    rowvec input(5, fill::randu);
    // mlp.forward_propagation(input).print("output: ");
    // mat pesos(5,3, fill::randu);
    // colvec bias(3, fill::randu);
    input.print("input:");
    input = 1/ input;
    input.print("input:");

    // pesos.print("pesos:");
    // bias.print("Bias:");
    // mat c = input*pesos + bias.t();
    // c.print("C:");
    // c = c + bias.t();
    // c.print("C:");
    return 0;
}