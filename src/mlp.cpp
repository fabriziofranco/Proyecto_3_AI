#include <iostream>
#include <cmath>
#include <ctime>
#include <utility>
#include <vector>
#include <armadillo>
#include <string>
#include <algorithm>

using namespace std;
using namespace arma;


class NeuralNetwork {
    private:
        int input_length;
        int n_hidden_layers;
        int output_length;
        field<colvec> bias;
        field<rowvec> neurons_outputs;
        field<rowvec> deltas;
        field<mat> weights;
        string activation_function;

private:

    rowvec sigm(rowvec input){
        rowvec output = 1 / (1+exp(-input));
        return output;
    }

    rowvec tanh(rowvec input){
        rowvec output = (2 / (1+exp(-input*2))) - 1;
        return output;
    }

    rowvec relu(rowvec input){
        rowvec output = input;
        for(int i=0; i < output.n_elem; i++)
            if (output(i) < 0) output(i) = 0;
        return output;
    }


    rowvec softmax(rowvec input) {
        double max_z = sum(input);
        rowvec output  = exp(input + max_z);
        output = output / sum(output);
        return output;
    }

    rowvec apply_activation_function(rowvec input, int index) {
        if (index == n_hidden_layers){
            return softmax(input);
        }
        else{
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

    }

    colvec forward_propagation(rowvec input) {
        colvec neths;
        for(int i = 0; i<=n_hidden_layers;i++){
            input = apply_activation_function(input*weights(i) + bias(i).t(), i);
            neurons_outputs(i) = input;
        }
        return input.t();
    }

    void fill_deltas(colvec output){
        deltas(n_hidden_layers) = (neurons_outputs(n_hidden_layers) - output.t());//*(neurons_outputs(n_hidden_layers).t());
        // neurons_outputs(n_hidden_layers).print();
        // output.t().print();
        // (neurons_outputs(n_hidden_layers)-output.t()).print();
        // for(int i = n_hidden_layers-1; i >= 0; --i){
        //    deltas(i) = deltas(i+1) * weights(i+1).t();
        // }
    }

    void print_deltas(){
        for(int i = 0; i < n_hidden_layers+1; ++i) deltas(i).print();
    }

    void update_weights(){
        // TO DO
    }

    void propagate_backward(colvec output) {
        fill_deltas(output);
        print_deltas();
        update_weights();
    }

public:

    NeuralNetwork(int input_length, int n_hidden_layers, vector<int> neurons_per_layer, int output_length, string activation_function="sigm") {

        this->input_length = input_length;
        this->n_hidden_layers = n_hidden_layers;
        this->output_length = output_length;
        this->activation_function = activation_function;
        
        bias = field<colvec>(n_hidden_layers+1);
        neurons_outputs = field<rowvec>(n_hidden_layers+1);
        deltas = field<rowvec>(n_hidden_layers+1);
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

    double fit(field<rowvec> X_train, rowvec Y_train, double alpha, int epochs){
        return 0;
    }

    double predict(field<rowvec> X_test, rowvec Y_test){
        return 0;
    }

};




int main(){

    arma_rng::set_seed(42);
    vector<int> capas{3, 2, 3, 2};
    NeuralNetwork mlp(5, 4, capas, 7, "tanh");
    // for(int i=0; i<mlp.weights.n_elem;i++){
    //     cout<<"Weight"<<i<<":\n"<<mlp.weights(i);
    // }
    rowvec input(5, fill::randu);
    // input = softmax(input);
    // input.print();
    // auto x = mlp.forward_propagation(input);
    // // x.print("Resultado forward:");
    // mlp.propagate_backward(x*1.2324);

    return 0;
}