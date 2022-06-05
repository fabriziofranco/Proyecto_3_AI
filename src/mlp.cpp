#include <iostream>
#include <cmath>
#include <ctime>
#include <utility>
#include <vector>
#include <armadillo>
#include <string>
#include <algorithm>
#include <filesystem>

using namespace std;
using namespace arma;


class NeuralNetwork {
    public:
        int input_length;
        int n_hidden_layers;
        int output_length;
        double scalar_rate;
        field<rowvec> bias;
        field<rowvec> neths, neurons_activated_outputs;
        field<rowvec> deltas;
        field<mat> weights, d_weights;
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

    rowvec apply_derivated_activation_function(rowvec input) {
        if (this->activation_function == "sigm"){
            return sigm(input) * (1- sigm(input));
        }
        else if(this->activation_function == "tanh"){
            rowvec output =  tanh(input);
            for(int i=0; i < output.n_elem; i++)
                output(i) = 1 - output(i) * output(i);
            return output;

        }
        else{
            rowvec output = input;
            for(int i=0; i < output.n_elem; i++)
                if (output(i) > 0)
                    output(i) = 1;
                else
                    output(i) = 0;
            return output;
        }
    }

public:
    rowvec forward_propagation(rowvec input) {
        for(int i = 0; i<=n_hidden_layers;i++){
            input = input * weights(i) + bias(i);
            cout<<size(input);
            neths(i) = input;
            input = apply_activation_function(input, i);
            neurons_activated_outputs(i) = input;
        }
        return input;
    }

    void update_weights(){
        for(int i = 0; i < size(weights); ++i){
            weights(i) = weights(i) - d_weights(i)*scalar_rate;
        }
    }

    void update_biases(){
        for(int i = 0; i < size(bias); ++i){
            bias(i) = bias(i) - deltas(i)*scalar_rate;
        }
    }

    void propagate_backward(rowvec input, rowvec y) {

        mat dLastLayer  = neurons_activated_outputs(n_hidden_layers) - y;
        deltas(n_hidden_layers) = dLastLayer;
        dLastLayer = neurons_activated_outputs(n_hidden_layers - 1).t() *  dLastLayer;
        d_weights(n_hidden_layers) = dLastLayer;
        
        for(int i = n_hidden_layers-1; i>=0;i--){
            dLastLayer = deltas(i+1)*weights(i+1).t();
            deltas(i) = dLastLayer % apply_derivated_activation_function(neurons_activated_outputs(i));
            if (i==0){
                dLastLayer = input.t() *dLastLayer;
            }
            else{
                dLastLayer = neurons_activated_outputs(i - 1).t() *dLastLayer;
            }
            d_weights(i) = dLastLayer;
        }
        update_weights();
        update_biases();
    }

    mat one_hot_encoder(int category){
        mat encoder(output_length,fill::zeros);
        encoder(category-1) = 1;
        return encoder;
    }

public:

    NeuralNetwork(int input_length, int n_hidden_layers, vector<int> neurons_per_layer, int output_length, string activation_function="sigm", double scalar_rate=0.005) {

        this->input_length = input_length;
        this->n_hidden_layers = n_hidden_layers;
        this->output_length = output_length;
        this->activation_function = activation_function;
        this->scalar_rate = scalar_rate;
        
        bias = field<rowvec>(n_hidden_layers+1);
        neths = field<rowvec>(n_hidden_layers+1);
        neurons_activated_outputs = field<rowvec>(n_hidden_layers+1);
        d_weights = field<mat>(n_hidden_layers+1);
        deltas = field<rowvec>(n_hidden_layers+1);
        weights = field<mat>(n_hidden_layers+1);
        

        for (int i = 0; i <= n_hidden_layers; i++){
            if(i==0){
                weights(i) = mat(input_length, neurons_per_layer[i],fill::randu);
                bias(i) = rowvec(neurons_per_layer[i], fill::randu);
            }
            else if(i<n_hidden_layers){
                weights(i) = mat(neurons_per_layer[i-1], neurons_per_layer[i],fill::randu);
                bias(i) = rowvec(neurons_per_layer[i], fill::randu);
            }
            else{
                weights(i) = mat(neurons_per_layer[i-1], output_length,fill::randu);
                bias(i) = rowvec(output_length, fill::randu);
            }    
        }
    }

    double fit(field<rowvec> X_train, rowvec Y_train, double alpha, int epochs){
        this->scalar_rate = alpha;
        for(int i = 0; i <epochs;i++){
           for(int i = 0; i < size(X_train); ++i){
            auto output = forward_propagation(X_train[i]);
            propagate_backward(X_train[i], Y_train[i]);
        } 
        }
        return 0;
    }

    double predict(field<rowvec> X_test, rowvec Y_test){
        auto Y_test_one_hot_encoded = Y_
        field<rowvec> Y_pred = field<rowvec>(size(Y_test));
        for(int i = 0; i < size(X_test); ++i){
            Y_pred[i] = forward_propagation(X_test[i]);
        }
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
    auto x = mlp.forward_propagation(input);
    x.print("Resultado forward:");
    mlp.propagate_backward(input, x*1.2324);

    for(int i = 0; i < mlp.deltas.n_elem;i++)
        cout<<size(mlp.deltas[i])<<" ";
    cout<<endl;


    for(int i = 0; i < mlp.weights.n_elem;i++)
        cout<<size(mlp.weights[i])<<" ";
    cout<<endl;

    for(int i = 0; i < mlp.d_weights.n_elem;i++)
        cout<<size(mlp.d_weights[i])<<" ";
    cout<<endl;
    return 0;
}