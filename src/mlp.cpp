#include <iostream>
#include <cmath>
#include <ctime>
#include <utility>
#include <vector>
#include <armadillo>
#include <string>
#include <algorithm>
#include <filesystem>
#include "parser.cpp"
using namespace std;
using namespace arma;


class NeuralNetwork {
    public:
        int input_length;
        int n_hidden_layers;
        int batch_size=64;
        int output_length;
        double scalar_rate=0.01;
        field<rowvec> bias;
        field<mat> neths, neurons_activated_outputs;
        field<mat> deltas;
        field<mat> weights, d_weights;
        string activation_function;

private:

    mat sigm(mat input){
        mat output = 1 / (1+exp(-input));
        return output;
    }

    mat tanh(mat input){
        mat output = (2 / (1+exp(-input*2))) - 1;
        return output;
    }

    mat relu(mat input){
        mat output = input;
        for(int i=0; i < size(output)[0]; i++)
            for(int j=0; j < size(output)[1]; j++)
                if (output(i,j) < 0) output(i,j) = 0;
        return output;
    }

    mat softmax(mat input) {
        double max_z;
        mat output=input;

        for(int i=0; i < input.n_rows; i++){
            max_z = sum(input.row(i));
            output.row(i)  = exp(input.row(i) + max_z);
            output.row(i) = output.row(i) / sum(output.row(i));
        }
        return output;
    }

    mat apply_activation_function(mat input, int index) {
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

    mat apply_derivated_activation_function(mat input) {
        if (this->activation_function == "sigm"){
            return sigm(input) % (1 - sigm(input));
        }
        else if(this->activation_function == "tanh"){
            mat output =  tanh(input);
            output = 1 -  output % output;
            return output;
        }
        else{
            mat output = input;
            for(int i=0; i < size(output)[0]; i++)
                for(int j=0; j < size(output)[1]; j++)
                    if (output(i,j) < 0) output(i,j) = 0;
                    else output(i,j) = 1;
            return output;
        }
    }

    void update_weights_and_biases(){
        for(int i = 0; i <= n_hidden_layers; ++i){
            weights(i) = weights(i) - d_weights(i)*scalar_rate;

            rowvec delta_rows(deltas(i).n_cols,fill::zeros);
            
            for(int j = 0; j < deltas(i).n_cols; j++){
                delta_rows(j) = sum(deltas(i).col(j))/ batch_size;
            }
            bias(i) = bias(i) - delta_rows * scalar_rate;
        }
    }


public:

    mat forward_propagation(mat input) {
        mat input_iter = input;
        for(int i = 0; i<=n_hidden_layers;i++){
            input_iter.print("Inp: ");
            weights(i).print("W: ");
            input_iter = input_iter * weights(i);
            input_iter.print("Inp: ");
            cout<<"------------------------------------------------------\n\n";
            for (int row = 0 ; row < input_iter.n_rows;row++){
                input_iter.row(row) = input_iter.row(row)  + bias(i);
            }
            neths(i) = input_iter;
            //input_iter = apply_activation_function(input_iter, i);
            neurons_activated_outputs(i) = input_iter;
        }
        input_iter.print("Inp: ");
        return input_iter;
    }

    void propagate_backward(mat input, mat y) {
        mat dLastLayer  = neurons_activated_outputs(n_hidden_layers) - y;


        deltas(n_hidden_layers) = dLastLayer;

        dLastLayer = neurons_activated_outputs(n_hidden_layers - 1).t() *  dLastLayer;
        // dLastLayer.print();
        d_weights(n_hidden_layers) = dLastLayer;
        
        for(int i = n_hidden_layers-1; i>=0;i--){
            dLastLayer = deltas(i+1)*weights(i+1).t();
            dLastLayer = dLastLayer % apply_derivated_activation_function(neurons_activated_outputs(i));
            deltas(i) = dLastLayer;
            if (i==0){
                dLastLayer = input.t() * dLastLayer;
            }
            else{
                dLastLayer = neurons_activated_outputs(i - 1).t() * dLastLayer;
            }
            d_weights(i) = dLastLayer;
        }
        update_weights_and_biases();
    }

    mat from_field_to_mat(field<rowvec> subset){
        mat mat_subset(subset.n_elem, subset(0).n_elem, fill::zeros);
        for(int i=0; i < mat_subset.n_rows;i++){
            mat_subset.row(i) = subset(i);
        }
        return mat_subset;
    }

public:

    NeuralNetwork(int input_length, int n_hidden_layers, vector<int> neurons_per_layer, int output_length, string activation_function="sigm", double scalar_rate=0.005) {

        this->input_length = input_length;
        this->n_hidden_layers = n_hidden_layers;
        this->output_length = output_length;
        this->activation_function = activation_function;
        this->scalar_rate = scalar_rate;
        
        bias = field<rowvec>(n_hidden_layers+1);
        neths = field<mat>(n_hidden_layers+1);
        neurons_activated_outputs = field<mat>(n_hidden_layers+1);
        d_weights = field<mat>(n_hidden_layers+1);
        deltas = field<mat>(n_hidden_layers+1);
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

    double fit(field<rowvec> X_train, field<rowvec> Y_train, field<rowvec> X_validation, field<rowvec> Y_validation, double learning_rate, int epochs){
        this-> batch_size = size(X_train)[0];
        this->scalar_rate = learning_rate;

        auto X_matrix = from_field_to_mat(X_train);
        auto Y_matrix = from_field_to_mat(Y_train);
        for(int i = 0; i <epochs;i++){
            auto output = forward_propagation(X_matrix);
            // propagate_backward(X_matrix, Y_matrix);
        } 
        return 0;
    }

    double predict(field<rowvec> X_test, field<rowvec> Y_test){
        auto X_matrix = from_field_to_mat(X_test);
        auto Y_matrix = from_field_to_mat(Y_test);
        forward_propagation(X_matrix).print();
        Y_matrix.print();
        return 0;
    }
};



int main(){
    arma_rng::set_seed(42);

    auto data = Parser::get_data(6, 10, 0.01, 0.1);
    auto X_train = data["X_train"]; auto y_train = data["y_train"];
    auto X_validation = data["X_validation"]; auto y_validation = data["y_validation"];
    auto X_test = data["X_test"]; auto y_test = data["y_test"];

    vector<int> capas{3, 2, 3, 2};
    NeuralNetwork mlp(200, 4, capas, 10, "relu");
    // mlp.weights.print();
    // mlp.forward_propagation(mat(5,4,fill::randu)).print();
    mlp.fit(X_train,y_train, X_validation, y_validation, 0.1, 1);
    // X_train.print();
    // mlp.d_weights.print();
    // mlp.predict(X_test,y_test);

    return 0;
}