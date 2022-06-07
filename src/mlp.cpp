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
        double scalar_rate;
        field<rowvec> bias;
        field<mat> neths, neurons_activated_outputs;
        field<mat> deltas;
        field<mat> weights, d_weights;
        string activation_function;

private:


    double generate_random_number(double min, double max){
        return min + (double) (rand()) / ((double)(RAND_MAX/(max-min)));
    }

    mat fill_mat_with_randoms(mat m){
        for(int i=0; i < size(m)[0]; i++)
            for(int j=0; j < size(m)[1]; j++)
                m(i,j) = generate_random_number(-5, 5);
        return m;
    }

    mat sigm(mat input){
        mat output = 1 / (1+exp(input * -1));
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

        for(int i=0; i < output.n_rows; i++){
            output.row(i) = input.row(i) - max(input.row(i));
            output.row(i) = exp(output.row(i));
            double suma = sum(output.row(i));
            output.row(i)  = output.row(i) / suma;
        }
        return output;
    }

    mat apply_activation_function(mat input, int index) {
        if (index == n_hidden_layers){
            cout.precision(3);
            cout.setf(ios::fixed);
            softmax(input).raw_print(cout,"Softmax:");
            return input;
        }
        else{
            if(index%2==0){
                return relu(input);
            }
            else{
                if (this->activation_function == "sigm") return sigm(input);
                else return tanh(input);
            }

        }
    }

    mat apply_derivated_activation_function(mat input, int index) {
        if(index%2 == 0){
            mat output = input;
            for(int i=0; i < size(output)[0]; i++)
                for(int j=0; j < size(output)[1]; j++)
                    if (output(i,j) < 0) output(i,j) = 0;
                    else output(i,j) = 1;
            return output;
        }
        else{
            if (this->activation_function == "sigm"){
                return sigm(input) % (1 - sigm(input));
            }
            else {
                mat output =  tanh(input);
                output = 1 -  output % output;
                return output;
            }
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
            input_iter = input_iter * weights(i);

            // for (int row = 0 ; row < input_iter.n_rows;row++){
            //     input_iter.row(row) = input_iter.row(row)  + bias(i);
            // }
            neths(i) = input_iter;
            input_iter = apply_activation_function(input_iter, i);
            neurons_activated_outputs(i) = input_iter;
        }

        // input_iter.raw_print(cout,"Inp: ");
        return input_iter;
    }


    void propagate_backward(mat input, mat y) {
        mat dLastLayer  = neurons_activated_outputs(n_hidden_layers) - y;

        deltas(n_hidden_layers) = dLastLayer;
        dLastLayer = neurons_activated_outputs(n_hidden_layers - 1).t() *  dLastLayer;
        d_weights(n_hidden_layers) = dLastLayer;
        
        for(int i = n_hidden_layers-1; i>=0;i--){
            dLastLayer = deltas(i+1)*weights(i+1).t();
            dLastLayer = dLastLayer % apply_derivated_activation_function(neurons_activated_outputs(i), i);
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

        // this->weights.print();

    }

    mat from_field_to_mat(field<rowvec> subset){
        mat mat_subset(subset.n_elem, subset(0).n_elem, fill::zeros);
        for(int i=0; i < mat_subset.n_rows;i++){
            mat_subset.row(i) = subset(i);
        }
        return mat_subset;
    }


public:

    NeuralNetwork(int input_length, int n_hidden_layers, vector<int> neurons_per_layer, int output_length, string activation_function="sigm") {

        this->input_length = input_length;
        this->n_hidden_layers = n_hidden_layers;
        this->output_length = output_length;
        this->activation_function = activation_function;
        
        bias = field<rowvec>(n_hidden_layers+1);
        neths = field<mat>(n_hidden_layers+1);
        neurons_activated_outputs = field<mat>(n_hidden_layers+1);
        d_weights = field<mat>(n_hidden_layers+1);
        deltas = field<mat>(n_hidden_layers+1);
        weights = field<mat>(n_hidden_layers+1);
        

        for (int i = 0; i <= n_hidden_layers; i++){
            if(i==0){       
                weights(i) = randu(input_length,neurons_per_layer[i]);
                weights(i) = fill_mat_with_randoms(weights(i));
                bias(i) = randu(1,neurons_per_layer[i]);

            }
            else if(i<n_hidden_layers){
                weights(i) = mat(neurons_per_layer[i-1], neurons_per_layer[i],fill::randu);
                weights(i) = fill_mat_with_randoms(weights(i));
                bias(i) = randu(1,neurons_per_layer[i]);
            }
            else{
                weights(i) = randu(neurons_per_layer[i-1], output_length);
                weights(i) = fill_mat_with_randoms(weights(i));
                bias(i) = randu(1,output_length);

            }    
        }
    }

    double fit(field<rowvec> X_train, field<rowvec> Y_train, field<rowvec> X_validation, field<rowvec> Y_validation, double learning_rate, int epochs){
        this-> batch_size = size(X_train)[0];
        this->scalar_rate = learning_rate;
        mat output;
        auto X_matrix = from_field_to_mat(X_train);
        auto Y_matrix = from_field_to_mat(Y_train);
        for(int i = 0; i <epochs;i++){
            output = forward_propagation(X_matrix);
            propagate_backward(X_matrix, Y_matrix);
        }
        // cout.precision(3);
        // cout.setf(ios::fixed);
        // output.raw_print(cout,"Output: "); 
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
    auto data = Parser::get_data(7, 10, 0.01, 0.1);
    auto X_train = data["X_train"]; auto y_train = data["y_train"];
    auto X_validation = data["X_validation"]; auto y_validation = data["y_validation"];
    auto X_test = data["X_test"]; auto y_test = data["y_test"];
    // for (auto it: y_train) cout<<it<<endl;
    vector<int> capas{80,60,40,20};
    NeuralNetwork mlp(100, 4, capas, 10, "tanh");
    // mlp.weights(3).print();
    // mlp.forward_propagation(mat(10,10,fill::randu)).print();
    mlp.fit(X_train,y_train, X_validation, y_validation, 0.001, 3);
    // mlp.d_weights.print();
    // randn(2,3).print();
    // X_train.print();
    // mlp.d_weights.print();
    // mlp.predict(X_test,y_test);*/
    //for(int i=0;i<10;++i)cout<<generate_random_number(-0.5,0.5)<<"\n";
    // mat x(5,4, fill::zeros);
    // mat y(6,3, fill::zeros);
    // mat z(2,7, fill::zeros);
    // fill_mat_with_randoms(x).print("x:");
    // fill_mat_with_randoms(y).print("y:");
    // fill_mat_with_randoms(z).print("z:");
    return 0;
}