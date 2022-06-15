#include <iostream>
#include <cmath>
#include <ctime>
#include <utility>
#include <vector>
#include <armadillo>
#include <string>
#include <algorithm>
#include <filesystem>
#include <fstream>

#include "../parsers/parser_senales.cpp"
#include "activation_functions.cpp"
#include "metrics.cpp"

namespace fs = std::experimental::filesystem;
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

    mat apply_activation_function(mat input, int index) {
        if (index == n_hidden_layers){
            return activation_functions::softmax(input);
        }
        else{
            if (this->activation_function == "sigm")
                return activation_functions::sigm(input);

            else if(this->activation_function == "tanh"){
                return activation_functions::tanh(input);
            }
            else{
                return activation_functions::relu(input);
            }
        }
    }

    mat apply_derivated_activation_function(mat input, int index) {
        if (this->activation_function == "sigm"){
            return activation_functions::sigm(input) % (1 - activation_functions::sigm(input));
        }
        else if(this->activation_function == "tanh"){
            mat output =  activation_functions::tanh(input);
            output = 1 -  output % output;
            return output;
        }
        else{
            mat output = input;
            for(int i=0; i < size(output)[0]; i++)
                for(int j=0; j < size(output)[1]; j++)
                    if (output(i,j) < 0) output(i,j) = 0.1;
                    else output(i,j) = 1;
            return output;
        }
    }

    void update_weights_and_biases(mat input){
        for(int i = 0; i <= n_hidden_layers; ++i){
            weights(i) = weights(i) - d_weights(i).t()*scalar_rate;

            rowvec delta_rows(deltas(i).n_cols,fill::zeros);
            
            for(int j = 0; j < deltas(i).n_cols; j++){
                delta_rows(j) = sum(deltas(i).col(j))/ batch_size;
            }
            bias(i) = bias(i) - delta_rows * scalar_rate;
        }
    }

    mat forward_propagation(mat input) {
        mat input_iter = input;
        for(int i = 0; i<=n_hidden_layers;i++){
            input_iter = input_iter * weights(i);

            for (int row = 0 ; row < input_iter.n_rows;row++){
                input_iter.row(row) = input_iter.row(row)  + bias(i);
            }
            neths(i) = input_iter;
            input_iter = apply_activation_function(input_iter, i);
            neurons_activated_outputs(i) = input_iter;
        }
        return input_iter;
    }

    void propagate_backward(mat input, mat y) {
        mat dLastLayer  = neurons_activated_outputs(n_hidden_layers) - y;

        deltas(n_hidden_layers) = dLastLayer;
        dLastLayer = dLastLayer.t() * neurons_activated_outputs(n_hidden_layers - 1);
        d_weights(n_hidden_layers) = dLastLayer;
        
        for(int i = n_hidden_layers-1; i>=0;i--){
            dLastLayer = deltas(i+1)*weights(i+1).t();
            dLastLayer = dLastLayer % apply_derivated_activation_function(neurons_activated_outputs(i), i);
            deltas(i) = dLastLayer;
            if (i==0){
                dLastLayer = dLastLayer.t() * input;
            }
            else{
                dLastLayer =  dLastLayer.t() * neurons_activated_outputs(i - 1);
            }
            d_weights(i) = dLastLayer;
        }

        update_weights_and_biases(input);
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
                weights(i) = mat(input_length,neurons_per_layer[i]);
                bias(i) = zeros(1,neurons_per_layer[i]);
            }
            else if(i<n_hidden_layers){
                weights(i) = mat(neurons_per_layer[i-1], neurons_per_layer[i]);
                bias(i) = zeros(1,neurons_per_layer[i]);
            }
            else{
                weights(i) = mat(neurons_per_layer[i-1], output_length);
                bias(i) = zeros(1,output_length);
            }    
        }

        if(activation_function =="sigm" or activation_function =="tanh"){
            this->weights = utils::xavier_normal_distribution(this->weights);
        }
        else{
            this->weights = utils::he_et_normal_distribution(this->weights);
        }

    }


    double fit(field<rowvec> X_train, field<rowvec> Y_train, field<rowvec> X_validation,
                field<rowvec> Y_validation, double learning_rate, double error=0.1,
                double max_iter=1e5, string mode="gd", string name_output_file="model_1"){

        this-> batch_size = size(X_train)[0];
        this->scalar_rate = learning_rate;

        mat output, output_validation;

        mat X_matrix = utils::from_field_to_mat(X_train);
        mat Y_matrix = utils::from_field_to_mat(Y_train);
        mat X_matrix_validation = utils::from_field_to_mat(X_validation);
        mat Y_matrix_validation = utils::from_field_to_mat(Y_validation);

        field<mat> best_weights = this->weights;
        field<rowvec> best_bias  = this->bias;
        
        string path = "../resultados/experimentos/"+ name_output_file;
        fs::create_directories(path);
        path += "/";


        if (mode=="gd"){
            int iter = 0; double iter_error = 1;
            double best_error = 1e10;

            while(iter_error>error and iter<max_iter){
                output_validation = forward_propagation(X_matrix_validation);
                output = forward_propagation(X_matrix);
                iter_error = metrics::save_error_and_acc(output,Y_matrix,output_validation,Y_matrix_validation, path + "error_and_accuracy.csv");

                if(iter_error < best_error){
                    best_weights = this->weights;
                    best_bias  = this->bias;
                    best_error = iter_error;
                }

                if(iter_error<=error){
                    break;
                }

                propagate_backward(X_matrix, Y_matrix);
                iter++;
            }

            this->weights = best_weights;
            this->bias = best_bias;
            
            this->weights.save(path + "weights.bin");
            this->bias.save(path + "bias.bin");


            output_validation = forward_propagation(X_matrix_validation);
            output = forward_propagation(X_matrix);
            metrics::print_matriz_confusion(output,Y_matrix,"TRAIN");
            metrics::print_matriz_confusion(output_validation,Y_matrix_validation,"VALIDATION");

            cout.precision(3);
            cout.setf(ios::fixed);
            cout<<"Train acc: ";
            metrics::print_accuracy(output,Y_matrix);
            cout<<"Validation acc: ";
            metrics::print_accuracy(output_validation,Y_matrix_validation);
            return 0;
        }

        else{
            // double error= 1;
            // while(error>=epochs){
            //     int pos  = round(generate_random_number(0,X_matrix.n_rows-1));
            //     output = forward_propagation(X_matrix.row(pos));
            //     output.print();
            //     propagate_backward(X_matrix.row(pos), Y_matrix.row(pos));

            //     output_validation = forward_propagation(X_matrix_validation);
            //     output = forward_propagation(X_matrix);
            //     error = metrics::save_error_and_acc(output,Y_matrix,output_validation,Y_matrix_validation,"prueba4.csv");
            //     metrics::print_matriz_confusion(output,Y_matrix,"TRAIN");
            //     metrics::print_matriz_confusion(output_validation,Y_matrix_validation,"VALIDATION");
            // }


            // cout.precision(3); cout.setf(ios::fixed);
            // cout<<"Fit acc: ";
            // metrics::print_accuracy(forward_propagation(X_matrix),Y_matrix);
            return 0;
        }
    }

    double predict(field<rowvec> X_test, field<rowvec> Y_test,string label="Test"){
        auto X_matrix = utils::from_field_to_mat(X_test);
        auto Y_matrix = utils::from_field_to_mat(Y_test);
        mat output = forward_propagation(X_matrix);
        cout.precision(3);
        cout.setf(ios::fixed);
        cout<<label+" acc: ";
        metrics::print_accuracy(output,Y_matrix);
        return 0;
    }

    void load_model(string name_model){
        string path = "../resultados/experimentos/"+ name_model+"/";
        this->weights.load(path + "weights.bin");
        this->bias.load(path + "bias.bin");
    }
};



int main(){
    arma_rng::set_seed(42);

    auto data = Parser_senales::get_data(3, 10, 0.8, 0.1, 5, false);
    auto X_train = data["X_train"]; auto y_train = data["y_train"];
    auto X_validation = data["X_validation"]; auto y_validation = data["y_validation"];
    auto X_test = data["X_test"]; auto y_test = data["y_test"];

    vector<int> capas{ 160, 80, 30};
    NeuralNetwork mlp(480, 3, capas, 10, "tanh");
    // mlp.fit(X_train,y_train, X_validation, y_validation, 1e-5, 0.15, 25000, "gd", "tanh_3_capas_normal");
    mlp.load_model("tanh_3_capas_normal");

    mlp.predict(X_train,y_train,"Train");
    mlp.predict(X_validation,y_validation,"Validation");
    mlp.predict(X_test,y_test,"Test");

    return 0;
}