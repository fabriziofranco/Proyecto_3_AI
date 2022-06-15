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

using namespace std;
using namespace arma;


struct activation_functions{

    static mat sigm(mat input){
        mat output = 1 / (1+exp(input * -1));
        return output;
    }

    static mat tanh(mat input){
        mat output = (2 / (1+exp(-input*2))) - 1;
        return output;
    }

    static mat relu(mat input){
        mat output = input;
        for(int i=0; i < size(output)[0]; i++)
            for(int j=0; j < size(output)[1]; j++)
                output(i,j) = max(output(i,j), output(i,j) * 0.1) ;
        return output;
    }

    static mat softmax(mat input) {
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
};