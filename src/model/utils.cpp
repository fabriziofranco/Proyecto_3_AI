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


struct utils{

    static field<mat> xavier_normal_distribution(field<mat> empty_weights){
        double size = empty_weights.n_elem;
        double fan_in = 0; double fan_out = 0;
        double n_rows = 0, n_cols = 0;
        double sd = 0; double mu = 0;
        for (int i = 0; i < size;i++){
            fan_in = empty_weights(i).n_rows;
            fan_out = empty_weights(i).n_cols;
            sd = sqrt(2/(fan_in+fan_out));
            empty_weights(i) = randn( fan_in, fan_out, distr_param(mu,sd));
        }
        return empty_weights;
    }

    static field<mat> he_et_normal_distribution(field<mat> empty_weights){
        return empty_weights;
    }

    static mat from_field_to_mat(field<rowvec> subset){
        mat mat_subset(subset.n_elem, subset(0).n_elem, fill::zeros);
        for(int i=0; i < mat_subset.n_rows;i++){
            mat_subset.row(i) = subset(i);
        }
        return mat_subset;
    }

    static bool equal_matrix(mat matrix1, mat matrix2){
        rowvec row_1 = matrix1.as_row();
        rowvec row_2 = matrix2.as_row();

        for(int i=0;i<row_1.n_elem;i++){
            if (row_1(i) != row_2(i))
                return false;
        }
        return true;
    }

    static rowvec get_log(rowvec _row){
        for(int i=0;i<_row.n_elem;i++){
            _row(i) = log10(_row(i));
        }
        return _row;
    }

};