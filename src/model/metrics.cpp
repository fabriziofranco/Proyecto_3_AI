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
#include <iomanip>
#include "utils.cpp"
using namespace std;
using namespace arma;


struct metrics{

    static void print_accuracy(mat output, mat Y_real){
        double total  = output.n_rows;
        double correctas = 0;

        for (int i=0; i<output.n_rows;i++){
                if (output.row(i).index_max() == Y_real.row(i).index_max()) 
                    correctas++;
        }
        cout<<correctas/total<<endl;
    }

    static void print_matriz_confusion(mat output, mat Y_real, string name){
        double total  = output.n_rows;
        mat resultados(output.n_cols,output.n_cols,fill::zeros);
        
        for (int i=0; i<output.n_rows;i++){
            int a = output.row(i).index_max();
            int b = Y_real.row(i).index_max(); 
            resultados(a,b) += 1;
        }


        cout.precision(0);
        cout.setf(ios::fixed);
        cout << setw(5);

        resultados.raw_print(cout,"Resultados "+ name+": ");
    }


    static double save_error_and_acc(mat output,mat Y_matrix, mat output_validation, mat Y_matrix_validation, string filename){

        std::ofstream outfile;

        outfile.open(filename, std::ios_base::app); // append instead of overwrite

        int size = output.n_rows;
        double cross_entropy = 0;

        for(int i =0;i<size;i++){
            double cross_iter = - sum(Y_matrix.row(i) % utils::get_log(output.row(i)));
            cross_entropy += cross_iter/size;
        }

        double error_train = cross_entropy;
        outfile <<cross_entropy<<", "; 

        size = output_validation.n_rows;
        cross_entropy = 0;

        for(int i =0;i<size;i++){
            double cross_iter = - sum(Y_matrix_validation.row(i) % utils::get_log(output_validation.row(i)));
            cross_entropy += cross_iter/size;
        }
        outfile <<cross_entropy<<", "; 


        double total  = output.n_rows;
        double correctas = 0;

        for (int i=0; i<output.n_rows;i++){
                if (output.row(i).index_max() == Y_matrix.row(i).index_max()) 
                    correctas++;
        }
        outfile <<correctas/total<<", ";

        total  = output_validation.n_rows;
        correctas = 0;

        for (int i=0; i<output_validation.n_rows;i++){
                if (output_validation.row(i).index_max() == Y_matrix_validation.row(i).index_max()) 
                    correctas++;
        }
        outfile <<correctas/total<<"\n";

        return error_train;
    }



};