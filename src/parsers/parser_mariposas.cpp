#include <string>
#include <iostream>
#include <fstream>
#include <experimental/filesystem>
#include <armadillo>
#include <algorithm>
#include <random>

using namespace std;
using namespace arma;
namespace fs = std::experimental::filesystem;



class Parser_mariposas{

    static int get_number_of_features(string filename){
        int f = 0;
        string line;
        fstream file(filename, ios::in);
        while(getline(file, line)){
            f++;
        }
        file.clear();
        file.close();
        return f;
    }

    static rowvec get_feature_vector(string filename, int features){
        rowvec img = rowvec(features);
        string line;
        fstream file(filename, ios::in);
        int i = 0;
        while(getline(file, line)){
            img(i) = stod(line);
            ++i;
        }
        file.clear();
        file.close();
        return img;
    }

    static rowvec one_hot_encode(int category, int total){
        rowvec encoded_label(total,fill::zeros);
        encoded_label(category-1) = 1;
        return encoded_label;
    }
public:

    static map<string,field<rowvec>> get_data(int cortes = 6, int categories=10, float train_proportion=0.8,
    float validation_proportion=0.1, int label_length=3, bool pca=false, bool augmentation=false){
        string line, path = "../data/mariposas/"+ to_string(cortes);
        if (pca){
            path +="_cortes_pca/";
        }
        else if(augmentation){
              path +="_cortes_augmentation/";
        }
        else{
            path +="_cortes/";
        }
         
        map<string,field<rowvec>> data;
        vector<rowvec> x_train, x_validation, x_test, y_train,y_validation, y_test;
        float test_proportion = (1 - train_proportion) - validation_proportion;

        for(int it=1; it<=categories; it++){
            vector<string> file_names;
            int features, images = 0, i = 0, label;
            bool flag = true;
            fstream file;
            if(augmentation){
                file.open("../data/mariposas/paths_augmentation.csv", ios::in);
            }
            else{
                file.open("../data/mariposas/paths.csv", ios::in);
            }

            file.clear();
            file.seekg(0,ios::beg);
            while(getline(file,line)){
                if (flag){
                    flag = false;
                    features = get_number_of_features(path+line);
                }
                label = stoi(line.substr(0,label_length));
                if(label == it){
                    images++;
                    file_names.push_back(line);
                }
            }

            field<rowvec> images_features = field<rowvec>(images);
            field<rowvec> labels = field<rowvec>(images);

            int train_index = 0, validation_index = round(train_proportion*images), test_index = round((train_proportion + validation_proportion)*images);
            file.clear();
            file.seekg(0,ios::beg);

            auto rng = std::default_random_engine {};

            std::shuffle(std::begin(file_names), std::end(file_names), rng);

            i=0;

            for (auto line : file_names){
                rowvec img = get_feature_vector(path+line, features);
                images_features(i) = img;
                labels(i) = one_hot_encode(it, categories);
                ++i;
            }

            auto x_train_class = images_features.rows(0,validation_index - 1);
            auto x_validation_class = images_features.rows(validation_index, test_index - 1);
            auto x_test_class = images_features.rows(test_index, images-1);

            auto y_train_class = labels.rows(0,validation_index - 1);
            auto y_validation_class = labels.rows(validation_index, test_index - 1);
            auto y_test_class = labels.rows(test_index, images-1);

            for(int k =0; k<x_train_class.n_elem;k++){
                x_train.push_back(x_train_class(k));
            }

            for(int k =0; k<x_validation_class.n_elem;k++){
                x_validation.push_back(x_validation_class(k));
            }

            for(int k =0; k<x_test_class.n_elem;k++){
                x_test.push_back(x_test_class(k));
            }

            for(int k =0; k<y_train_class.n_elem;k++){
                y_train.push_back(y_train_class(k));
            }

            for(int k =0; k<y_validation_class.n_elem;k++){
                y_validation.push_back(y_validation_class(k));
            }

            for(int k =0; k<y_test_class.n_elem;k++){
                y_test.push_back(y_test_class(k));
            }
        }

        field<rowvec> temp_x_train(x_train.size());
        field<rowvec> temp_x_validation(x_validation.size());
        field<rowvec> temp_x_test(x_test.size());

        field<rowvec> temp_y_train(y_train.size());
        field<rowvec> temp_y_validation(y_validation.size());
        field<rowvec> temp_y_test(y_test.size());

        for(int pos = 0; pos < x_train.size(); pos++){
            temp_x_train(pos) = x_train[pos];
        }

        for(int pos = 0; pos < x_validation.size(); pos++){
            temp_x_validation(pos) = x_validation[pos];
        }

        for(int pos = 0; pos < x_test.size(); pos++){
            temp_x_test(pos) = x_test[pos];
        }

        for(int pos = 0; pos < y_train.size(); pos++){
            temp_y_train(pos) = y_train[pos];
        }

        for(int pos = 0; pos < y_validation.size(); pos++){
            temp_y_validation(pos) = y_validation[pos];
        }

        for(int pos = 0; pos < y_test.size(); pos++){
            temp_y_test(pos) = y_test[pos];
        }


        data["X_train"] = temp_x_train;
        data["X_validation"] = temp_x_validation;
        data["X_test"] = temp_x_test;

        data["y_train"] = temp_y_train;
        data["y_validation"] = temp_y_validation;
        data["y_test"] = temp_y_test;
        return data;
    }
};