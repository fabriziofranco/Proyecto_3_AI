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



class Parser{

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

    static map<string,field<rowvec>> get_data(int cortes = 6, int categories=10, float train_proportion=0.8, float validation_proportion=0.1){
        string line, path = "data/"+ to_string(cortes) +"_cortes/";
        map<string,field<rowvec>> data;

        int features, images = 0, i = 0, label;
        bool flag = true;
        fstream file("data/paths.csv", ios::in);
        while(getline(file,line)){
            if (flag){
                flag = false;
                features = get_number_of_features(path+line);
            }
            images++;
        }
        field<rowvec> images_features = field<rowvec>(images);
        field<rowvec> labels = field<rowvec>(images);
        vector<string> file_names(images);

        float test_proportion = (1 - train_proportion) - validation_proportion;
        int train_index = 0, validation_index = round(train_proportion*images), test_index = round((train_proportion + validation_proportion)*images);
        file.clear();
        file.seekg(0,ios::beg);

        while(getline(file,line)){
            file_names[i] = line;
            ++i;
        } 

        file.clear();
        file.seekg(0,ios::beg);
        i=0;

        auto rng = std::default_random_engine {};

        std::shuffle(std::begin(file_names), std::end(file_names), rng);

        for (auto line : file_names){
            label = stoi(line.substr(0,3));
            rowvec img = get_feature_vector(path+line, features);
            images_features(i) = img;
            labels(i) = one_hot_encode(label, categories);
            ++i;
        }


        data["X_train"] = images_features.rows(0,validation_index - 1);
        data["X_validation"] = images_features.rows(validation_index, test_index - 1);
        data["X_test"] = images_features.rows(test_index, images-1);

        data["y_train"] = labels.rows(0,validation_index - 1);
        data["y_validation"] = labels.rows(validation_index, test_index - 1);
        data["y_test"] = labels.rows(test_index, images-1);

        file.close();

        return data;
    }
};