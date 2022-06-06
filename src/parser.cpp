#include <string>
#include <iostream>
#include <fstream>
#include <experimental/filesystem>
#include <armadillo>

using namespace std;
using namespace arma;
namespace fs = std::experimental::filesystem;



class Parser{

    int get_number_of_features(string filename){
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

    rowvec get_feature_vector(string filename, int features){
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

public:

    static map<string,field<rowvec>> get_data(int cortes = 6){
        string line, path = "data/3_cortes/";
        int features, images = 0, i = 0;
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
        file.clear();
        file.seekg(0,ios::beg);
        while(getline(file,line)){
            rowvec img = get_feature_vector(path+line, features);
            images_features(i) = img;
            ++i;
        }
        cout<<images<<"\n"<<features<<"\n";
        file.close();
        }
}