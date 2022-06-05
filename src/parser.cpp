#include <string>
#include <iostream>
#include <fstream>
#include <experimental/filesystem>

using namespace std;
namespace fs = std::experimental::filesystem;

int main() {
    std::string path = "data/3_cortes/";
    for (const auto & entry : fs::directory_iterator(path))
        std::cout << entry.path() << std::endl;
}