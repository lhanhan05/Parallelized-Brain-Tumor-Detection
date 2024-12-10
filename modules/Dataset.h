#include <string>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace Eigen;

class Dataset {
public:
    Dataset(vector<string> folder_paths, int batch_size);
    Tensor<int, 2> labels;
    Tensor<float, 4> images;

    void loadImages(vector<string> folder_paths);

    std::pair<Tensor<float, 4>, Tensor<int, 2>> get_batch(int batchIdx);

};