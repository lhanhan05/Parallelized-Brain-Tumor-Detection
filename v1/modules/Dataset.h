#include <string>
#include <vector>  // Include the vector header
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace Eigen;
using std::vector;
using std::string;

class Dataset {
public:
    Dataset(vector<string>& folder_paths, int batch_size);
    Tensor<int, 2> labels;
    Tensor<float, 4> images;

    void loadImages(vector<string>& folder_paths);
    void shuffleImages();

    std::pair<Tensor<float, 4>, Tensor<int, 2>> get_batch(int batchIdx);

private:
    std::vector<std::vector<unsigned char>> image_vector;  // Using 2D vector to hold image data
    int batchSize;
};