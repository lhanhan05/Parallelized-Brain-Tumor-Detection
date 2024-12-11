#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "models/Conv1dModel.h"
#include "modules/Dataset.h"

using namespace std;
using namespace Eigen;

int main() {
    vector<string> data_folders;
    data_folders.push_back("./Data/glioma");
    data_folders.push_back("./Data/meningioma");
    data_folders.push_back("./Data/normal");
    data_folders.push_back("./Data/pituitary");

    int batch_size = 32;
    int epochs = 10;
    float learning_rate = 0.001;
    float momentum_coeff = 0.9;

    Conv1dModel model;
    Dataset dataset(data_folders, batch_size);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        float epochLoss = 0.0;
        int correctCount = 0;
        int totalPreds = 0;

        for (int batch_idx = 0; batch_idx < dataset.images.size()*4/5; batch_idx++) {
            auto [batch_images, batch_labels] = dataset.get_batch(batch_idx);

            auto [loss, predictions] = model.forward(batch_images, batch_labels);
            epochLoss += loss;

            for (int i = 0; i < predictions.dimension(0); ++i) {
                if (predictions(i) == batch_labels(i)) {
                    correctCount += 1;
                }
            }
            totalPreds += predictions.dimension(0);

            model.backward();
            model.update(learning_rate, momentum_coeff);
        }

        float accuracy = static_cast<float>(correctCount) / totalPreds;
        std::cout << "Epoch " << epoch + 1 << " - Training Loss: " << epochLoss 
                  << ", Accuracy: " << accuracy  << std::endl;
    }

    return 0;
}