#define STB_IMAGE_IMPLEMENTATION
#include "../stb_image.h"
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace Eigen;
using namespace std;

class Dataset {
public:
    Tensor<int, 2> labels;
    Tensor<float, 4> images;
    Dataset(vector<string>& folder_paths, int batchSize) {
        loadImages(folder_paths);
        shuffle_images();
        this->batchSize = batchSize;
    }

    void loadImages(vector<string>& folder_paths) {
        vector<string> filenames;

        for(auto folder_path : folder_paths){
            for (const auto& entry : filesystem::directory_iterator(folder_path)) {
                filenames.push_back(entry.path().string());
            }
        }
        
        int width, height, channels;
        unsigned char* img_data = stbi_load("./Data/glioma/G_1.jpg", &width, &height, &channels, 0);
        images = Tensor<float, 4>(filenames.size(), 3, width, height);
        labels = Tensor<int, 2>(filenames.size(), 1);
        int idx = 0;
        for (const auto& filename : filenames) {

            unsigned char* img_data = stbi_load(filename.c_str(), &width, &height, &channels, 0);
            if (img_data) {
                vector<unsigned char> grayscale_img(width * height);
                for (int i = 0; i < width * height; ++i) {
                    unsigned char r = img_data[i * channels];
                    unsigned char g = img_data[i * channels + 1];
                    unsigned char b = img_data[i * channels + 2];
                    int y = i/width;
                    int x = i%width;
                    images(idx, 0, x, y) = r;
                    images(idx, 1, x, y) = g;
                    images(idx, 2, x, y) = b;
                }


                if (filename.find("G") == 0) {
                    labels(idx, 0) = 0;
                } else if (filename.find("M") == 0) {
                    labels(idx, 0) = 1;
                } else if (filename.find("N") == 0) {
                    labels(idx, 0) = 2;
                } else { // P
                    labels(idx, 0) = 3;
                }
                stbi_image_free(img_data);
                idx+=1;
            } else {
                cerr << "Failed to load image: " << filename << endl;
            }
        }
    }

    std::pair<Tensor<float, 4>, Tensor<int, 2>> get_batch(int batchIdx) {
        int startIdx = batchIdx * batchSize;
        int endIdx = min(startIdx + batchSize, static_cast<int>(images.dimension(0)));
        int currBatchSize = endIdx - startIdx;

        Tensor<float, 4> batch_images(currBatchSize, images.dimension(1), images.dimension(2), images.dimension(3));
        Tensor<int, 2> batch_labels(currBatchSize, labels.dimension(1));

        for (int i = 0; i < currBatchSize; ++i) {
            batch_images.chip(i, 0) = images.chip(startIdx + i, 0);
            batch_labels.chip(i, 0) = labels.chip(startIdx + i, 0);
        }

        return {batch_images, batch_labels};
    }

    void shuffle_images() {
        int num_samples = images.dimension(0);

        vector<int> indices(num_samples);
        for (int i = 0; i < num_samples; ++i) {
            indices[i] = i;
        }

        random_device rd;
        mt19937 gen(rd());
        shuffle(indices.begin(), indices.end(), gen);

        Tensor<float, 4> shuffled_images(images.dimensions());
        Tensor<int, 2> shuffled_labels(labels.dimensions());

        for (int i = 0; i < num_samples; ++i) {
            shuffled_images.chip(i, 0) = images.chip(indices[i], 0);
            shuffled_labels.chip(i, 0) = labels.chip(indices[i], 0);
        }

        images = shuffled_images;
        labels = shuffled_labels;
    }

private:
    std::vector<std::vector<unsigned char>> image_vector;  // Using 2D vector to hold image data
    int batchSize;
};
