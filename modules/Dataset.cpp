// // #include <opencv2/opencv.hpp>
// using namespace cv;

// class Dataset {
// public:
//     Dataset(const std::string& folder_path, int batch_size) {
//         load_images(folder_path);
//         this->batch_size = batch_size;
//     }

//     void loadImages(const std::string& folder_path) {
//         vector<String> filenames;
//         glob(folder_path, filenames, false);

//         int N = filenames.size();
//         labels = Tensor<int, 2>(N, 1);

//         for (const auto& filename : filenames) {
//             Mat img = imread(filename, IMREAD_COLOR);
//             if (!img.empty()) {
//                 cvtColor(img, img, COLOR_BGR2GRAY);
//                 resize(img, img, Size(32, 32));
//                 images.push_back(img);

//                 if (filename.find("G") == 0) {
//                     labels(i, 0) = 0;
//                 } else if (filename.find("M") == 0){
//                     labels(i, 0) = 1;
//                 } else if (filename.find("N") == 0){
//                     labels(i, 0) = 2;
//                 } else{ // P
//                     labels(i, 0) = 3;
//                 }
//             }
//         }
//     }

//     void convertImagesToTensor(const std::vector<cv::Mat>& images) {
//         int N = images.size();
//         int C = 1;
//         int H = images[0].rows;
//         int W = images[0].cols;
        
//         img_tensor(N, C, H, W);

//         for (int i = 0; i < N; ++i) {
//             Mat img = images[i];
//             img_tensor.convertTo(img, CV_32F, 1.0 / 255.0);

//             for (int h = 0; h < H; ++h) {
//                 for (int w = 0; w < W; ++w) {
//                     img_tensor(i, 0, h, w) = img.at<float>(h, w);
//                 }
//             }
//         }
//     }

// private:
//     std::vector<Mat> images;
//     int batch_size;

//     Tensor<float, 2> labels;
//     Tensor<float, 4> img_tensor;
// };



#define STB_IMAGE_IMPLEMENTATION
#include "../stb_image.h"
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>

class Dataset {
public:
    Dataset(const std::string& folder_path, int batch_size) {
        loadImages(folder_path);
        this->batch_size = batch_size;
    }

    void loadImages(const std::string& folder_path) {
        // Assuming you're using a C++17 or later standard for filesystem support
        std::vector<std::string> filenames;

        // Traverse the directory to find all files in the folder
        for (const auto& entry : std::filesystem::directory_iterator(folder_path)) {
            filenames.push_back(entry.path().string());
        }

        int N = filenames.size();
        labels = Tensor<int, 2>(N, 1);

        for (const auto& filename : filenames) {
            int width, height, channels;

            // Load the image using stb_image
            unsigned char* img_data = stbi_load(filename.c_str(), &width, &height, &channels, 0);
            if (img_data) {
                // Process the image (convert to grayscale, resize)
                // Resize is not directly supported by stb_image, but you can use other libraries or manually resize here
                // For grayscale conversion, you can manually calculate the average of R, G, B channels
                std::vector<unsigned char> grayscale_img(width * height);
                for (int i = 0; i < width * height; ++i) {
                    unsigned char r = img_data[i * channels];
                    unsigned char g = img_data[i * channels + 1];
                    unsigned char b = img_data[i * channels + 2];
                    grayscale_img[i] = static_cast<unsigned char>(0.299 * r + 0.587 * g + 0.114 * b);  // Simple RGB to grayscale conversion
                }

                // Store the grayscale image as a Mat object-like structure (e.g., in a 2D vector or Tensor)
                images.push_back(grayscale_img);

                // Set labels based on filename prefix (G, M, N, P)
                if (filename.find("G") == 0) {
                    labels(i, 0) = 0;
                } else if (filename.find("M") == 0) {
                    labels(i, 0) = 1;
                } else if (filename.find("N") == 0) {
                    labels(i, 0) = 2;
                } else { // P
                    labels(i, 0) = 3;
                }

                // Free the image memory after processing
                stbi_image_free(img_data);
            } else {
                std::cerr << "Failed to load image: " << filename << std::endl;
            }
        }
    }

    void convertImagesToTensor(const std::vector<std::vector<unsigned char>>& images) {
        int N = images.size();
        int C = 1;  // Grayscale images
        int H = 32; // Resized height
        int W = 32; // Resized width
        
        img_tensor(N, C, H, W);

        for (int i = 0; i < N; ++i) {
            const auto& img = images[i];
            
            // Convert to float and normalize
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    img_tensor(i, 0, h, w) = static_cast<float>(img[h * W + w]) / 255.0f;
                }
            }
        }
    }

private:
    std::vector<std::vector<unsigned char>> images;  // Using 2D vector to hold image data
    int batch_size;
    Tensor<int, 2> labels;
    Tensor<float, 4> img_tensor;
};
