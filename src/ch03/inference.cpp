//
// Created by 赵其岭 on 2022/8/9.
//
#include "ch03/inference.hpp"

void inference() {
    // MNIST_DATA_LOCATION set by MNIST cmake config
    std::cout << "MNIST data directory: " << MNIST_DATA_LOCATION << std::endl;

    // Load MNIST data
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
            mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);

    std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
    std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
    std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
    std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;

    int cnt = 0;
    for (auto it = dataset.test_images[0].begin(); it != dataset.test_images[0].end(); ++it) {
        std::cout << float(*it) / 256 << " ";
        cnt++;
        if (cnt % 28 == 0) {
            std::cout << std::endl;
        }
    }
}