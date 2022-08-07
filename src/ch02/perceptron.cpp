//
// Created by 赵其岭 on 2022/8/7.
//
#include "ch02/perceptron.hpp"

bool perceptron(int a, int b, double x1, double x2, double bias) {
    Eigen::Matrix<double, 1, 2> input;
    input << a, b;
    Eigen::Vector2d weight(x1, x2);
//    std::cout << input * weight << std::endl;
    return input.dot(weight) + bias > 0;
}

// [w1, w2] = [0.5, 0.5] bias = -0.7
bool and_neural(int a, int b) {
    return perceptron(a, b, 0.5, 0.5, -0.7);
}

bool nand_neural(int a, int b) {
    return perceptron(a, b, -0.5, -0.5, 0.7);
}

bool or_neural(int a, int b) {
    return perceptron(a, b, 0.5, 0.5, -0.2);
}

bool xor_neural(int a, int b) {
    bool s1 = nand_neural(a, b);
    bool s2 = or_neural(a, b);
    return and_neural(s1, s2);
}

void test_perceptron() {
    for (int a = 0; a <= 1; a++) {
        for (int b = 0; b <= 1; b++) {
            std::cout << "a: " << a << ", b=" << b << std::endl
            << "AND is " << and_neural(a, b) << std::endl
            << "NAND is " << nand_neural(a, b) << std::endl
            << "OR is " << or_neural(a, b) << std::endl
            << "XOR is " << xor_neural(a, b) << std::endl
            << "----------" << std::endl;
        }
    }
}