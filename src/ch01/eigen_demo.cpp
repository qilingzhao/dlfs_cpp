//
// Created by 赵其岭 on 2022/8/6.
//
#include "ch01/eigen_demo.hpp"

// Ref: http://zhaoxuhui.top/blog/2019/08/21/eigen-note-1.html

void create_matrix() {
    // 在Eigen中, 向量和矩阵都是 Matrix 模版类的对象

    // 创建矩阵需要Matrix<>构造, 前三个为必须参数，后三个为可选参数
    // 第一个参数为数据类型，后两个参数分别为矩阵的行数和列数
    Eigen::Matrix<double, 3, 4> mat_34d;
    std::cout << mat_34d << std::endl;

    // 当在编译器确定不了矩阵大小，可以使用 Eigen::Dynamic 来生成动态矩阵
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> mat_dyna_i;
    // 使用 resize() 确定矩阵大小, 这里的 resize() 对矩阵是"毁灭性"的, 如果只改变大小，需要使用 conservativeResize()
    mat_dyna_i.resize(2, 2);
    std::cout << mat_dyna_i << std::endl;

    // Eigen 对于小于等于4的矩阵创建，已经定义好类型了, 格式为 Matrix{row}{col}{type}
    // 下面就是定一个 2*2的 double 类型矩阵, 2代表了行数为2, 当行数和列数相同时省略, d代表 type，即double类型
    Eigen::Matrix2d mat_2d;
    mat_2d(1, 0) = 10;
    std::cout << mat_2d << std::endl;

    // 3 * 3 类型为 int 的矩阵
    Eigen::Matrix3i mat_3i;

    // 当 row 或者 col不确定的时候，使用X表示.
    // 2 * x 类型为 int 矩阵
    Eigen::Matrix2Xi mat_2xi;
}

void operate_matrix() {
    // 使用 逗号表达式 可为矩阵复制
    Eigen::Matrix2d mat_2d;
    mat_2d << 1, 2, 3, 4;
    std::cout << mat_2d << std::endl;

    // 可以是用 m(1, 2) 的方式获取和修改矩阵中的元素
    mat_2d(0, 0) = -1;
    std::cout << mat_2d << std::endl;

    // rows(), cols() 获取行数和列数
    std::cout << "rows: " << mat_2d.rows() << " cols: " << mat_2d.cols() << std::endl;

    // Eigen 重载了 + - * / 操作，但是 Eigen本身对数据敏感，不支持不同数据类型见的自动转化
    // + - 操作不支持矩阵和标量直接操作，如果需要整体 + 或者 - 某个数值，需要构建一个相同的矩阵。
    Eigen::Matrix2i mat_2i;
    mat_2i << 1, 2, 3, 4;
    int delta = 10;
    Eigen::Matrix2i mat_plus10 = mat_2i +
            Eigen::MatrixXi::Constant(mat_2i.rows(), mat_2i.cols(), delta);
    std::cout << mat_plus10 << std::endl;

    std::cout << mat_2i * 10 << std::endl;

    // 所有向量默认为列向量, 如果想用行向量需要使用 RowVector
    Eigen::Vector3d vec_3d(1, 2, 3);
    // 标量积即每个分量乘以相同数，结果还是一个向量，对应向量的缩放
    std::cout << "* result: " << vec_3d * 3 << std::endl;
    // 点积是两个向量每个分量对应相乘再相加，结果为标量，两个向量之间的夹角，以及在b向量在a向量方向上的投影
    std::cout << "dot result: " << vec_3d.dot(vec_3d) << std::endl;
    // 叉积是将两个向量按叉积公式展开相乘，结果仍然是一个向量，并且两个向量的叉积与这两个向量组成的坐标平面垂直
    std::cout << "cross result: " << vec_3d.cross(vec_3d) << std::endl;

    // colwise() 返回矩阵每列的值
    // rowwise() 返回矩阵每行的值
    // 可以用来实现 矩阵 和 向量的广播操作
    Eigen::Matrix<int, 2, 4> mat_24_i;
    mat_24_i << 1, 2,  3, 4,  5, 6,  7, 8;
    std::cout << mat_24_i << std::endl;
    Eigen::Vector2i vec2;
    vec2 << 10, 20;
    // 第一行 +10, 第二行 +20
    Eigen::Matrix<int, 2, 4> res1 = mat_24_i.colwise() + vec2;
    std::cout << res1 << std::endl;
}

void block() {
    //  在Eigen中有两种方式进行块操作，
    //  一种是动态尺寸块(dynamic-size block)：mat.block(i,j,p,q)，
    //  另一种是固定尺寸块(fixed-size block)：mat.block<p,q>(i,j)。
    //  它们在语义上等价，都表示从矩阵的i行j列开始取大小为p行q列的块
    Eigen::MatrixXd m = Eigen::MatrixXd::Random(5, 5);
    std::cout << m.block<3, 1>(0, 0) << std::endl;
    std::cout << m.block(0, 0, 3, 1) << std::endl;

    // row() col() 可取单独一行/列
    std::cout << m.row(0) << std::endl;
}

void eigen_map() {
    // Map 主要用于使用C++原生数组生成 Matrix
    // 需要3个模板参数分别是数据类型、行数、列数。而如果采用预定义类型，则直接传入即可，不需要模板参数了。
    // 此外Map的构造函数中还需要两个其它参数：指向数据的指针以及修改后矩阵的大小
    double data[8];
    for (int i = 0; i < 8; ++i) {
        data[i] = i;
    }
    Eigen::Map<Eigen::MatrixXd> md1(data, 2, 4);
    std::cout << md1 << std::endl;
    // 需要注意的是Map构造默认是按照列优先的，即按照列依次从上倒下、从左到右填充，
    // 若想换成行优先只需要在模板参数里加上RowMajor即可。
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> md2(data, 2, 4);
    std::cout << md2 << std::endl;
}

void test_eigen() {
    eigen_map();

}