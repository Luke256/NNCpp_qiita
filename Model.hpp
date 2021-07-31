#pragma once
#include <bits/stdc++.h>
#include "Layer.hpp"

class Model{
private:
    // mse : mean squared error
    // cen : mean cross-entropy error

    int m_input_size, m_output_size;
    std::vector<Layers::Dense>model;
    std::string m_loss;
    std::vector<std::vector<long double>> diff_error; // 相対誤差。maeとしても使えるけど、今回は二乗誤差の逆伝播(絶対値)、softmaxの逆伝播(そのまま)に使う

    // 損失関数がエントロピーなら最後のレイヤーの活性化関数はsoftmaxなので、何もしない
    // そうでない場合は一度損失関数を通して逆伝播させる

public:
    Model(int input_size);
    void AddDenseLayer(int unit, std::string activation);
    std::vector<std::vector<long double>> predict(const std::vector<std::vector<long double>>&data);
    void backward();
    std::vector<std::vector<long double>> numerical_gradient_layer(std::vector<std::vector<long double>>&batch_x, std::vector<std::vector<long double>>&batch_y, int x);
    std::vector<long double> numerical_gradient_bias(std::vector<std::vector<long double>>&batch_x, std::vector<std::vector<long double>>&batch_y, int x);
    std::vector<long double> fit(int step, long double learning_rate, std::vector<std::vector<long double>>&x, std::vector<std::vector<long double>>&y, int batch_size, std::string loss);
    long double caluculate_loss(std::vector<std::vector<long double>>&batch_x, std::vector<std::vector<long double>>&batch_y);
    void print();
};