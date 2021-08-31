#pragma once
#include "bits/stdc++.h"

enum ActivationType{
    Sigmoid,
    Linear,
    SoftMax,
    Relu
};

class Activation
{
public:
    std::vector<std::vector<double>> last_data;

    std::vector<std::vector<double>> sigmoid(std::vector<std::vector<double>> &x);

    std::vector<std::vector<double>> linear(std::vector<std::vector<double>> &x);

    std::vector<std::vector<double>> softmax(std::vector<std::vector<double>> &x);

    std::vector<std::vector<double>> relu(std::vector<std::vector<double>> &x);

    std::vector<std::vector<double>> relu_back(std::vector<std::vector<double>> &x);

    std::vector<std::vector<double>> sigmoid_back(std::vector<std::vector<double>> &x);

    std::vector<std::vector<double>> softmax_back(std::vector<std::vector<double>> &x);

    ActivationType m_name;

    Activation();
    Activation(ActivationType name);
    std::vector<std::vector<double>> forward(std::vector<std::vector<double>> &x);

    std::vector<std::vector<double>> backward(std::vector<std::vector<double>> &x);
};