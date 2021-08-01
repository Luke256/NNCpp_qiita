#pragma once
#include "bits/stdc++.h"

class Activation
{
public:
    std::vector<std::vector<long double>> last_data;

    std::vector<std::vector<long double>> sigmoid(std::vector<std::vector<long double>> &x);

    std::vector<std::vector<long double>> linear(std::vector<std::vector<long double>> &x);

    std::vector<std::vector<long double>> softmax(std::vector<std::vector<long double>> &x);

    std::vector<std::vector<long double>> relu(std::vector<std::vector<long double>> &x);

    std::vector<std::vector<long double>> relu_back(std::vector<std::vector<long double>> &x);

    std::vector<std::vector<long double>> sigmoid_back(std::vector<std::vector<long double>> &x);

    std::vector<std::vector<long double>> softmax_back(std::vector<std::vector<long double>> &x);

    std::string m_name;

    Activation();
    Activation(std::string name);
    std::vector<std::vector<long double>> forward(std::vector<std::vector<long double>> &x);

    std::vector<std::vector<long double>> backward(std::vector<std::vector<long double>> &x);
};