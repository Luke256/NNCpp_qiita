#include "Activation.hpp"

std::vector<std::vector<double>> Activation::sigmoid(std::vector<std::vector<double>> &x)
{
    std::vector<std::vector<double>> res;

    for (auto batch : x)
    {
        std::vector<double> t;
        for (auto i : batch)
        {
            t.push_back(1.0 / (1.0 + exp(-i)));
        }
        res.push_back(t);
    }

    last_data = res;

    return res;
}

std::vector<std::vector<double>> Activation::linear(std::vector<std::vector<double>> &x)
{
    std::vector<std::vector<double>> res;
    for (auto batch : x)
    {
        std::vector<double> t;
        for (double i : batch)
        {
            t.push_back(i);
        }
        res.push_back(t);
    }

    return res;
}

std::vector<std::vector<double>> Activation::softmax(std::vector<std::vector<double>> &x)
{
    std::vector<std::vector<double>> res;

    for (auto batch : x)
    {
        std::vector<double> t;
        double c = *max_element(batch.begin(), batch.end());
        double sum = 0;
        for (double i : batch)
        {
            sum += exp(i - c);
        }
        for (double i : batch)
        {
            t.push_back(exp(i - c) / sum);
        }
        res.push_back(t);
    }
    return res;
}

std::vector<std::vector<double>> Activation::relu(std::vector<std::vector<double>> &x)
{
    std::vector<std::vector<double>> res;

    for (auto batch : x)
    {
        std::vector<double> t;
        for (double i : batch)
        {
            t.push_back((i >= 0) * i);
        }
        res.push_back(t);
    }
    return res;
}

std::vector<std::vector<double>> Activation::relu_back(std::vector<std::vector<double>> &x)
{
    std::vector<std::vector<double>> res = x;

    // 対応する最後に受け取った入力が0未満なら0、そうでないのならxをそのまま
    for (int i = 0; i < x.size(); ++i)
    {
        for (int j = 0; j < x[i].size(); ++j)
        {
            if (last_data[i][j] < 0) res[i][j] = 0;
        }
    }

    return res;
}


std::vector<std::vector<double>> Activation::sigmoid_back(std::vector<std::vector<double>> &x){
    std::vector<std::vector<double>>res;

    for(int i = 0; i < x.size(); ++i){
        std::vector<double>t;
        for(int j = 0; j < x[i].size(); ++j){
            t.push_back(x[i][j] * (1 - last_data[i][j]) * last_data[i][j]);
        }
        res.push_back(t);
    }

    return res;
}

std::vector<std::vector<double>> Activation::softmax_back(std::vector<std::vector<double>> &x){
    return x;
}

Activation::Activation() {}
Activation::Activation(ActivationType name) : m_name(name) {}
std::vector<std::vector<double>> Activation::forward(std::vector<std::vector<double>> &x)
{
    last_data = x;
    if (m_name == ActivationType::Sigmoid) return sigmoid(x);
    else if (m_name == ActivationType::Linear) return linear(x);
    else if (m_name == ActivationType::SoftMax) return softmax(x);
    else if (m_name == ActivationType::Relu) return relu(x);
    else return x;
}

std::vector<std::vector<double>> Activation::backward(std::vector<std::vector<double>> &x)
{
    if(m_name == ActivationType::Sigmoid) return sigmoid_back(x);
    else if (m_name == ActivationType::SoftMax) return softmax_back(x);
    else if (m_name == ActivationType::Relu) return relu_back(x);
    else return x;
}
