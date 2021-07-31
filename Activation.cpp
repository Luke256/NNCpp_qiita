#include "Activation.hpp"

std::vector<std::vector<long double>> Activation::sigmoid(std::vector<std::vector<long double>> &x)
{
    std::vector<std::vector<long double>> res;

    for (auto batch : x)
    {
        std::vector<long double> t;
        for (auto i : batch)
        {
            t.push_back(1.0 / (1.0 + exp(-i)));
        }
        res.push_back(t);
    }

    last_data = res;

    return res;
}

std::vector<std::vector<long double>> Activation::leaner(std::vector<std::vector<long double>> &x)
{
    std::vector<std::vector<long double>> res;
    for (auto batch : x)
    {
        std::vector<long double> t;
        for (long double i : batch)
        {
            t.push_back(i);
        }
        res.push_back(t);
    }

    return res;
}

std::vector<std::vector<long double>> Activation::softmax(std::vector<std::vector<long double>> &x)
{
    std::vector<std::vector<long double>> res;

    for (auto batch : x)
    {
        std::vector<long double> t;
        long double c = *max_element(batch.begin(), batch.end());
        long double sum = 0;
        for (long double i : batch)
        {
            sum += exp(i - c);
        }
        for (long double i : batch)
        {
            t.push_back(exp(i - c) / sum);
        }
        res.push_back(t);
    }
    return res;
}

std::vector<std::vector<long double>> Activation::relu(std::vector<std::vector<long double>> &x)
{
    std::vector<std::vector<long double>> res;

    for (auto batch : x)
    {
        std::vector<long double> t;
        for (long double i : batch)
        {
            t.push_back((i >= 0) * i);
        }
        res.push_back(t);
    }
    return res;
}

std::vector<std::vector<long double>> Activation::relu_back(std::vector<std::vector<long double>> &x)
{
    std::vector<std::vector<long double>> res = x;

    // 対応する最後に受け取った入力がo未満なら0、そうでないのならxをそのまま
    for (int i = 0; i < x.size(); ++i)
    {
        for (int j = 0; j < x[i].size(); ++j)
        {
            if (last_data[i][j] < 0) res[i][j] = 0;
        }
    }

    return res;
}


std::vector<std::vector<long double>> Activation::sigmoid_back(std::vector<std::vector<long double>> &x){
    std::vector<std::vector<long double>>res;

    for(int i = 0; i < x.size(); ++i){
        std::vector<long double>t;
        for(int j = 0; j < x[i].size(); ++j){
            t.push_back(x[i][j] * (1 - last_data[i][j]) * x[i][j]);
        }
        res.push_back(t);
    }

    return res;
}

std::vector<std::vector<long double>> Activation::softmax_back(std::vector<std::vector<long double>> &x){
    return x;
}

Activation::Activation() {}
Activation::Activation(std::string name) : m_name(name) {}
std::vector<std::vector<long double>> Activation::forward(std::vector<std::vector<long double>> &x)
{
    last_data = x;
    if (m_name == "sigmoid") return sigmoid(x);
    else if (m_name == "leaner") return leaner(x);
    else if (m_name == "softmax") return softmax(x);
    else if (m_name == "relu") return relu(x);
    else return x;
}

std::vector<std::vector<long double>> Activation::backward(std::vector<std::vector<long double>> &x)
{
    if(m_name == "sigmoid") return sigmoid_back(x);
    else if(m_name == "softmax") return softmax_back(x);
    if (m_name == "relu")
        return relu_back(x);
    else
        return x;
}