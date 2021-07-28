#pragma once
#include "bits/stdc++.h"
#include "Activation.hpp"

namespace Layers{
    class Dense{
    // private:
    public:
        Activation::Activation m_activation;
        std::vector<long double>bias;
        std::vector<std::vector<long double>> neuron; // neuron[i][j]:=the weight from previous j to current i.
        std::vector<std::vector<long double>> last_data;

        std::vector<std::vector<long double>>grad_layer;
        std::vector<long double>grad_bias;


        Dense(int previous_unit, int unit, std::string activation);
        std::vector<std::vector<long double>>forward(std::vector<std::vector<long double>>&data); // 1dim:data set / 2dim:data detail
        std::vector<std::vector<long double>>backward(std::vector<std::vector<long double>>&data);
    };
};