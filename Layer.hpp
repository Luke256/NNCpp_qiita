#pragma once
#include "bits/stdc++.h"
#include "Activation.hpp"

namespace Layers{
    class Dense{
    // private:
    public:
        Activation m_activation;
        std::vector<double>bias;
        std::vector<std::vector<double>> neuron; // neuron[i][j]:=the weight from previous j to current i.
        std::vector<std::vector<double>> last_data;

        std::vector<std::vector<double>>grad_layer;
        std::vector<double>grad_bias;

        std::vector<std::vector<double>> h_layer; // AdaGrad用パラメータ
        std::vector<double> h_bias; // AdaGrad用パラメータ

        
        std::vector<std::vector<double>> v_layer; // Momentum用パラメータ
        std::vector<double> v_bias; // Momentum用パラメータ

        Dense(int input_unit, int unit, ActivationType activation);
        std::vector<std::vector<double>>forward(std::vector<std::vector<double>>&data); // 1dim:data set / 2dim:data detail
        std::vector<std::vector<double>>backward(std::vector<std::vector<double>>&data);
    };
};
