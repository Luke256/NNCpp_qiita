#pragma once
#include "bits/stdc++.h"

namespace Loss{
    double mean_squared_error(std::vector<std::vector<double>> &y, std::vector<std::vector<double>> &t);
    double mean_cross_entropy_error(std::vector<std::vector<double>> &y, std::vector<std::vector<double>> &t);
    std::vector<std::vector<double>> mean_squared_error_back( std::vector<std::vector<double>>absolute_error);
    std::vector<std::vector<double>> mean_cross_entropy_error_back(std::vector<std::vector<double>>error);
};
