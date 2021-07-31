#pragma once
#include <bits/stdc++.h>

namespace Loss{
    long double mean_squared_error(std::vector<std::vector<long double>> &y, std::vector<std::vector<long double>> &t);
    long double mean_cross_entropy_error(std::vector<std::vector<long double>> &y, std::vector<std::vector<long double>> &t);
    std::vector<std::vector<long double>> mean_squared_error_back(long double x, std::vector<std::vector<long double>>absolute_error);
    std::vector<std::vector<long double>> mean_cross_entropy_error_back(std::vector<std::vector<long double>>error);
};