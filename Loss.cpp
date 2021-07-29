#include "Loss.hpp"

namespace Loss{

    long double mean_squared_error(std::vector<std::vector<long double>> &y, std::vector<std::vector<long double>> &t){
        long double res = 0;
        if(y.size()!=t.size()){
            std::cerr << "invalid size. y has size " << y.size() << ", but t has size " << t.size() << "." << std::endl;
        }
        for(int i=0; i < y.size(); ++i){
            long double sum=0;
            for(int j=0; j<y[0].size(); ++j){
                sum += (y[i][j] - t[i][j]) * (y[i][j] - t[i][j]);
            }

            sum /= 2;
            res+=sum;
        }

        return res / y.size();
    }

    long double mean_cross_entropy_error(std::vector<std::vector<long double>> &y, std::vector<std::vector<long double>> &t){
        long double res = 0;
        if(y.size()!=t.size()){
            std::cerr << "invalid size. y has size " << y.size() << ", but t has size " << t.size() << "." << std::endl;
        }
        for(int i=0; i < y.size(); ++i){
            long double sum=0;
            long double delta = 1e-7;
            for(int j=0; j<y[0].size(); ++j){
                sum += t[i][j] * std::log(y[i][j] + delta);
            }
            sum *= -1;
            res+=sum;
        }
        return res / y.size();
    }

    std::vector<std::vector<long double>> mean_squared_error_back(long double x, std::vector<std::vector<long double>>absolute_error){
        // std::vector<std::vector<long double>>res;

        // for(int i = 0; i < absolute_error.size(); ++i){
        //     std::vector<long double>t;
        //     for(int j = 0; j < absolute_error[i].size(); ++j){
        //         t.push_back(2*x*abs(absolute_error[i][j]));
        //     }
        //     res.push_back(t);
        // }

        // return res;
        return absolute_error;
    }

    std::vector<std::vector<long double>> mean_cross_entropy_error_back(std::vector<std::vector<long double>>error){
        return error;
    }

};