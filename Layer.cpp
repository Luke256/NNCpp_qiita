#include "Layer.hpp"

Layers::Dense::Dense(int input_unit, int unit, std::string activation):
bias(unit),
neuron(unit, std::vector<long double>(input_unit)),
m_activation(activation)
{

    double sigma = 0.05;
    if(activation == "relu") sigma = std::sqrt(2.0 / (double)input_unit);
    else if(activation == "sigmoid" || activation == "leaner") sigma = std::sqrt(1.0 / (double)input_unit);
    else sigma = 0.05;

    // initialize neuron and bias in random
    std::random_device seed;
    std::mt19937 engine(seed());
    std::normal_distribution<> generator(0.0, sigma);
    for(int i = 0; i < unit; ++i){
        bias[i] = generator(engine);
        for(int j = 0; j < input_unit; ++j){
            neuron[i][j] = generator(engine);
        }
    }
}

std::vector<std::vector<long double>> Layers::Dense::forward(std::vector<std::vector<long double>> &data){
    last_data=data;

    std::vector<std::vector<long double>>ans;

    // データごとに処理
    for(int index = 0; auto &i : data){
        if(i.size() != neuron[0].size()){
            std::cerr << "unexpected data was taken. expected data's size was " << neuron[0].size() << ". but the data's size was " << i.size() << "." << std::endl;
        }

        std::vector<long double>res;
        for(int j = 0; j < neuron.size(); ++j){
            long double t = 0;
            for(int k = 0; k < neuron[j].size(); ++k){
                t+=i[k]*neuron[j][k];
            }

            t-=bias[j];

            res.push_back(t);
        }

        // data[index] = res; // 置き換え
        ans.push_back(res);

        ++index;
    }

    ans = m_activation.forward(ans); // 活性化関数を適用(一気に)

    return ans;
}


std::vector<std::vector<long double>> Layers::Dense::backward(std::vector<std::vector<long double>> &data){
    // 返すもの:dx
    // 返さないけど計算するもの:
    // dw, db (= 勾配)

    // まずは活性化関数を通す
    data = m_activation.backward(data);

    // dxを計算
    std::vector<std::vector<long double>> ans;
    for(int i = 0; i < data.size(); ++i){
        std::vector<long double>res;
        for(int j = 0; j < neuron[0].size(); ++j){
            long double t=0;
            for(int k = 0; k < neuron.size(); ++k){
                t += neuron[k][j] * data[i][k];
            }
            res.push_back(t);
        }
        ans.push_back(res);
    }

    // dwを計算(dw : ニューロンの勾配)
    grad_layer.clear();
    for(int i = 0; i < neuron.size(); ++i){
        std::vector<long double>res;
        for(int j = 0; j < neuron[i].size(); ++j){
            long double t = 0;
            for(int k = 0; k < data.size(); ++k){
                t += data[k][i] * last_data[k][j];
            }
            res.push_back(t);
        }
        grad_layer.push_back(res);
    }

    // dbを計算(db : バイアスの勾配)
    grad_bias.clear();
    for(int i = 0; i < bias.size(); ++i){
        long double t = 0;
        for(int j = 0; j < data.size(); ++j){
            t += data[j][i];
        }
        grad_bias.push_back(t);
    }


    return ans;
}