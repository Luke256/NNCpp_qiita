#include "Model.hpp"
#include "Loss.hpp"

Model::Model(int input_size):
m_input_size(input_size),
m_output_size(input_size)
{

}

void Model::AddDenseLayer(int unit, std::string activation){
    Layers::Dense dense(m_output_size, unit, activation);
    model.push_back(dense);
    m_output_size = unit;
}

std::vector<std::vector<long double>> Model::predict(const std::vector<std::vector<long double>>&data){
    std::vector<std::vector<long double>>res = data;
    for(int index=0; auto &layer : model){
        res = layer.forward(res);
        ++index;
    }

    return res;
}

void Model::backward(){
    std::vector<std::vector<long double>>data;
    // data = diff_error;
    if(m_loss == "mse") data = Loss::mean_squared_error_back(1, diff_error);
    else data = Loss::mean_cross_entropy_error_back(diff_error);


    for(int i = model.size() - 1; i>=0; --i){
        data = model[i].backward(data);
    }
}

// パラメータ(ニューロン間)用
std::vector<std::vector<long double>> Model::numerical_gradient_layer(std::vector<std::vector<long double>>&batch_x, std::vector<std::vector<long double>>&batch_y, int x){
    
    long double h = 1e-4;
    std::vector<std::vector<long double>> grad(model[x].neuron.size(), std::vector<long double>(model[x].neuron[0].size()));
    for(int i = 0; i < grad.size(); ++i){
        for(int j = 0; j < grad[i].size(); ++j){
            long double tmp = model[x].neuron[i][j];

            model[x].neuron[i][j] = tmp + h;
            long double fxh1 = caluculate_loss(batch_x, batch_y);

            model[x].neuron[i][j] = tmp - h;
            long double fxh2 = caluculate_loss(batch_x, batch_y);

            grad[i][j] = (fxh1 - fxh2) / (2 * h);
            model[x].neuron[i][j] = tmp;
        }
    }

    return grad;
}

// バイアス用
std::vector<long double> Model::numerical_gradient_bias(std::vector<std::vector<long double>>&batch_x, std::vector<std::vector<long double>>&batch_y, int x){
    // f: この場合損失を返す関数であればいい
    long double h = 1e-4;
    std::vector<long double> grad(model[x].bias.size());
    for(int i = 0; i < grad.size(); ++i){
        long double tmp = model[x].bias[i];

        model[x].bias[i] = tmp + h;
        long double fxh1 = caluculate_loss(batch_x, batch_y);

        model[x].bias[i] = tmp - h;
        long double fxh2 = caluculate_loss(batch_x, batch_y);

        grad[i] = (fxh1 - fxh2) / (2 * h);
        model[x].bias[i] = tmp;
    }

    return grad;
}

// 学習
std::vector<long double> Model::fit(int step, long double learning_rate, std::vector<std::vector<long double>>&x, std::vector<std::vector<long double>>&y, int batch_size, std::string loss){

    std::vector<long double>history;
    m_loss=loss;


    for(int current_step = 0; current_step < step; ++current_step){
        std::vector<std::vector<long double>>batch_x, batch_y;
        diff_error.clear();
        for(int i = 0; i < batch_size; ++i){
            batch_x.push_back(x[(batch_size*current_step+i)%x.size()]);
            batch_y.push_back(y[(batch_size*current_step+i)%y.size()]);
        }

        // 逆伝播の準備
        auto test_y = predict(batch_x);
        for(int i = 0; i < test_y.size(); ++i){
            std::vector<long double>t;
            for(int j = 0; j < test_y[i].size(); ++j){
                t.push_back((test_y[i][j] - batch_y[i][j])/batch_size);
            }
            diff_error.push_back(t);
        }

        // 逆伝播
        backward();


        long double loss_step;
        if(m_loss == "mse") loss_step = Loss::mean_squared_error(test_y, y);
        else loss_step = Loss::mean_cross_entropy_error(test_y, y);


        for(int index = 0; auto &layer : model){
            // 勾配を計算
            // 数値微分(遅い)
            // std::vector<std::vector<long double>>layer_grad = numerical_gradient_layer(batch_x, batch_y, index);
            // std::vector<long double>bias_grad = numerical_gradient_bias(batch_x, batch_y, index);

            // 誤差逆伝播法(早い)
            std::vector<std::vector<long double>>layer_grad = layer.grad_layer;
            std::vector<long double>bias_grad = layer.grad_bias;

            // 計算した勾配に従って重みを調整
            // 1.layerの調整
            for(int i = 0; i < layer_grad.size(); ++i){
                for(int j = 0; j < layer_grad[i].size(); ++j){
                    layer.neuron[i][j] -= learning_rate * layer_grad[i][j];
                }
            }

            // 2.biasの調整
            for(int i = 0; i < bias_grad.size(); ++i){
                layer.bias[i] -= learning_rate * bias_grad[i];
            }

            ++index;
        }

        // 学習経過の記録
        history.push_back(loss_step);
    }

    return history;
}

long double Model::caluculate_loss(std::vector<std::vector<long double>>&batch_x, std::vector<std::vector<long double>>&batch_y){
    std::vector<std::vector<long double>>t = predict(batch_x);
    if(m_loss == "cen") return Loss::mean_cross_entropy_error(t, batch_y);
    return Loss::mean_squared_error(t, batch_y);
}

void Model::print(){
    for(auto layer : model){
        for(auto i : layer.neuron){
            for(auto j : i) std::cout << j << " ";
            std::cout << std::endl;
        }

        for(auto i : layer.bias){
            std::cout << i<< " ";
        }
        std::cout << std::endl << std::endl;
    }
}