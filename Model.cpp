#include "Model.hpp"

NNModel::NNModel(int input_size):
m_input_size(input_size),
m_output_size(input_size),
compiled(false)
{

}

void NNModel::AddDenseLayer(int unit, ActivationType activation){
    Layers::Dense dense(m_output_size, unit, activation);

    model.push_back(dense);
    m_output_size = unit;
}

std::vector<std::vector<double>> NNModel::predict(const std::vector<std::vector<double>>&data){
    std::vector<std::vector<double>>res = data;
    for(int index=0; auto &layer : model){
        res = layer.forward(res);
        ++index;
    }

    return res;
}

void NNModel::backward(){
    std::vector<std::vector<double>>data;
    // data = diff_error;
    if(m_loss == "mse") data = Loss::mean_squared_error_back(diff_error);
    else data = Loss::mean_cross_entropy_error_back(diff_error);

    for(int i = model.size() - 1; i>=0; --i){
        data = model[i].backward(data);
    }
}

// パラメータ(ニューロン間)用
std::vector<std::vector<double>> NNModel::numerical_gradient_layer(std::vector<std::vector<double>>&batch_x, std::vector<std::vector<double>>&batch_y, int index){
    
    double h = 1e-4;
    std::vector<std::vector<double>> grad(model[index].neuron.size(), std::vector<double>(model[index].neuron[0].size()));
    for(int i = 0; i < grad.size(); ++i){
        for(int j = 0; j < grad[i].size(); ++j){
            double tmp = model[index].neuron[i][j];

            model[index].neuron[i][j] = tmp + h;
            double fxh1 = caluculate_loss(batch_x, batch_y);

            model[index].neuron[i][j] = tmp - h;
            double fxh2 = caluculate_loss(batch_x, batch_y);

            grad[i][j] = (fxh1 - fxh2) / (2 * h);
            model[index].neuron[i][j] = tmp;
        }
    }

    return grad;
}

// バイアス用
std::vector<double> NNModel::numerical_gradient_bias(std::vector<std::vector<double>>&batch_x, std::vector<std::vector<double>>&batch_y, int index){
    // f: この場合損失を返す関数であればいい
    double h = 1e-4;
    std::vector<double> grad(model[index].bias.size());
    for(int i = 0; i < grad.size(); ++i){
        double tmp = model[index].bias[i];

        model[index].bias[i] = tmp + h;
        double fxh1 = caluculate_loss(batch_x, batch_y);

        model[index].bias[i] = tmp - h;
        double fxh2 = caluculate_loss(batch_x, batch_y);

        grad[i] = (fxh1 - fxh2) / (2 * h);
        model[index].bias[i] = tmp;
    }

    return grad;
}

// 学習
std::vector<double> NNModel::fit(int step, double learning_rate, std::vector<std::vector<double>>&x, std::vector<std::vector<double>>&y, int batch_size, std::string loss){


    std::vector<double>history;
    m_loss=loss;

    double Adam_R_beta = RMSProp_beta; // AdamのRMSProp部分
    double Adam_M_beta = Momentum_alpha; // AdamのMomentum部分

    if(!compiled){
        std::cout << "コンパイルがまだです" << std::endl;
        return history;
    }

    for(int current_step = 0; current_step < step; ++current_step){
        std::vector<std::vector<double>>batch_x, batch_y;
        diff_error.clear();
        for(int i = 0; i < batch_size; ++i){
            batch_x.push_back(x[(batch_size*current_step+i)%x.size()]);
            batch_y.push_back(y[(batch_size*current_step+i)%y.size()]);
        }

        // 逆伝播の準備
        auto test_y = predict(batch_x);
        for(int i = 0; i < test_y.size(); ++i){
            std::vector<double>t;
            for(int j = 0; j < test_y[i].size(); ++j){
                t.push_back((test_y[i][j] - batch_y[i][j])/batch_size);
            }
            diff_error.push_back(t);
        }

        // 逆伝播
        backward();


        double loss_step;
        if(m_loss == "mse") loss_step = Loss::mean_squared_error(test_y, batch_y);
        else loss_step = Loss::mean_cross_entropy_error(test_y, batch_y);


        for(int index = 0; auto &layer : model){
            // 勾配を計算
            // 数値微分(遅い)
            // std::vector<std::vector<double>>layer_grad = numerical_gradient_layer(batch_x, batch_y, index);
            // std::vector<double>bias_grad = numerical_gradient_bias(batch_x, batch_y, index);

            // 誤差逆伝播法(早い)
            std::vector<std::vector<double>>layer_grad = layer.grad_layer;
            std::vector<double>bias_grad = layer.grad_bias;

            // 計算した勾配に従って重みを調整
            // 1.layerの調整
            for(int i = 0; i < layer_grad.size(); ++i){
                for(int j = 0; j < layer_grad[i].size(); ++j){
                    if(m_optimizer == "AdaGrad"){
                        layer.h_layer[i][j] += layer_grad[i][j] * layer_grad[i][j];
                        layer.neuron[i][j] -= learning_rate * layer_grad[i][j] * (1 / (sqrt(layer.h_layer[i][j]) + 1e-7));
                    }
                    else if(m_optimizer == "RMSProp"){
                        layer.h_layer[i][j] = RMSProp_beta * layer.h_layer[i][j] + (1 - RMSProp_beta) * layer_grad[i][j] * layer_grad[i][j];
                        layer.neuron[i][j] -= learning_rate * layer_grad[i][j] * (1 / (sqrt(layer.h_layer[i][j] + 1e-7)));
                    }
                    else if(m_optimizer == "Momentum"){
                        layer.v_layer[i][j] = Momentum_alpha * layer.v_layer[i][j] + learning_rate * layer_grad[i][j];
                        layer.neuron[i][j] += layer.v_layer[i][j];
                    }
                    else if(m_optimizer == "Adam"){
                        layer.h_layer[i][j] = RMSProp_beta * layer.h_layer[i][j] + (1 - RMSProp_beta) * layer_grad[i][j] * layer_grad[i][j];
                        layer.v_layer[i][j] = Momentum_alpha * layer.v_layer[i][j] + (1 - Momentum_alpha) * layer_grad[i][j];
                        double om = layer.h_layer[i][j] / (1 - Adam_R_beta);
                        double ov = layer.v_layer[i][j] / (1 - Adam_M_beta);
                        layer.neuron[i][j] -= learning_rate * ov * (1 / (sqrt(om) + 1e-7));
                    }
                    else layer.neuron[i][j] -= learning_rate * layer_grad[i][j];
                }
            }

            // 2.biasの調整
            for(int i = 0; i < bias_grad.size(); ++i){
                if(m_optimizer == "AdaGrad"){
                    layer.h_bias[i] += bias_grad[i] * bias_grad[i];
                    layer.bias[i] -= learning_rate * bias_grad[i] * (1 / (sqrt(layer.h_bias[i]) + 1e-7));
                }
                else if(m_optimizer == "RMSProp"){
                    layer.h_bias[i] = RMSProp_beta * layer.h_bias[i] + (1 - RMSProp_beta) * bias_grad[i] * bias_grad[i];
                    layer.bias[i] -= learning_rate * bias_grad[i] * (1 / (sqrt(layer.h_bias[i] + 1e-7)));
                }
                else if(m_optimizer == "Momentum"){
                    layer.v_bias[i] = Momentum_alpha * layer.v_bias[i] + learning_rate * bias_grad[i];
                    layer.bias[i] += layer.v_bias[i];
                }
                else if(m_optimizer == "Adam"){
                    layer.h_bias[i] = RMSProp_beta * layer.h_bias[i] + (1 - RMSProp_beta) * bias_grad[i] * bias_grad[i];
                    layer.v_bias[i] = Momentum_alpha * layer.v_bias[i] + (1 - Momentum_alpha) * bias_grad[i];
                    double om = layer.h_bias[i] / (1 - Adam_R_beta);
                    double ov = layer.v_bias[i] / (1 - Adam_M_beta);
                    layer.bias[i] -= learning_rate * ov * (1 / (sqrt(om) + 1e-7));
                }
                else layer.bias[i] -= learning_rate * bias_grad[i];
            }
            ++index;
        }
        Adam_R_beta *= RMSProp_beta;
        Adam_M_beta *= Momentum_alpha;
        // 学習経過の記録
        history.push_back(loss_step);

        if((current_step + 1) % 100 == 0) std::cout << current_step << "ステップ終了 loss : " << loss_step << std::endl;
    }

    return history;
}

double NNModel::caluculate_loss(std::vector<std::vector<double>>&batch_x, std::vector<std::vector<double>>&batch_y){
    std::vector<std::vector<double>>t = predict(batch_x);
    if(m_loss == "cen") return Loss::mean_cross_entropy_error(t, batch_y);
    return Loss::mean_squared_error(t, batch_y);
}

void NNModel::print(){
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

void NNModel::compile(std::string optimizer){
    m_optimizer = optimizer;
    if(optimizer == "AdaGrad" || optimizer == "RMSProp"){
        for(auto &i : model){
            i.h_layer.resize(i.neuron.size(), std::vector<double>(i.neuron[0].size()));
            i.h_bias.resize(i.bias.size());
        }
    }
    else if(optimizer == "SGD"){}
    else if(optimizer == "Momentum"){
        for(auto &i : model){
            i.v_layer.resize(i.neuron.size(), std::vector<double>(i.neuron[0].size()));
            i.v_bias.resize(i.bias.size());
        }
    }
    else if(optimizer == "Adam"){
        for(auto &i : model){
            i.h_layer.resize(i.neuron.size(), std::vector<double>(i.neuron[0].size()));
            i.h_bias.resize(i.bias.size());
        }
        for(auto &i : model){
            i.v_layer.resize(i.neuron.size(), std::vector<double>(i.neuron[0].size()));
            i.v_bias.resize(i.bias.size());
        }
    }
    else{
        std::cout << "最適化関数" << optimizer << "が見つかりませんでした。代わりに、SGDが適用されます" << std::endl;
    }

    compiled = true;
}
