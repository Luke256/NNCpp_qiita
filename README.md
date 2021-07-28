# CppNN
## 概要
C++上でのニューラルネットワーク

- 使用するデータ型:`std::vector<std::vector<long double>>`
    ({{データ1}, {データ2}, {データ3}, ...} と言った感じ)
- 使用可能なレイヤー:Denseのみ
- 使用可能な活性化関数:恒等関数("leaner")、シグモイド関数("sigmoid")、ReLU関数("relu")、softmax関数 / デフォルト:恒等関数
- 使用可能な損失関数:二乗誤差("mse")、クロスエントロピー誤差("cen") / デフォルト:クロスエントロピー誤差

## 使用コード例
```C++
#include "Model.hpp"
#include "bits/stdc++.h"

using namespace std;

int main(){
    Model model(2);
    model.AddDenseLayer(64, "relu");
    model.AddDenseLayer(2, "softmax");

    vector<vector<long double>> x = 
    {
        {1, 0},
        {1, 1},
        {0, 0},
        {0, 1}
    };
    vector<vector<long double>> y = 
    {
        {0, 1},
        {1, 0},
        {1, 0},
        {0, 1}
    };


    auto res = model.predict(x);
    for(int i=0;i<4;++i){
        for(auto j:x[i])cout << j<<" ";
        cout << ": ";
        for(auto j:res[i])cout << j<<" ";
        cout << endl;
    }
    cout << endl;

    auto history = model.fit(1000, 0.01, x, y, 4, "cen");

    res = model.predict(x);

    for(int i=0;i<4;++i){
        for(auto j:x[i])cout << j<<" ";
        cout << ": ";
        for(auto j:res[i])cout << round(j) << "(" << j<<") ";
        cout << endl;
    }

}
```

## 注意
- 最後のレイヤーの活性化関数にsoftmaxを使用した場合、学習時の損失関数は`cen`を推奨
