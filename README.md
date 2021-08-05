# CppNN
## 概要
C++上でのニューラルネットワーク

- 使用するデータ型:`std::vector<std::vector<long double>>`
    ({{データ1}, {データ2}, {データ3}, ...} と言った感じ)
- 使用可能なレイヤー:Denseのみ
- 使用可能な活性化関数:恒等関数("linear")、シグモイド関数("sigmoid")、ReLU関数("relu")、softmax関数 / デフォルト:恒等関数
- 使用可能な損失関数:二乗誤差("mse")、クロスエントロピー誤差("cen") / デフォルト:クロスエントロピー誤差
- 使用可能な最適化関数:確率的勾配降下法("sgd")、AdaGrad("adagrad") / デフォルト:確率的勾配降下法

## 使用コード例

Makefileがあるので、ざっと見て使えそうだったら使ってみてください

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
- 最後のレイヤーの活性化関数にsoftmaxを使用した場合、学習時の損失関数は`cen`を推奨(この逆も推奨)
- 最後のレイヤーの活性化関数にlinearを使用した場合、学習時の損失関数は`mse`を推奨(この逆も推奨)

つまり、クロスエントロピー誤差は最後のレイヤーにsoftmaxが使われることを、二乗誤差は最後のレイヤーにlinearが使われることを想定しています。

これは、フレームワークとしてのものではなく、あくまで「C++でニューラルネットワークを実装するとこんな感じだよ」というものなので、ハイパーパラメータが勝手に決められている場合があります。

## 軽いリファレンス

<br>

### Model::Model(int input_size)
<hr>

- input_size : 入力の数

<br>

### void Model::AddDenseLayer(int unit, std::string activation)
<hr>

- unit : ユニット数(ニューロンの数)
- activation : 活性化関数(概要を参照。デフォルトはlinear)

指定されたユニット数・活性化関数のレイヤーを追加します。

<br>

### std::vector<std::vector<long double>> Model::predict(const std::vector<std::vector<long double>>&data)
<hr>

- data : 入力データ

入力データに対して推論を行います

<br>

### std::vector<long double> Model::fit(int step, long double learning_rate, std::vector<std::vector<long double>>&x, std::vector<std::vector<long double>>&y, int batch_size, std::string loss);
<hr>

- step : 学習回数
- learning_rate : 学習率
- x : 入力データ
- y : 教師データ
- batch_size : バッチ学習時のバッチに含めるデータ数
- loss : 損失関数(概要を参照。デフォルトはクロスエントロピー誤差)

<br>

### void Model::print()
<hr>

モデルの各パラメータを表示します。
