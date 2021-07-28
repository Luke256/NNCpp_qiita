#include "Model.hpp"
// #include "Activation.hpp"
#include "bits/stdc++.h"
#define rep(i,n) for(ll i=0;i<n;++i)

using namespace std;

int main(){
    // Activation::Activation a("relu");
    // vector<vector<long double>> x={
    //     {1,0,3,-1},
    //     {-1, 4, 1, 5}
    // };
    // auto t=a.forward(x);
    // t={
    //     {0.1, 2.2, 1, -0.4},
    //     {0.1, 2.2, 1, -0.4}
    // };
    // auto b=a.backward(t);

    // for(auto i:x){
    //     for(auto j:i)cout << j << " ";
    //     cout << endl;
    // }
    // cout << endl;
    // for(auto i:b){
    //     for(auto j:i)cout << j << " ";
    //     cout << endl;
    // }


    Model model(2);
    model.AddDenseLayer(64, "relu");
    model.AddDenseLayer(2, "softmax");

    // return 0;



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

    // model.print();


    auto history = model.fit(1000, 0.01, x, y, 4, "cen");

    // exit(0);

    // cout << endl;
    res = model.predict(x);

    // model.print();
    for(int i=0;i<4;++i){
        for(auto j:x[i])cout << j<<" ";
        cout << ": ";
        for(auto j:res[i])cout << round(j) << "(" << j<<") ";
        cout << endl;
    }
    cout << endl << endl;
    // for(auto i : history) cout << i << " ";

}