make -j4 2> error.log

if [ -s error.log ];then
    if grep -q "error" error.log;then
        echo コンパイルエラーが発生しました
    else
        echo "ビルド終了(Warning)"
        ./Main
    fi
else
    echo "ビルド終了(正常)"
    ./Main
fi
echo
