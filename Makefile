CXX = g++-10

Main : main.cpp Model.o Activation.o Layer.o Loss.o
	g++-10 -std=c++20 main.cpp Model.o Activation.o Layer.o Loss.o -o Main

Model.o : Model.cpp
	g++-10 -std=c++20 -c Model.cpp

Activation.o : Activation.cpp
	g++-10 -std=c++20 -c Activation.cpp

Layer.o : Layer.cpp
	g++-10 -std=c++20 -c Layer.cpp

Loss.o : Loss.cpp
	g++-10 -std=c++20 -c Loss.cpp

clear :
	rm ./*.o