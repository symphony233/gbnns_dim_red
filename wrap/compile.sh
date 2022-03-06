swig -python c_support.i
g++ -c -Ofast -std=c++11 -fopenmp -march=native -fpic -w -ftree-vectorize c_support_wrap.c c_support.cpp -I"/usr/include/python3.8/"
g++ -shared c_support.o c_support_wrap.o -o _c_support.so
