g++ -Ofast -std=c++11 -fopenmp -march=native -fpic -w -ftree-vectorize prepare_graph.cpp -o build_graph
./build_graph sift angular
