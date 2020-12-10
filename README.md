# CIDR-cpp
C++ implementation of the CIDR algorithm for single-cell RNA transcriptomics data

To run, first make sure that you have mlpack and ALGLIB isntalled in your machine. Afterwards, you can run "g++ -std=c++11 -larmadillo -lboost_serialization -lmlpack -lgsl -lalglib draft1.cpp kde.c kde.h scPCA.hpp scPCA.cpp -o CIDR_cpp -fopenmp". The "-fopenmp" flag is necessary for some of the mlpack functionality. The "-l" flags aren't necessary as long as the libraries are linked to the executable.
