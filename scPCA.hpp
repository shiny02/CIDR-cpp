#include <mlpack/core.hpp>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <armadillo>

namespace scPCA {
    void cpp_dist(arma::mat dist, arma::mat truth, arma::mat counts, int ncol, double threshold);
}
