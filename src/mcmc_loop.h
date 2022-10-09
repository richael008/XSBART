#include <ctime>
#include "tree.h"
#include "forest.h"
#include <chrono>
#include "model.h"

#include "state.h"
#include "cdf.h"
#include "X_struct.h"
//#include "MH.h"

void mcmc_loop(matrix<size_t> &Xorder_std, bool verbose, matrix<double> &sigma_draw_xinfo, vector<vector<tree>> &trees, double no_split_penalty, std::unique_ptr<State> &state, NormalModel *model, std::unique_ptr<X_struct> &x_struct,arma::mat &PerMatrix_std);

