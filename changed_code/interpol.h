#pragma once
#include <Eigen/Dense>
#include <cmath>

using Eigen::VectorXd;

double lin_interpol_new(const VectorXd& x, const VectorXd& y, const double x_int);
