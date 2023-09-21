#include <Eigen/Dense>
#include <cmath>
#include "interpol.h"
# include <iostream>
# include <iomanip>

using Eigen::VectorXd;


double lin_interpol_new(const Eigen::VectorXd& x, const Eigen::VectorXd& y, const double x_int) {
    long i = 0, Nx = x.size();
    double a = 0, b = 0;
    // ---- case of an actual interpolation -----
    if (x_int >= x(0) && x_int <= x(Nx - 1)) {
        long left = 0;
        long right = Nx - 1;
        while (right - left > 1) {
            long mid = (left + right) / 2;
            if (x(mid) <= x_int) {
                left = mid;
            } else {
                right = mid;
            }
        }
        i = left;

        a = (y(i + 1) - y(i)) / (x(i + 1) - x(i));
        b = y(i) - a * x(i);
    }
    // ---- case of an extrapolation toward the lower edge ----
    if (x_int < x(0)) {
        a = (y(1) - y(0)) / (x(1) - x(0));
        b = y(0) - a * x(0);
    }
    // ---- case of an extrapolation toward the upper edge ----
    if (x_int > x(Nx - 1)) {
        a = (y(Nx - 1) - y(Nx - 2)) / (x(Nx - 1) - x(Nx - 2));
        b = y(Nx - 1) - a * x(Nx - 1);
    }
    return a * x_int + b;
}
