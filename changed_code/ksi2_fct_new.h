#pragma once 
#include <Eigen/Dense>
#include "../unchanged_code/interpol.h"
#ifdef _OPENMP
   #include <omp.h>
#else
   #define omp_get_thread_num() 0
#endif

Eigen::VectorXd ksi_fct2_new(const Eigen::VectorXd& nu, const Eigen::VectorXd& nu_p, const Eigen::VectorXd& nu_g, const Eigen::VectorXd& Dnu_p, const Eigen::VectorXd& DPl, const long double q, const std::string norm_method="fast");
Eigen::VectorXd ksi_fct2_precise(const Eigen::VectorXd& nu, const Eigen::VectorXd& nu_p, const Eigen::VectorXd& nu_g, const Eigen::VectorXd& Dnu_p, const Eigen::VectorXd& DPl, const long double q);


