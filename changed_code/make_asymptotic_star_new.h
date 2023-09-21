#pragma once 
#include <Eigen/Dense>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include "../unchanged_code/version_solver.h"
#include "../unchanged_code/data_solver.h"
//#include "../unchanged_code/bump_DP.h"
//#include "string_handler.h"
#include "../unchanged_code/interpol.h"
#include "../unchanged_code/noise_models.h" // get the harvey_1985 function
#include "../changed_code/solver_mm_new.h"
#include "../changed_code/ksi2_fct_new.h"

Params_synthetic_star make_synthetic_asymptotic_star_new(Cfg_synthetic_star cfg_star);