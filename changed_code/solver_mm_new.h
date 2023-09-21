// -------------------
// ---- Functions adapted from the solver_mm.py function ----
/* This contains all the functions for solving the asymptotic relation
# of the mixed modes, as they have been tested during their development
# All this arise from reading few papers from Benoit Mosser and 
# The PhD thesis from Charlotte Gehand:
# https://arxiv.org/pdf/1203.0689.pdf (Mosser paper on mixed modes)
# https://arxiv.org/pdf/1004.0449.pdf (older Mosser paper on scaling relations for gaussian_width, Amp etc.. - 2010 -)
# https://arxiv.org/pdf/1011.1928.pdf (The universal pattern introduced with the curvature - Fig. 3 - )
# https://arxiv.org/pdf/1411.1082.pdf
# https://tel.archives-ouvertes.fr/tel-02128409/document

# Examples and tests function have been built using asymptotic relations in the python original code.
# But note that they should be applicable to ANY set of value of:
#	 nu_p(nu), nu_g(nu), Dnu_p(nu) and DPl(nu) (meaning handling glitches)
*/
// ------------------
#pragma once 
#include <Eigen/Dense>
#include <Eigen/Core>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <random>
#include <chrono>
#ifdef _OPENMP
   #include <omp.h>
#else
   #define omp_get_thread_num() 0
#endif

//#include "version_solver.h"
#include "../unchanged_code/data_solver.h"
#include "../unchanged_code/string_handler.h"
#include "../unchanged_code/interpol.h"
//#include "derivatives_handler.h"
#include "../changed_code/linfit.h"
#include "../unchanged_code/solver_mm.h"

using Eigen::MatrixXd;
using Eigen::VectorXi;
using Eigen::VectorXd;

Eigen::VectorXd removeDuplicates(const Eigen::VectorXd& nu_m_all, double tol);


// This function uses solver_mm to find solutions from a spectrum
// of pure p modes and pure g modes following the asymptotic relations at the second order for p modes and the first order for g modes
//
//	Dnu_p: Average large separation for the p modes
//	epsilon: phase offset for the p modes
//	el: Degree of the mode
//	delta0l: first order shift related to core structure (and to D0)
//	alpha_p: Second order shift relate to the mode curvature
//	nmax: radial order at numax
//	DPl: average Period spacing of the g modes
//	alpha: phase offset for the g modes
//	q: coupling strength
//	sigma_p: standard deviation controling the randomisation of individual p modes. Set it to 0 for no spread
//	sigma_g: standard deviation controling the randomisation of individial g modes. Set it to 0 for no spread
//  fmin: minimum frequency to consider for p modes. Note that mixed mode solutions may be of lower frequency than this
//  fmax: maximum frequency to consider for p modes. Note that mixed mode solutions may be of lower frequency than this
//  resol: Control the grid resolution. Might be set to the resolution of the spectrum
//  returns_pg_freqs: If true, returns the values for calculated p and g modes
//  verbose: If true, print the solution on screen , VectorXd nu_l0_in=FixedXD::Zero(1)
Data_eigensols solve_mm_asymptotic_O2p_new(const long double Dnu_p, const long double epsilon, const int el, const long double delta0l, const long double alpha_p, 
	const long double nmax, const long double DPl, const long double alpha, const long double q, const long double sigma_p, 
	const long double fmin, const long double fmax, const long double resol, bool returns_pg_freqs=true, bool verbose=false);


// This function uses solver_mm to find solutions from a spectrum
// of pure p modes and pure g modes following the asymptotic relations at the second order for p modes and the first order for g modes
//
//	nu_l0_in: Frequencies for the l=0 modes. Used to derive nu_l1_p and therefore Dnu and epsilon
//	el: Degree of the mode
//	delta0l: first order shift related to core structure (and to D0)
//	alpha_p: Second order shift relate to the mode curvature
//	nmax: radial order at numax
//	DPl: average Period spacing of the g modes
//	alpha: phase offset for the g modes
//	q: coupling strength
//	sigma_p: standard deviation controling the randomisation of individual p modes. Set it to 0 for no spread
//	sigma_g: standard deviation controling the randomisation of individial g modes. Set it to 0 for no spread
//  resol: Control the grid resolution. Might be set to the resolution of the spectrum
//  returns_pg_freqs: If true, returns the values for calculated p and g modes
//  verbose: If true, print the solution on screen 
Data_eigensols solve_mm_asymptotic_O2from_l0_new(const VectorXd& nu_l0_in, const int el, const long double delta0l, 
    const long double DPl, const long double alpha, const long double q, const long double sigma_p, 
	const long double resol, bool returns_pg_freqs=true, bool verbose=false, const long double freq_min=0, const long double freq_max=1e6);


// This function uses solver_mm to find solutions from a spectrum
// of pure p modes and pure g modes following the asymptotic relations at the second order for p modes and the first order for g modes
//
// CONTRARY TO *froml0, this function takes nu(l) as in input directly. HOWEVER NOTE THAT 
// IN THIS FORM, IT USES linfit() TO COMPUTE Dnu from nu(l). THIS IS NOT OPTIMAL AS Dnu(l=0) 
// IS USUALLY SLIGHTLY DIFFERENT THAN Dnu(l)
// ANOTHER DIFFERENCE IS THAT THIS FUNCTION SEARCH SOLUTION OVER +/- 1.75*Dnu_p_local 
// WHILE THE OTHER SIMILAR FUNCTIONS (O2 of froml0) SEARCH OVER A CONSTANT WINDOW Dnu_p
// BOTH ARE ACCEPTABLE BUT WILL LEAD TO SLIGHTLY DIFFERENT SET OF SOLUTIONS
//	nu_p_all: Frequencies for the l modes.
//	el: Degree of the mode
//	alpha_p: Second order shift relate to the mode curvature
//	nmax: radial order at numax
//	DPl: average Period spacing of the g modes
//	alpha: phase offset for the g modes
//	q: coupling strength
//	sigma_p: standard deviation controling the randomisation of individual p modes. Set it to 0 for no spread
//	sigma_g: standard deviation controling the randomisation of individial g modes. Set it to 0 for no spread
//  resol: Control the grid resolution. Might be set to the resolution of the spectrum
//  returns_pg_freqs: If true, returns the values for calculated p and g modes
//  verbose: If true, print the solution on screen 
Data_eigensols solve_mm_asymptotic_O2from_nupl_new(const VectorXd& nu_p_all, const int el, //const long double delta0l, 
    const long double DPl, const long double alpha, const long double q, const long double sigma_p, 
	const long double resol, bool returns_pg_freqs=true, bool verbose=false, const long double freq_min=0, const long double freq_max=1e6);
