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

#include "../unchanged_code/data_solver.h"
#include "../unchanged_code/string_handler.h"
#include "../unchanged_code/interpol.h"
#include "../changed_code/linfit.h"
#include "../unchanged_code/solver_mm.h"

using Eigen::MatrixXd;
using Eigen::VectorXi;
using Eigen::VectorXd;

Eigen::VectorXd removeDuplicates(const Eigen::VectorXd& nu_m_all, double tol) {
    Eigen::VectorXd uniqueVec;
    
    for(int i = 0; i < nu_m_all.size(); i++) {
        bool isDuplicate = false;
        
        for(int j = 0; j < uniqueVec.size(); j++) {
            if(std::abs(nu_m_all[i] - uniqueVec[j]) <= tol) {
                isDuplicate = true;
                break;
            }
        }
        
        if(!isDuplicate) {
            uniqueVec.conservativeResize(uniqueVec.size() + 1);
            uniqueVec[uniqueVec.size() - 1] = nu_m_all[i];
        }
    }
    
    return uniqueVec;
}


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
	const long double fmin, const long double fmax, const long double resol, bool returns_pg_freqs, bool verbose)
{
	const bool returns_axis=true;
	const int Nmmax=100000; //Ngmax+Npmax;
	const double tol=2*resol; // Tolerance while searching for double solutions of mixed modes
	
	unsigned seed_p = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine gen_p(seed_p); //, gen_g(seed_g);
	std::normal_distribution<double> distrib_p(0.,sigma_p);
	
	bool success;
	int np, ng, np_min, np_max, ng_min, ng_max;
	double nu_p, nu_g, Dnu_p_local, DPl_local; // Dnu_p_local and DPl_local are important if modes do not follow exactly the asymptotic relation.
	double fact=0.04;  // Default factor
	double r;

	VectorXd nu_p_all, nu_g_all, nu_m_all(Nmmax);

	Data_eigensols nu_sols;
	Deriv_out deriv_p, deriv_g;

	// Use fmin and fmax to define the number of pure p modes and pure g modes to be considered
	np_min=int(floor(fmin/Dnu_p - epsilon - el/2 - delta0l));
	np_max=int(ceil(fmax/Dnu_p - epsilon - el/2 - delta0l));

	np_min=int(floor(np_min - alpha_p*std::pow(np_min - nmax, 2) /2.));
	np_max=int(ceil(np_max + alpha_p*std::pow(np_max - nmax, 2) /2.)); // CHECK THIS DUE TO - -

	ng_min=int(floor(1e6/(fmax*DPl) - alpha));
	ng_max=int(ceil(1e6/(fmin*DPl) - alpha));

	if (np_min <= 0)
	{
		np_min=1;
	}
	if (fmin <= 150) // overrides of the default factor in case fmin is low
	{
		fact=0.01;
	}
	if (fmin <= 50)
	{
		fact=0.005;
	}
	// Handling the p and g modes, randomized or not
	nu_p_all.resize(np_max-np_min);
	nu_g_all.resize(ng_max-ng_min);
	for (int np=np_min; np<np_max; np++)
	{
		nu_p=asympt_nu_p(Dnu_p, np, epsilon, el, delta0l, alpha_p, nmax);
		if (sigma_p == 0)
		{
			nu_p=asympt_nu_p(Dnu_p, np, epsilon, el, delta0l, alpha_p, nmax);
		} else{
			r = distrib_p(gen_p);
			nu_p=asympt_nu_p(Dnu_p, np, epsilon, el, delta0l, alpha_p, nmax, r);
		}		
		nu_p_all[np-np_min]=nu_p;
	}

	for (int ng=ng_min; ng<ng_max;ng++)
	{
		nu_g=asympt_nu_g(DPl, ng, alpha);
		nu_g_all[ng-ng_min]=nu_g;
	}
	deriv_p=Frstder_adaptive_reggrid(nu_p_all);
	deriv_g.deriv.resize(nu_g_all.size());
	deriv_g.deriv.setConstant(DPl);
	std::vector<double> filteredVec;
	#pragma omp parallel for collapse(2)
	for (size_t np = 0; np < nu_p_all.size(); np++) {
		for (size_t ng = 0; ng < nu_g_all.size(); ng++) {
			double nu_p = nu_p_all[np];
			double nu_g = nu_g_all[ng];
			double Dnu_p_local = Dnu_p * (1.0 + alpha_p * (np + np_min - nmax));
			double DPl_local = DPl;

			Data_coresolver sols_iter = solver_mm(nu_p, nu_g, Dnu_p_local, DPl_local, q, nu_p - 1.75 * Dnu_p, nu_p + 1.75 * Dnu_p, resol, returns_axis, verbose, fact);
			if (sols_iter.nu_m.size() > 0) {
				for (int i = 0; i < sols_iter.nu_m.size(); i++) {
					if (sols_iter.nu_m[i] >= fmin && sols_iter.nu_m[i] <= fmax) {
						#pragma omp critical
						{
							filteredVec.push_back(sols_iter.nu_m[i]);
						}
					}
				}
			}
		}
	}
	
	std::sort(filteredVec.begin(), filteredVec.end());
	filteredVec.erase(std::unique(filteredVec.begin(), filteredVec.end(), [tol](double a, double b) {
		return std::abs(a - b) <= tol;
	}), filteredVec.end());

	nu_m_all.resize(filteredVec.size());
	std::copy(filteredVec.begin(), filteredVec.end(), nu_m_all.data());
	nu_m_all.resize(filteredVec.size());
	//#pragma omp parallel for
	//for (int i = 0; i < filteredVec.size(); i++) {
	//	nu_m_all[i] = filteredVec[i];
	//}

	if (returns_pg_freqs == true)
	{
		nu_sols.nu_m=nu_m_all;
		nu_sols.nu_p=nu_p_all;
		nu_sols.nu_g=nu_g_all;
		nu_sols.dnup=deriv_p.deriv;
		nu_sols.dPg=deriv_g.deriv;
		return nu_sols;
	} else
	{
		nu_sols.nu_m=nu_m_all;
		return nu_sols;
	}
}



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
	const long double resol, bool returns_pg_freqs, bool verbose, const long double freq_min, const long double freq_max)
{

	const bool returns_axis=true;
	const int Nmmax=100000; //Ngmax+Npmax;
	const double tol=2*resol; // Tolerance while searching for double solutions of mixed modes
	unsigned seed_p = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine gen_p(seed_p); //, gen_g(seed_g);
	std::normal_distribution<double> distrib_p(0.,sigma_p);
	
	int np, ng, ng_min, ng_max;//, np_min, np_max, attempts;
	double nu_p, nu_g, Dnu_p, epsilon, Dnu_p_local, DPl_local, fmin, fmax; // Dnu_p_local and DPl_local are important if modes does not follow exactly the asymptotic relation.
	double fact=0.04;  // Default factor

	VectorXi test;
	VectorXd fit, nu_p_all, nu_g_all, nu_m_all(Nmmax), results(Nmmax);	

	Data_coresolver sols_iter;
	Data_eigensols nu_sols;
	Deriv_out deriv_p, deriv_g;

	//tmp=linspace(0, nu_l0_in.size()-1, nu_l0_in.size());
	const Eigen::VectorXd tmp = Eigen::VectorXd::LinSpaced(nu_l0_in.size(), 0, nu_l0_in.size()-1);
    
	fit=linfit_new(tmp, nu_l0_in); // fit[0] is the slope ==> Dnu and fit[1] is the ordinate at origin ==> fit[1]/fit[0] = epsilon
	Dnu_p=fit[0];
	epsilon=fit[1]/fit[0];
	epsilon=epsilon - floor(epsilon);
	fmin=nu_l0_in.minCoeff() - Dnu_p;
	fmax=nu_l0_in.maxCoeff() + Dnu_p;

	nu_m_all.setConstant(-9999);
	if (fmin < 0){
		fmin=0;
	}

	ng_min=int(floor(1e6/(fmax*DPl) - alpha));
	ng_max=int(ceil(1e6/(fmin*DPl) - alpha));

	if (fmin <= 150) // overrides of the default factor in case fmin is low
	{
		fact=0.01;
	}
	if (fmin <= 50)
	{
		fact=0.005;
	}

	// Handling the p and g modes, randomized or not
	nu_g_all.resize(ng_max-ng_min);

	// Step of extrapolating edges to avoid egdes effect when shifting l=0 frequencies to generate l=1 p modes
	nu_p_all=asympt_nu_p_from_l0_Xd(nu_l0_in, Dnu_p, el, delta0l, fmin, fmax);
	
	for (int ng=ng_min; ng<ng_max;ng++)
	{
		nu_g=asympt_nu_g(DPl, ng, alpha);
		nu_g_all[ng-ng_min]=nu_g;
	}
	
	deriv_p=Frstder_adaptive_reggrid(nu_p_all);
	deriv_g.deriv.resize(nu_g_all.size());
	deriv_g.deriv.setConstant(DPl);
	std::vector<double> filteredVec;
	#pragma omp parallel for collapse(2)
	for (size_t np = 0; np < nu_p_all.size(); np++) {
		for (size_t ng = 0; ng < nu_g_all.size(); ng++) {
			double nu_p = nu_p_all[np];
			double nu_g = nu_g_all[ng];
			// This is the local Dnu_p which differs from the average Dnu_p because of the curvature. The solver needs basically d(nu_p)/dnp , which is Dnu if O2 terms are 0.
			Dnu_p_local=deriv_p.deriv[np]; 
			DPl_local=DPl; // The solver needs here d(nu_g)/dng. Here we assume no core glitches so that it is the same as DPl. 	
			
			Data_coresolver sols_iter = solver_mm(nu_p, nu_g, Dnu_p_local, DPl_local, q, nu_p - 1.75 * Dnu_p, nu_p + 1.75 * Dnu_p, resol, returns_axis, verbose, fact);
			if (sols_iter.nu_m.size() > 0) {
				for (int i = 0; i < sols_iter.nu_m.size(); i++) {
					if (sols_iter.nu_m[i] >= freq_min && sols_iter.nu_m[i] <= freq_max) {
						#pragma omp critical
						{
							filteredVec.push_back(sols_iter.nu_m[i]);
						}
					}
				}
			}
		}
	}
	std::sort(filteredVec.begin(), filteredVec.end());
	filteredVec.erase(std::unique(filteredVec.begin(), filteredVec.end(), [tol](double a, double b) {
		return std::abs(a - b) <= tol;
	}), filteredVec.end());

	nu_m_all.resize(filteredVec.size());
	std::copy(filteredVec.begin(), filteredVec.end(), nu_m_all.data());
	nu_m_all.resize(filteredVec.size());

	if (returns_pg_freqs == true)
	{
		nu_sols.nu_m=nu_m_all;
		nu_sols.nu_p=nu_p_all;
		nu_sols.nu_g=nu_g_all;
		nu_sols.dnup=deriv_p.deriv;
		nu_sols.dPg=deriv_g.deriv;
		return nu_sols;
	} else
	{
		nu_sols.nu_m=nu_m_all;
		return nu_sols;
	}
}


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
	const long double resol, bool returns_pg_freqs, bool verbose, const long double freq_min, const long double freq_max)
{

	const bool returns_axis=true;
	const int Nmmax=100000; //Ngmax+Npmax;
	const double tol=2*resol; // Tolerance while searching for double solutions of mixed modes
	unsigned seed_p = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine gen_p(seed_p); //, gen_g(seed_g);
	std::normal_distribution<double> distrib_p(0.,sigma_p);

	int np, ng, ng_min, ng_max;//, np_min, np_max, attempts;
	double nu_p, nu_g, Dnu_p, epsilon, Dnu_p_local, DPl_local, fmin, fmax; // Dnu_p_local and DPl_local are important if modes does not follow exactly the asymptotic relation.
	double fact=0.04;  // Default factor

	VectorXi test;
	VectorXd fit, nu_g_all, nu_m_all(Nmmax);	

	Data_coresolver sols_iter;
	Data_eigensols nu_sols;
	Deriv_out deriv_p, deriv_g;

	//tmp=linspace(0, nu_p_all.size()-1, nu_p_all.size());
	const Eigen::VectorXd tmp = Eigen::VectorXd::LinSpaced(nu_p_all.size(), 0, nu_p_all.size()-1);
    
	// There is a problem here because nu_p_all IS NOT for l=0
	// This function need to be updated to include Dnu_p as input.
	fit=linfit_new(tmp, nu_p_all); // fit[0] is the slope ==> Dnu and fit[1] is the ordinate at origin ==> fit[1]/fit[0] = epsilon
	Dnu_p=fit[0];

	fmin=nu_p_all.minCoeff() - Dnu_p; // Range for setting the number of g modes 
	fmax=nu_p_all.maxCoeff() + Dnu_p;

	nu_m_all.setConstant(-9999);
	if (fmin < 0){
		fmin=0;
	}

	ng_min=int(floor(1e6/(fmax*DPl) - alpha));
	ng_max=int(ceil(1e6/(fmin*DPl) - alpha));

	if (fmin <= 150) // overrides of the default factor in case fmin is low
	{
		fact=0.01;
	}
	if (fmin <= 50)
	{
		fact=0.005;
	}

	// Handling the p and g modes, randomized or not
	nu_g_all.resize(ng_max-ng_min);

	for (int ng=ng_min; ng<ng_max;ng++)
	{
		nu_g=asympt_nu_g(DPl, ng, alpha);
		nu_g_all[ng-ng_min]=nu_g;
	}	
	deriv_p=Frstder_adaptive_reggrid(nu_p_all);
	deriv_g.deriv.resize(nu_g_all.size());
	deriv_g.deriv.setConstant(DPl);
	
	std::vector<double> filteredVec;
	#pragma omp parallel for collapse(2)
	for (size_t np = 0; np < nu_p_all.size(); np++) {
		for (size_t ng = 0; ng < nu_g_all.size(); ng++) {
			double nu_p = nu_p_all[np];
			double nu_g = nu_g_all[ng];
			// This is the local Dnu_p which differs from the average Dnu_p because of the curvature. The solver needs basically d(nu_p)/dnp , which is Dnu if O2 terms are 0.
			Dnu_p_local=deriv_p.deriv[np]; 
			DPl_local=DPl; // The solver needs here d(nu_g)/dng. Here we assume no core glitches so that it is the same as DPl. 	
			Data_coresolver sols_iter = solver_mm(nu_p, nu_g, Dnu_p_local, DPl_local, q, nu_p - 1.75 * Dnu_p_local, nu_p + 1.75 * Dnu_p_local, resol, returns_axis, verbose, fact);
			if (sols_iter.nu_m.size() > 0) {
				for (int i = 0; i < sols_iter.nu_m.size(); i++) {
					if (sols_iter.nu_m[i] >= freq_min && sols_iter.nu_m[i] <= freq_max) {
						#pragma omp critical
						{
							filteredVec.push_back(sols_iter.nu_m[i]);
						}
					}
				}
			}
		}
	}
	std::sort(filteredVec.begin(), filteredVec.end());
	filteredVec.erase(std::unique(filteredVec.begin(), filteredVec.end(), [tol](double a, double b) {
		return std::abs(a - b) <= tol;
	}), filteredVec.end());

	nu_m_all.resize(filteredVec.size());
	std::copy(filteredVec.begin(), filteredVec.end(), nu_m_all.data());
	nu_m_all.resize(filteredVec.size());
	
	
	if (returns_pg_freqs == true)
	{
		nu_sols.nu_m=nu_m_all;
		nu_sols.nu_p=nu_p_all;
		nu_sols.nu_g=nu_g_all;
		nu_sols.dnup=deriv_p.deriv;
		nu_sols.dPg=deriv_g.deriv;
		return nu_sols;
	} else
	{
		nu_sols.nu_m=nu_m_all;
		return nu_sols;
	}
}