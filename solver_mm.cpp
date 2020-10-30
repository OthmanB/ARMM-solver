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
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include "version_solver.h"
#include "data.h"
#include "string_handler.h"
#include "interpol.h"

using Eigen::MatrixXd;
using Eigen::VectorXi;
using Eigen::VectorXd;

// Taken from https://stackoverflow.com/questions/27028226/python-linspace-in-c
template<typename T>
std::vector<double> linspace(T start_in, T end_in, int num_in)
{

  std::vector<double> linspaced;

  double start = static_cast<double>(start_in);
  double end = static_cast<double>(end_in);
  double num = static_cast<double>(num_in);

  if (num == 0) { return linspaced; }
  if (num == 1) 
    {
      linspaced.push_back(start);
      return linspaced;
    }

  double delta = (end - start) / (num - 1);

  for(int i=0; i < num-1; ++i)
    {
      linspaced.push_back(start + delta * i);
    }
  linspaced.push_back(end); // I want to ensure that start and end
                            // are exactly the same as the input
  return linspaced;
}
// -----

// My linspace function
VectorXd linspace(const long double start_in, const long double end_in, const long num_in)
{
	if (num_in == 0) {
		std::cout << " num_in in linspace is 0. Cannot create a linspace vector. Returning -1." << std::endl;
		VectorXd linspaced(1);
		linspaced[0]=-1;
		return linspaced;
	}
	VectorXd linspaced(num_in);

	const long double delta = (end_in - start_in) / (num_in - 1);
	for(long i=0 ; i< num_in ; i++){
		linspaced[i]=start_in + delta*i;
	}

	/*std::cout << "start_in =" << start_in << std::endl;
	std::cout << "end_in =" << end_in << std::endl;
	std::cout << "num_in =" << num_in << std::endl;
	std::cout << "delta =" << delta << std::endl;
	std::cout << "linspaced.size() =" << linspaced.size() << std::endl;
	
	for(long i=0; i<linspaced.size(); i++)
	{
		std::cout << linspaced[i] << std::endl;
	}
	std::cout << "linspace needs a thorough test" << std::endl;
	std::cout << "exiting now"<< std::endl;
	exit(EXIT_SUCCESS);
	*/
	return linspaced;
}

// Function that detects sign changes
// If the sign went from + to - tag it with a -1
// If the sign went from - to + tag it with a +1
// If there is no change of sign tag it with a 0
// 0 is dealt as a zone of change of sign as well. eg. if we pass from 0 to 2, then the result is +1
// Inputs:
//    - x: input vector for which we want to know sign changes
//    - return_indices: if true (default), the function returns positions at which the sign changed
//						if false, it returns a vector of size(x)-1 with the tags for the sign changes (or not)
VectorXi sign_change(const VectorXd x, bool return_indices=true)
{
	bool bool_tmp;
	VectorXi s(x.size()-1), pos_s(x.size()-1);
	s.setConstant(0); // Vector of tags for sign changes
	pos_s.setConstant(-1); // Vector of indices
	long i, j=0; 
	for (i = 0; i<x.size()-1; i++)
	{
		if ((   x[i+1]>=0 && x[i] >=0   )|| (  x[i+1]<=0 && x[i]   )) // No sign change case or values are constant at 0
		{
			s[i]=0;
		}
		if (  (x[i+1]>=0 && x[i] <0) || (x[i+1]>0 && x[i] <=0)  ) // Sign change case from - to +
		{
			s[i]=1;
			pos_s[j]=i;
			j=j+1;
		}
		if (  (x[i+1]<=0 && x[i] >0) || (x[i+1]<=0 && x[i] >=0)   ) // Sign change case from + to -
		{
			s[i]=-1;
			pos_s[j]=i;
			j=j+1;
		}
	}
	pos_s.conservativeResize(j); // Discard slots that where not used in the vector
	/*
	std::cout << "sign_change needs a thorough test" << std::endl;
	std::cout << " x.size() = " << x.size() << std::endl; 
	for (i=0; i<x.size()-1; i++)
	{
		std::cout << "[" << i << "] " << x[i] << "  " << x[i+1] << "  " << s[i] << std::endl;
	}
	std::cout << "-----" << std::endl;
	for (i=0; i<pos_s.size(); i++)
	{
		std::cout << pos_s[i] << std::endl;
	}
	std::cout << "exiting now"<< std::endl;
	exit(EXIT_SUCCESS);
	*/
	if (return_indices == true)
	{
		return pos_s;
	} else
	{
		return s;
	}
}


VectorXd pnu_fct(const VectorXd nu, const long double nu_p)
{
	if (nu.size() == 0){
		std::cout << "Vector nu in pnu_fct is of size 0. Cannot pursue." << std::endl;
		std::cout << "The program will exit now" << std::endl;
		exit(EXIT_FAILURE);
	} else{
		VectorXd tmp(nu.size()), pnu(nu.size());
		tmp.setConstant(nu_p);
		pnu=nu - tmp;
		return pnu;
	}
}

long double pnu_fct(const long double nu, const long double nu_p)
{
	long double pnu=nu-nu_p;
	return pnu;
}

VectorXd gnu_fct(const VectorXd nu, const long double nu_g, const long double Dnu_p, const long double DPl, const long double q)
{
	const long double pi = 3.141592653589793238L;
	VectorXd X(nu.size()), gnu(nu.size()), tmp(nu.size());
	tmp.setConstant(nu_g);

	X=pi * (nu.cwiseInverse() - tmp.cwiseInverse())*1e6 / DPl;
	tmp=q*X.array().tan(); //.tan();
	gnu=Dnu_p * tmp.array().atan()/pi; //.atan() / pi;
	return gnu;
}

long double gnu_fct(const long double nu, const long double nu_g, const long double Dnu_p, const long double DPl, const long double q)
{
	const long double pi = 3.141592653589793238L;
	long double X, gnu;
	
	X=pi * (1./nu - 1./nu_g)*1e6 / DPl;
	gnu=Dnu_p * atan(q*tan(X))/pi; //.atan() / pi;
	return gnu;
}

/*
This the main function that solves the mixed mode asymptotic relation
which is of the type p(nu) = g(nu)
This solver specifically solve the case:
      nu - nu_p = Dnu*arctan(q tan(1/(nu DPl) - 1/(nu_g*DPl)))
      It tries to find the intersect between p(nu) and g(nu)
      using an interpolation and over a range of frequency such that nu is in [numin, numax]
Parameters:
	- Mandatory: 
	     nu_p (double) : frequency of a given p-mode (in microHz)
	     nu_g (double): frequency of a given g-mode (in microHz)
	     Dnu_p (double): Large separation for p modes (in microHz)
	     DP1 (double): Period spacing for g modes (in seconds)
	     q (double): Coupling term (no unit, should be between 0 and 1)
	- Optional:
		numin (double): Minimum frequency considered for the solution (in microHz)
		numax (double): Maximum frequency considered for the solution (in microHz)
		resol (double): Base resolution for the interpolated base function. The interpolation may miss solutions 
		       if this is set too low. Typically, the resolution parameter should be higher than the
		       spectral resolution of your spectrum so that all resolved modes should be found.
		       This is also used for creating the nu axis for visualisation (in microHz).
		factor (double): Define how fine will be the new tiny grid used for performing the interpolation. This is important
				to avoid extrapolation (which is forbiden and will result in crash of the code). Typically, the default
				value factor=0.05 can compute mixed modes for frequency down to 80microHz. Going below requires a smaller factor

		returns_axis: If True, returns nu, pnu and gnu (see optional reutrns below). Mainly for debug
Returns a structure with:
	nu_m: An array with all solutions that match p(nu) = g(nu)
	nu (optional): The frequency axis used as reference for finding the intersection
	pnu (optional): The curve for p(nu)
	gnu (optional): The curve g(nu)
*/
Data_coresolver solver_mm(const long double nu_p, const long double nu_g, const long double Dnu_p, const long double DPl, const long double q, 
	const long double numin, const long double numax, const long double resol, const bool returns_axis=false, const bool verbose=false, const long double factor=0.05)
{
	int i, Nsize=0;
	long double range_min, range_max, nu_m_proposed, ratio,  ysol_pnu, ysol_gnu;
	Data_coresolver results;
	VectorXi idx;
	VectorXd nu, pnu, gnu, nu_local, pnu_local, gnu_local, nu_m, ysol_all;

	// Generate a frequency axis that has a fixed resolution and that span from numin to numax
	nu=linspace(numin, numax, long((numax-numin)/resol));
	/*
	std::cout << " numin =" << numin << std::endl;
	std::cout << " numax =" << numax << std::endl;
	std::cout << " resol =" << resol << std::endl;
	std::cout << "long((numax-numin)/resol) = " << long((numax-numin)/resol) << std::endl;
	*/
	// Function p(nu) describing the p modes
	pnu=pnu_fct(nu, nu_p);
	// Function g(nu) describing the g modes 
	gnu=gnu_fct(nu, nu_g, Dnu_p, DPl, q);

	/* Find when p(nu) = g(nu) by looking for solution of p(nu) - g(nu) = 0
	#     Method 1: Direct Interpolation... Works only for single solutions ==> Not used here
	#int_fct = interpolate.interp1d(pnu - gnu, nu)
	#nu_m=int_fct(0)
	#     Method 2: (a) Find indices close to sign changes for p(nu) - g(nu)
	#               (b) Then perform an iterative interpolation in narrow ranges
	#                   near the approximate solutions. How narrow is the range is defined
	#					by the resolution parameter resol, which in this case can be view
	#					as the minimum precision.
	*/
	idx=sign_change(pnu-gnu);	
	/*
	std::cout << " === C++ === " << std::endl;
	std::cout << " nu_p =" << nu_p << std::endl;
	std::cout << " nu_g =" << nu_g << std::endl;
	std::cout << " len(nu)= " << nu.size() << std::endl;
	std::cout << "' len(pnu)= " << pnu.size() << std::endl;
	std::cout << "' len(gnu)= " << gnu.size() << std::endl;
	std::cout << "idx =" << idx << std::endl;
	std::cout << "idx.size() = " << idx.size() << std::endl;
	*/
	for (long ind=0; ind<idx.size();ind++)
	{
		//std::cout << "idx[ind] =" << idx[ind] << std::endl;
		// Define a small local range around each of the best solutions
		range_min=nu[idx[ind]] - 2*resol;
		range_max=nu[idx[ind]] + 2*resol;
		// Redefine nu, pnu and gnu for that local range
		nu_local=linspace(range_min, range_max, long((range_max-range_min)/(resol*factor)));
		pnu_local=pnu_fct(nu_local, nu_p);
		gnu_local=gnu_fct(nu_local, nu_g, Dnu_p, DPl, q);	

		//std::cout << " len(nu_local)= " << nu_local.size() << std::endl;
		//std::cout << " len(pnu_local)= " << pnu_local.size() << std::endl;
		//std::cout << " len(gnu_local)= " << gnu_local.size() << std::endl;

		// Perform the interpolation on the local range and append the solution to the nu_m list
		nu_m_proposed=lin_interpol(pnu_local - gnu_local, nu_local, 0);
		try
		{	
			ysol_gnu=gnu_fct(nu_m_proposed, nu_g, Dnu_p, DPl, q);
			ysol_pnu=pnu_fct(nu_m_proposed, nu_p);
		}
		catch(...)
		{
			std::cout << "Interpolation issue detected. Debuging information:" << std::endl;
			std::cout << "    nu_p: " <<  nu_p << std::endl;
			std::cout << "    nu_g: " <<  nu_g << std::endl;
			std::cout << "    Dnu_p: "<< Dnu_p << std::endl;
			std::cout << "    DPl: "<< DPl << std::endl;
			std::cout << "    q: " << q << std::endl;
			std::cout << "    numin: "<< numin << std::endl;
			std::cout << "    numax: "<< numax << std::endl;
			std::cout << "    resol:"<< resol << std::endl;
			std::cout << "    factor:"<< factor << std::endl;
			std::cout << " ------------" << std::endl;
			std::cout << "range_min/max: "<< range_min << range_max << std::endl;
			std::cout << "  nu_local: "<< nu_local << std::endl;
			std::cout << "  pnu_local: "<< pnu_local << std::endl;
			std::cout << "  gnu_local: "<< gnu_local << std::endl;
			std::cout << " ------------" << std::endl;
			std::cout << " int_fct  ==>  nu_local      /   pnu_local - gnu_local : " << std::endl;
			for (i=0; i<nu_local.size(); i++)
			{
				std::cout << "    " <<  nu_local[i]<< pnu_local[i]-gnu_local[i] << std::endl;
			}
			exit(EXIT_FAILURE);
		}
		ratio=ysol_gnu/ysol_pnu;
		if (verbose == true)
		{	std::cout << "-------"<< std::endl;
			std::cout << "nu_m:"<<  nu_m_proposed<< std::endl;
			std::cout << "Ratio:"<<  ratio << std::endl;
		}
		// Sometimes, the interpolator mess up due to the limits of validity for the atan function
		// The way to keep real intersection is to verify after interpolation that we really
		// have p(nu_m_proposed) = g(nu_m_proposed). We then only keeps solutions that satisfy
		// a precision criteria of 0.1%.
		if ((ratio >= 0.999) && (ratio <= 1.001))
		{
			Nsize=nu_m.size() +1;
			nu_m.conservativeResize(Nsize);
			nu_m[Nsize-1]=nu_m_proposed;
		}
	}
	//ysol_gnu=gnu_fct(nu_m, nu_g, Dnu_p, DPl, q);
	ysol_all=gnu_fct(nu_m, nu_g, Dnu_p, DPl, q);
	
	if (returns_axis == true){
		results.nu_m=nu_m;
		results.ysol=ysol_all;
		results.nu=nu;
		results.pnu=pnu;
		results.gnu=gnu;
		return results;
	}
	else{
		results.nu_m=nu_m;
		results.ysol=ysol_all;
		return results;
	}
}

// Function to test solver_mm()
// This is a typical SG case, with  density of g modes << density of p modes
void test_sg_solver_mm()
{

	Data_coresolver sols;

	const long double Dnu_p=60; //microHz
	const long double DP1= 400; //microHz, typical for a RGB with Dnu_p=10 (see Fig. 1 of Mosser+2014, https://arxiv.org/pdf/1411.1082.pdf)

	// Generate a p-mode that follow exactly the asymptotic relation of p modes
	const long double D0=Dnu_p/100. ;
	const long double epsilon=0.4;
	const long double np=10.;
	const long double nu_p=(np + epsilon + 1./2.)*Dnu_p - 2*D0;

	// Generate a g-mode that follow exactly the asymptotic relation of g modes for a star with radiative core
	const long double ng=10;
	const long double alpha=0;
	const long double nu_g=1e6/(ng*DP1);

	// Use the solver
	const long double q=0.2; // Fix the coupling term
	const long double resol=0.01;
	sols=solver_mm(nu_p, nu_g, Dnu_p, DP1, q, nu_p - Dnu_p/2, nu_p + Dnu_p/2, resol, true, true, 0.05);
	//std::cout << "nu_m: "  <<  sols.nu_m << std::endl;
}


// Function to test solver_mm()
// This is a typical RGB case, with  density of g modes >> density of p modes
void test_rgb_solver_mm()
{
	Data_coresolver sols;

	const long double Dnu_p=15; // microHz
	const long double DP1= 80; // microHz, typical for a RGB with Dnu_p=10 (see Fig. 1 of Mosser+2014, https://arxiv.org/pdf/1411.1082.pdf)

	// Generate a p-mode that follow exactly the asymptotic relation of p modes
	const long double D0=Dnu_p/100.; 
	const long double epsilon=0.4;
	const long double np=10.;
	const long double nu_p=(np + epsilon + 1./2.)*Dnu_p - 2*D0;
	// Generate a g-mode that follow exactly the asymptotic relation of g modes for a star with radiative core
	const long double ng=50;
	const long double alpha=0.01;
	const long double nu_g=1e6/(ng*DP1);

	// Use the solver
	const long double q=0.1; // Fix the coupling term
	const long double resol=0.005;
	sols=solver_mm(nu_p, nu_g, Dnu_p, DP1, q, nu_p - Dnu_p/2, nu_p + Dnu_p/2, resol, true, true, 0.05);
	//std::cout << "nu_m: "  <<  sols.nu_m << std::endl;

}

int main(void)
{
	std::cout << " Testing solver_mm() for the case of an SubGiant..." << std::endl;
	test_sg_solver_mm();
	std::cout << " Testing solver_mm() the case of a RedGiant..." << std::endl;
	test_rgb_solver_mm();
}