// -------------------
// ---- Functions adapted from the bump_DP.py function ----
// This contains all the function that describes the Bumped period spacing
// and how they can be used to derived mode amplitudes from inertia ratio
// as they have been developed and tested. This arises from the following publications:
// https://arxiv.org/pdf/1509.06193.pdf (Inertia and ksi relation)
// https://www.aanda.org/articles/aa/pdf/2015/08/aa26449-15.pdf (Eq. 17 for the rotation - splitting relation)
// https://arxiv.org/pdf/1401.3096.pdf (Fig 13 and 14 for the evolution - rotation relation in SG and RGB) 
// https://arxiv.org/pdf/1505.06087.pdf (Eq. 3 for determining log(g) using Teff and numax, used for getting the evolution stage in conjonction with Fig. 13 from above)
// https://iopscience.iop.org/article/10.1088/2041-8205/781/2/L29/pdf (Inertia and Height relation)
// https://arxiv.org/pdf/1707.05989.pdf (Fig.5, distribution of surface rotation for 361 RGB stars)

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
#include "noise_models.h" // get the harvey_1985 function


/*
# the ksi function as defined in equation 14 of Mosser+2017 (https://arxiv.org/pdf/1509.06193.pdf)
# Inputs: 
# 	nu: The freqency axis (microHz)
# 	nu_p : Frequenc of a p-mode (microHz)
#	nu_g : Frequency of a g-mode (microHz)
#	Dnu_p: Large separation. DOES NOT NEED TO BE A CONSTANT (fonction of np or nu to account for glitches)
#   DPl: Period Spacing (seconds)
#	q : Coupling term (no unit)
*/
VectorXd ksi_fct1(const VectorXd nu, const long double nu_p, const long double nu_g, const long double Dnu_p, const long double DPl, const long double q):
	
	const long double pi = 3.141592653589793238L;
	VectorXd cos_upterm, cos_downterm, front_term, tmp(nu.size());

	tmp.setConstant(1./nu_g);
	cos_upterm=pi * 1e6 * (1./nu.cwiseInverse() - tmp)/DPl;

	tmp.setConstant(nu_p);
	cos_downterm=pi * (nu - nu_p) /Dnu_p;
	front_term= 1e-6 * nu.array().square() * DPl / (q * Dnu_p); // relation accounting for units in Hz and in seconds

	ksi=1./(1. + front_term * cos_upterm.array().cos().square()/cos_downterm.array().cos().square());
	
	return ksi;


/*
# Variant of ksi_fct that deals with arrays for nu_p, nu_g, Dnu_p, DPl
# This requires that nu_p and Dnu_p have the same dimension
# Also nu_g and DPl must have the same dimension
# Additional parameter:
#  - norm-method: When set to 'fast', normalise by the max of the ksi_pg calculated at the
#				   Frequencies nu given by the user
#				   When set to 'exact', normalise by the max of a heavily interpolated function
#				   of ksi_pg. Allows a much a higher precision, but will be slower
#				   This could be usefull as in case of low ng, the norm is badly estimated in
#				   'fast' mode. Then we need to use a more continuous function to evaluate the norm
*/
VectorXd ksi_fct2(const VectorXd nu, const VectorXd nu_p, const VectorXd nu_g, const VectorXd Dnu_p, const VectorXd DPl, const long double q, const std::string norm_method='fast')
{
	const int Lp=nu_p.size();
	const int Lg=nu_g.size();
	const long double resol=1e6/(4*365.*86400.); // Fix the grid resolution to 4 years (converted into microHz)

	VectorXd ksi_pg(nu.size()), nu4norm;

	int Ndata;
	long double norm_coef;

	ksi_pg.setConstant(nu.size());
	for (int np=0; np<Lp;np++)
	{
		for (int ng=0; ng<Lg; ng++)
		{
			ksi_tmp=ksi_fct1(nu, nu_p[np], nu_g[ng], Dnu_p[np], DPl[ng], q);
			ksi_pg=ksi_pg + ksi_tmp;
		}
	}
	if (norm_method == 'fast'){
		norm_coef=max(ksi_pg);
	}
	else{ // We build a very resolved 'continuous function of the frequency to calculate the norm'
		if (min(nu_p) >= min(nu_g)){
			fmin=min(nu_g);
		} else{
			fmin=min(nu_g);
		}
		if (max(nu_p) >= max(nu_g)){
			fmin=max(nu_p);
		} else{
			fmin=max(nu_g);
		}
		Ndata=int((fmax-fmin)/resol);
		nu4norm=linspace(fmin, fmax, Ndata);
		ksi4norm.resize(nu4norm);
		for np in range(Lp):
			for ng in range(Lg):
				ksi_tmp=ksi_fct1(nu4norm, nu_p[np], nu_g[ng], Dnu_p[np], DPl[ng], q);
				ksi4norm=ksi4norm + ksi_tmp;
		norm_coef=max(ksi4norm);
	}
	ksi_pg=ksi_pg/norm_coef;	
	return ksi_pg;
}

VectorXd gamma_l_fct2(const VectorXd ksi_pg, const VectorXd nu_m, const VectorXd nu_p_l0, const VectorXd width_l0, const long double hl_h0_ratio, const int el)
{
	long double width0_at_l;
	VectorXd width_l(ksi_pg.size());

	if (  (nu_p0.size() != width0.size()) || (ksi_pg.size() != nu_m.size()) )
	{
		std::cout << "Inconsistency between the size of the Width and l=0 frequency array or between ksi_pg and nu_m arrays" << std::endl;
		std::cout << "Cannot pursue. The program will exit now" << std::endl;
		exit(EXIT_FAILURE);
	} else
	{
		// Perform the interpolation
		for (int i=0; i<ksi_pg.size(); i++)
		{
			width0_at_l=lin_interpol(nu_p_l0, width_l0, nu_m[i]);
			width_l[i]=width0_at_l * (1. - ksi_pg[i])/ std::sqrt(hl_h0_ratio);
		}
		// ---- DEBUG LINES ----
		std::cout << "   DEBUG FOR gamma_l_fct2..." << std::endl;
		std::cout << "ksi_pg     ,   width_l    ,   hl_h0_ratio" << std::endl;
		for (int i=0; i<ksi_pg.size(); i++)
		{
			std::cout << ksi_pg[i] << "    " << width_l[i] << "    "  << hl_h0_ratio[i] << std::endl;
		}
	}
	return width_l;
}

VectorXd h_l_rgb(const VectorXd ksi_pg)
{
	const double tol=1e-5;
	VectorXi pos;

	hl_h0=sqrt(1. - ksi_pg);
	pos=where_dbl(hl_h0, 0, tol);
	for (int i=0;i<pos.size();i++)
	{
		hl_h0[pos[i]] = 1e-10;
	}
	// --- DEBUG LINES ---
	std::cout << "   DEBUG FOR h_l_rgb..." << std::endl;
	for (int i=0; i<ksi_pg.size(); i++)
	{
		std::cout << ksi_pg[i] << "    " << hl_h0[i] << std::endl;
	}
	return hl_h0;
}

// Put here the code for reading template files that contain heights and width profiles
void read_templatefile(const std::string file){

}

Data_2vectXd width_height_load_rescale(const VectorXd nu_star, const long double Dnu_star, const long double numax_star, const std::string file)
{
	int Nref, Nstar;
	long double epsilon_star, n_at_numax_star, w_tmp, h_tmp;
	VectorXd tmp, tmp_ref, nu_ref, en_list_ref, en_list_star, w_star, h_star;
	Data_2vectXd out;

	template_file template_data;

	template_data= read_templatefile(file);
	nu_ref=template_data.data_ref.col(0); //[:,0] // CHECKS NEEDED REALLY COL OR IS IT ROW?
	height_ref=template_data.data_ref.col(1);//[:,1]
	gamma_ref=template_data.data_ref.col(2);//[:,2]

	height_ref_at_numax=lin_interpol(nu_ref, height_ref, numax_ref);
	gamma_ref_at_numax=lin_interpol(nu_ref, gamma_ref, numax_ref);
	n_at_numax_ref=numax_ref/Dnu_ref - epsilon_ref;
	
	Nref=nu_ref.size();
	tmp.resize(Nref);
	en_list_ref.resize(Nref);

	tmp.setConstant(epsilon_ref);
	en_list_ref=nu_ref/Dnu_ref - tmp; // This list will be monotonic
	// ------------------------------------------------------------------------------------
	// Rescaling using the base frequencies given above for the Sun
	epsilon_star=mean(nu_star/Dnu_star % 1);
	n_at_numax_star=numax_star/Dnu_star - epsilon_star;
	
	Nstar=nu_star.size();
	tmp.resize(Nstar);
	en_list_star.resize(Nstar);

	tmp.setConstant(epsilon_star);
	en_list_star=nu_star/Dnu_star - tmp;

	tmp.resize(n_at_numax_ref.size());
	tmp.setConstant(n_at_numax_ref);
	tmp_ref=en_list_ref - tmp;
	for (int en=0; en<en_list_star.size(); en++){
		w_tmp=lin_interpol(tmp_ref, gamma_ref/gamma_ref_at_numax, en_list_star[en] - n_at_numax_star);
		h_tmp=lin_interpol(tmp_ref, height_ref/height_ref_at_numax, en_list_star[en] - n_at_numax_star); 
		w_star[en]=w_tmp;
		h_star[en]=h_tmp;
	}
	h_star=h_star/max(h_star); // Normalise so that that HNR(numax)=1 if white noise N0=1

	out.vectXd1=w_star;
	out.vectXd2=h_star;
	return out;
}


// Simple way of computing the core rotation from the surface rotation. Used if we want uniform 
// distribution of rotation in the envelope and a uniform population of core-to-envelope ratios 
// 	 (1) rot_envelope: average rotation in the envelope
//	 (2) core2envelope_star: average rotation in the core 
Data_rot2zone rot_2zones_v2(const long double rot_envelope, const long doulbe core2envelope_star, const std::string output_file_rot="")
{

	Data_rot2zone rot2data;

	// Determine the core rotation
	const long double rot_core=core2envelope_star * rot_envelope;
	//std::ostringstream strg;
	std::ofstream outfile;
	if (output_file_rot != "")
	{
		outfile.open(output_file_rot.c_str());
		if (outfile.is_open()){
			//outfile << strg.str().c_str();
			outfile << "#Average envelope rotation (microHz) /  Average core rotation  (microHz)\n";
			outfile << rot_envelope <<  "  " << rot_core;
			outfile.close();
		}
	}

	rot2data.rot_env=rot_envelope;
	rot2data.rot_core=rot_core;

	return rot2data;
}

// Simple way of computing the core rotation from the surface rotation. Used if we want uniform 
// distribution of rotation in the envelope and a uniform population of core rotation
// 	 (1) rot_envelope: average rotation in the envelope
//	 (2) rot_core: average rotation in the core 
Data_rot2zone rot_2zones_v3(const long double rot_envelope, const long double rot_core, const std::string output_file_rot=""):
	
	std::ofstream outfile;
	if (output_file_rot != "")
	{
		outfile.open(output_file_rot.c_str());
		if (outfile.is_open()){
		outfile << "#Average envelope rotation (microHz) /  Average core rotation  (microHz)\n";
		outfile << rot_envelope <<  "  " << rot_core;
		outfile.close();

	rot2data.rot_env=rot_envelope;
	rot2data.rot_core=rot_core;

	return rot2data;

// Function that determine the rotation in the envelope, here approximated to be the surface rotation.
// Inspired by the surface rotation from Ceillier et al. 2017 (https://arxiv.org/pdf/1707.05989.pdf), Fig. 5
// They have a skewed distribution going for ~30 days to ~160 days with a peak around 60 days. 
// For simplicity, I just insert a truncated gaussian distribution with rotation between 30  and 90 and a median of 60.
// The truncation happens at sigma. Values are given in days
// Returns: 
//	rot_s: rotation frequency in microHz
long double rot_envelope(const long double med=60., const long double sigma=3.)
{
	const long double var=30./sigma; // in days
	
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine gen(seed); 
	std::normal_distribution<double> distrib(med,var);
	
	long double period_s=distrib(gen);
	long double rot_s;

	if (period_s < med - var*sigma)
	{
		period_s=med-var*sigma;
	}
	if (period_s > med + var*sigma)
	{
		period_s=med+var*sigma;
	}
	rot_s=1e6/(86400.*period_s);

	return rot_s;
}

/*
# Splitting of modes assuming a two-zone (or two-zone averaged) rotation profile
# rot_envelope and rot_core must be in Hz units (or micro, nano, etc...)
# ksi_pg: The ksi function that describes the degree of mixture between p and g modes in function of the more frequency
# rot_envelope: average rotation in the envelope. Must be a scalar
# rot_core: average rotation in the core. Must be a scalar
# Returns:
#	dnu_rot: A vector of same size as ksi_pg
*/
VectorXd dnu_rot_2zones(const VectorXd ksi_pg, const long double rot_envelope, const long double rot_core)
{
	VectorXd rc(ksi_pg.size()), re(ksi_pg.size());

	rc.setConstant(rot_core/2);
	re.setConstant(rot_envelope);

	return ksi_pg*(rc - re) + re;
}	


/*
# The main function that generate a set of parameters used to generate Lorentzian profiles FOR SIMULATIONS IN THE C++ SIMULATOR
# Assumptions: 
#     - The frequencies of l=0, l=2 and l=3 modes follow exactly the asymtptotic relation for the p mdoes
#	  - The frequencies of the l=1 modes follow exactly the asymptotitc relation for the mixed modes
#	  - Widths of l=0, 2, 3 modes are rescaled using the synthetic relation from Appourchaux+2014 applied to the solar profile
#	  - Widths of l=1 mixed modes are defined using the ksi function, scaled using l=0 modes. Thus l=0 modes fixes the upper
#		limit of the mode width
#	  - Heights of l=0 modes are rescaled using the measured heights of the solar modes
#	  - Heights of l=1 mixed modes are set based on equipartition of energy accounting for damping and excitation assuming no radiative pressure
#		GrosJean+201X showed that this was valid for not-too-evolved RGB and for SG.
#	  - Bolometric visibilities in height are assumed to be 1, 1.5, 0.5, 0.07 for l=0,1,2,3 respectively
#	  - Splitting is implemented in different ways... see the various models
# 	  Warning for widths and heights: if fmin/fmax is too large, you may have an error as an extrapolation 
#									  would be required, which I forbid by code. An fmin/fmax that englobes
#									  10 radial orders should not pose any issue.
# Input: 
#	Dnu_star: Large separation for the p modes
#   epsilon_star: phase offset term for the asymtptotic relation of the p modes
#   delta0l_star, alpha_p_star, nmax_star: Instead of D0_star, these parameters can be used to create 2nd order asymptotic p modes (see Mosser+2018, Eq.22) 
#   DP1_star: The period spacing for l=1 g modes 
#	alpha_star: The phase offset term for the asymptotic relation of the g modes
#   q_star: Coupling coeficient between p and g modes
#	fmin: Minimum frequency for the modes that should be included in the calculations
#   fmax: Maximum frequency for the modes that should be included in the calculations
# Outputs:
#	nu_lx: Frequencies of the l=x modes. x is between 0 and 3
#	nu_p_l1: Base p modes frequencies used to build the frequencies for the l=1 mixed modes
#	nu_g_l1: Base p modes frequencies used to build the frequencies for the l=1 mixed modes
#	width_lx: Widths of the l=x modes. x is between 0 and 3
#   height_lx: Heights of the l=x modes. x is between 0 and 3 
*/
void make_synthetic_asymptotic_star(const Cfg_synthetic_star cfg_star):

	std::random_device rd;
	std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
	std::uniform_real_distribution<double> distrib(xmin,xmax);
	std::uniform_real_distribution<double> distrib(0 , 1);

	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine gen_m(seed); 
	std::normal_distribution<double> distrib_m(0.,cfg_star.sigma_m);


	int el;
	long double tmp, c, xmin ,xmax, delta0l_star;

	VectorXd tmpXd, noise_params_harvey1985(4), Noise_l0, nu_m_l1, height_l1, height_l1p, width_l1; // Simulate a single harvey profile

	Data_2vectXd width_height_l0;
	Data_rot2zone rot2data;
	
	//Defining what should be Hmax_l0 in order to get the desired HNR
	//                   0           1         2           3            4           5          6       7
	//noise_params_harvey_like=[A_Pgran ,  B_Pgran , C_Pgran   ,  A_taugran ,  B_taugran  , C_taugran    , p      N0] // 
	noise_params_harvey1985[0] = cfg_star.noise_params_harvey_like[0] * std::pow(cfg_star.numax_star*1e-6,cfg_star.noise_params_harvey_like[1]) + cfg_star.noise_params_harvey_like[2]; // Granulation Amplitude
	noise_params_harvey1985[1] = cfg_star.noise_params_harvey_like[3] * std::pow(cfg_star.numax_star*1e-6, cfg_star.noise_params_harvey_like[4]) + cfg_star.noise_params_harvey_like[5]; // Granulation timescale (in seconds)
	noise_params_harvey1985[0] = noise_params_harvey1985[0]/noise_params_harvey1985[1];
	noise_params_harvey1985[1]= noise_params_harvey1985[1]/1000.;

	noise_params_harvey1985[2]=cfg_star.noise_params_harvey_like[6];
	noise_params_harvey1985[3]=noise_params[7];

	// Fix the resolution to 4 years (converted into microHz)
	resol=1e6/(4*365.*86400.);

	// ----- l=0 modes -----
	// This section generate l=0 modes following the asymptotic relation of p modes, and make
	// rescaled width and height profiles for the star using the solar width and height profiles

	// Use fmin and fmax to define the number of pure p modes and pure g modes to be considered
	np_min=int(floor(cfg_star.fmin/cfg_star.Dnu_star - cfg_star.epsilon_star));
	np_max=int(ceil(cfg_star.fmax/cfg_star.Dnu_star - cfg_star.epsilon_star));

	np_min=int(floor(np_min - cfg_star.alpha_p_star*std::pow(np_min - nmax, 2) /2.));
	np_max=int(ceil(np_max - cfg_star.alpha_p_star*std::pow(np_max - nmax, 2) /2.)); 

	if (nmin < 1){
		nmin=1;
	}
	for (en=np_min; en<np_max; en++){
		tmp=asympt_nu_p(cfg_star.Dnu_star, en, cfg_star.epsilon_star, 0, 0, cfg_star.alpha_p_star, cfg_star.alpha_p_star);
		nu_l0[en-np_min]=tmp;
	}
	
	std::cout << "cfg_star.Dnu_star=", cfg_star.Dnu_star << std::endl;
	std::cout << "cfg_star.epsilon_star=", cfg_star.epsilon_star << std::endl;
	std::cout << "cfg_star.alpha_p_star=", cfg_star.alpha_p_star << std::endl;
	std::cout << "cfg_star.alpha_p_star=", cfg_star.alpha_p_star << std::endl;
	std::cout << "cfg_star.numax_star=", cfg_star.numax_star << std::endl;
	std::cout << "nu_l0=", nu_l0 << std::endl;

	width_height_l0=width_height_load_rescale(nu_l0, cfg_star.Dnu_star, cfg_star.numax_star, cfg_star.filetemplate); // Function that ensure that Hmax and Wplateau is at numax
	
	Noise_l0.resize(nu_l0.size());
	Noise_l0.setZero();
	Noise_l0=harvey1985(noise_params_harvey1985, nu_l0, Noise_l0, 1); // Iterate on Noise_l0 to update it by putting the noise profile with one harvey profile

	c=1; // This is the ratio of HNR between the reference star and the target simulated star: maxHNR_l0/maxHNR_ref.
	hmax_l0=cfg_star.maxHNR_l0*Noise_l0*c;
	height_l0=height_l0*hmax_l0; // height_l0 being normalised to 1 on width_height_load_rescale, getting the desired hmax_l0 requires just to multiply height_l0 by hmax_l0

	if (std::abs(cfg_star.H0_spread) > 0)
	{
		try
		{
			for (int i=0; i<height_l0.size();i++)
			{
				xmin=height_l0[i]*(1. - std::abs(cfg_star.H0_spread)/100.);
				xmax=height_l0[i]*(1. + std::abs(cfg_star.H0_spread)/100.);
				height_l0[i]=xmin + (xmax-xmin)*distrib(gen);
			}
		}
		catch(...)
		{
			std::cout << "Error debug info:" << std::endl;
			std::cout << "nu_l0 = ", nu_l0 << std::endl;
			std::cout << "hmax_l0 = ", hmax_l0 << std::endl;
			std::cout << "Height_l0: ", height_l0 << std::endl;
			std::cout << "cfg_star.H0_spread: ", cfg_star.H0_spread << std::endl;
			exit(EXIT_FAILURE);
		}
	width_l0=width_l0*cfg_star.Gamma_max_l0;

	// ------- l=1 modes ------
	// Use the solver to get mixed modes
	el=1;
	delta0l_star=-el*(el + 1) * cfg_star.delta0l_percent_star / 100.;
	freqs=solve_mm_asymptotic_O2p(cfg_star.Dnu_star, cfg_star.epsilon_star, el, delta0l_star, cfg_star.alpha_p_star, cfg_star.alpha_p_star, DPl_star, cfg_star.alpha_g_star, cfg_star.q_star, cfg_star.sigma_p, cfg_star.fmin, cfg_star.fmax, resol, false, false);
	
	// Filter solutions that endup at frequencies higher/lower than the nu_l0 because we will need to extrapolate height/widths otherwise...
	posOK=where_in_range(freqs.nu_m, min(nu_l0), max(nu_l0), false);
	nu_m_l1.resize(posOK.size());
	for (int i=0; i<posOK.size();i++)
	{
		nu_m_l1[i]=freqs.nu_m[posOK[i]];
		if (cfg_star.sigma_m !=0) // If requested, we add a random gaussian qty to the mixed mode solution
		{
			r = distrib_m(gen_m);
			nu_m_l1[i]=nu_m_l1+r;
		}
	}

	// Generating widths profiles for l=1 modes using the ksi function
	Dnu_p.resize(freqs.nu_p.size());
	Dnu_p.setConstant(cfg_star.Dnu_star); // MIGHT NEED TO BE ADAPTED IN THE CASE OF THE DERIVATIVE IS NOT DNU ==> CURVATURE
	std::cout << "see comment on Dnu_p before getting further" << std::endl;
	exit(EXIT_SUCCESS);
	DPl.resize(freqs.nu_g.size());
	DPl.setConstant(cfg_star.DP1_star);
	
	ksi_pg=ksi_fct2(nu_m_l1, nu_p_l1, nu_g_l1, Dnu_p, DPl, cfg_star.q_star); // assunme Dnu_p, DPl and q constant
	h1_h0_ratio=h_l_rgb(ksi_pg); // WARNING: Valid assummption only not too evolved RGB stars (below the bump, see Kevin mail 10 August 2019)
//	std::cout << 'Len(h1_h0_ratio):',len(h1_h0_ratio))

	height_l1p.resize(nu_m_l1.size());
	for (int i=0; i<nu_l0.size();i++)
	{
		tmp=lin_interpol(nu_l0, height_l0, nu_m_l1[i]);
		height_l1p[i]=tmp;
	}
	height_l1p=height_l1p*cfg_star.Vl[0];
//	std::cout << 'Len(Height_l1p):', len(height_l1p))
	height_l1=h1_h0_ratio * height_l1p;
	width_l1=gamma_l_fct2(ksi_pg, nu_m_l1, nu_l0, width_l0, h1_h0_ratio, el);

	// Generating splittings with a two-zone averaged rotation rates
	if (cfg_star.rot_env_input >=0)
	{
		cfg_star.Teff_star=-1;
		if (cfg_star.rot_ratio_input > 0)
		{
			rot2data=rot_2zones_v2(cfg_star.rot_env_input, cfg_star.rot_ratio_input, cfg_star.output_file_rot); //rot_env, rot_c
		}
		else
		{
			rot2data=rot_2zones_v3(cfg_star.rot_env_input, cfg_star.rot_core_input, cfg_star.output_file_rot); //rot_env, rot_c
		}
	} else{
		// Determine the envelope rotation
		cfg_star.rot_env_input=rot_envelope(); // Warning: This function has optional arguments
		if (cfg_star.Teff_star >=0)
			std::cout << " The option of determining the rotation rate through Teff_star is not anymore supported in this C++ only code version." << std::endl;
			std::cout << " For that purpose, please use the older version that integrates calls to python to handle mixed modes" << std::endl;
			std::cout << " The program will be terminated" << std::endl;
			exit(EXIT_SUCCESS);
			//rot_env, rot_c, rot_env_true, rot_c_true=rot_2zones_v1(rot_env_input, cfg_star.numax_star, Teff_star, sigma_logg=0.1, randomize_core_rot=True, output_file_rot=output_file_rot)
		{
	}
	a1_l1=dnu_rot_2zones(ksi_pg, rot2data.rot_env, rot2data.rot_core);

	// ------- l=2 modes -----
	el=2;
	delta0l_star=-el*(el + 1) * cfg_star.delta0l_percent_star / 100.;
	nu_l2.resize(nmax-nmin);
	for (int en=nmin; en<nmax;en++)
	{
		tmp=asympt_nu_p(cfg_star.Dnu_star, en, cfg_star.epsilon_star, el, delta0l_star, cfg_star.alpha_p_star, ncfg_star.alpha_p_star);
		nu_l2[en-nmin]=tmp;
	}

	// Filter solutions that endup at frequencies higher/lower than the nu_l0 because we will need to extrapolate height/widths otherwise...
	posOK=where_in_range(nu_l2, min(nu_l0), max(nu_l0), false);
	tmpXd=nu_l2;
	nu_l2.resize(posOK.size());
	height_l2.resize(posOK.size());
	width_l2.resize(posOK.size());
	for (int i=0; i<posOK.size();i++)
	{
		nu_l2[i]=tmpXd[posOK[i]];
		tmp=lin_interpol(nu_l0, height_l0, nu_l2[i]);
		height_l2[i]=tmp*cfg_star.Vl[1];
		tmp=lin_interpol(nu_l0, width_l0, nu_l2[i]);
		width_l2[i]=tmp;		
	}

	// Assume that the l=2 modes are only sensitive to the envelope rotation
	a1_l2.resize(nu_l2.size());
	a1_l2.setConstant(rot_env);

	// ------ l=3 modes ----
	el=3;
	delta0l_star=-el*(el + 1) * cfg_star.delta0l_percent_star / 100.;
	nu_l3.resize(nmax-nmin);
	for (int en=nmin; en<nmax;en++)
	{
		tmp=asympt_nu_p(cfg_star.Dnu_star, en, cfg_star.epsilon_star, el, delta0l_star, cfg_star.alpha_p_star, ncfg_star.alpha_p_star);
		nu_l3[en-nmin]=tmp;
	}
	
	posOK=where_in_range(nu_l3, min(nu_l0), max(nu_l0), false);
	tmpXd=nu_l3;
	nu_l3.resize(posOK.size());
	height_l3.resize(posOK.size());
	width_l3.resize(posOK.size());
	for (int i=0; i<posOK.size();i++)
	{
		nu_l3[i]=tmpXd[posOK[i]];
		tmp=lin_interpol(nu_l0, height_l0, nu_l3[i]);
		height_l3[i]=tmp*cfg_star.Vl[2];
		tmp=lin_interpol(nu_l0, width_l0, nu_l3[i]);
		width_l3[i]=tmp;		
	}

	// Assume that the l=3 modes are only sensitive to the envelope rotation
	a1_l3.resize(nu_l3.size());
	a1_l3.setConstant(rot_env);//=numpy.repeat(rot_env, len(nu_l3))

	// CONTINUE HERE
	
	return nu_l0, nu_p_l1, nu_g_l1, nu_m_l1, nu_l2, nu_l3, width_l0, width_l1, width_l2, width_l3, height_l0, height_l1, height_l2, height_l3, a1_l1, a1_l2, a1_l3

