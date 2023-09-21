#include <Eigen/Dense>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
//#include "../unchanged_code/version_solver.h"
#include "../unchanged_code/data_solver.h"
#include "../unchanged_code/bump_DP.h"
//#include "string_handler.h"
#include "../unchanged_code/interpol.h"
#include "../unchanged_code/noise_models.h" // get the harvey_1985 function
#include "../changed_code/solver_mm_new.h"
#include "../changed_code/ksi2_fct_new.h"

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
#   DPl_star: The period spacing for l=1 g modes 
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
Params_synthetic_star make_synthetic_asymptotic_star_new(Cfg_synthetic_star cfg_star)
{
	std::random_device rd;
	std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
	//std::uniform_real_distribution<double> distrib(xmin,xmax);
	std::uniform_real_distribution<double> distrib(0 , 1);

	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine gen_m(seed); 
	std::normal_distribution<double> distrib_m(0.,cfg_star.sigma_m);


	int en, el, np_min, np_max;
	long double r, tmp, resol, c, xmin ,xmax, delta0l_star;
	VectorXi posOK;
	VectorXd tmpXd, noise_params_harvey1985(4), noise_l0, hmax_l0, Dnu_p, DPl, ksi_pg, h1_h0_ratio;
		//nu_p_l1, nu_g_l1, 
	VectorXd nu_l0, nu_m_l1, nu_l2, nu_l3, 
		height_l0, height_l1, height_l2, height_l3, height_l1p, 
		width_l0, width_l1, width_l2, width_l3,
		a1_l1, a1_l2, a1_l3, // Average rotations
		a2_l1, a2_l2, a2_l3, a4_l2, a4_l3, a6_l3, // Activity or Magnetic effects
		a3_l2, a3_l3, a5_l3; // Latitudinal differential rotation effects

	Data_2vectXd width_height_l0;
	Data_rot2zone rot2data;
	Params_synthetic_star params_out;
	Data_eigensols freqs;

	//Defining what should be Hmax_l0 in order to get the desired HNR
	//                   04           1         2           3            4           5          6       7
	//noise_params_harvey_like=[A_Pgran ,  B_Pgran , C_Pgran   ,  A_taugran ,  B_taugran  , C_taugran    , p      N0] // 
	noise_params_harvey1985[0] = cfg_star.noise_params_harvey_like[0] * std::pow(cfg_star.numax_star*1e-6,cfg_star.noise_params_harvey_like[1]) + cfg_star.noise_params_harvey_like[2]; // Granulation Amplitude
	noise_params_harvey1985[1] = cfg_star.noise_params_harvey_like[3] * std::pow(cfg_star.numax_star*1e-6, cfg_star.noise_params_harvey_like[4]) + cfg_star.noise_params_harvey_like[5]; // Granulation timescale (in seconds)
	noise_params_harvey1985[0] = noise_params_harvey1985[0]/noise_params_harvey1985[1];
	noise_params_harvey1985[1]= noise_params_harvey1985[1]/1000.;

	noise_params_harvey1985[2]=cfg_star.noise_params_harvey_like[6];
	noise_params_harvey1985[3]=cfg_star.noise_params_harvey_like[7];

	// Fix the resolution to 4 years (converted into microHz)
	resol=1e6/(4*365.*86400.);
	// ----- l=0 modes -----
	// This section generate l=0 modes following the asymptotic relation of p modes, and make
	// rescaled width and height profiles for the star using the solar width and height profiles

	// Use fmin and fmax to define the number of pure p modes and pure g modes to be considered
	np_min=int(floor(cfg_star.fmin/cfg_star.Dnu_star - cfg_star.epsilon_star));
	np_max=int(ceil(cfg_star.fmax/cfg_star.Dnu_star - cfg_star.epsilon_star));
	np_min=int(floor(np_min - cfg_star.alpha_p_star*std::pow(np_min - cfg_star.nmax_star, 2) /2.));
	np_max=int(ceil(np_max + cfg_star.alpha_p_star*std::pow(np_max - cfg_star.nmax_star, 2) /2.));  // The minus plus is there because (np_max - nmax_star)^2 is always positive
	
	if (np_min < 1)
	{
		np_min=1;
	}
	nu_l0.resize(np_max-np_min);
	for (en=np_min; en<np_max; en++)
	{
		tmp=asympt_nu_p(cfg_star.Dnu_star, en, cfg_star.epsilon_star, 0, 0, cfg_star.alpha_p_star, cfg_star.nmax_star);
		nu_l0[en-np_min]=tmp;
	}
	width_height_l0=width_height_load_rescale(nu_l0, cfg_star.Dnu_star, cfg_star.numax_star, cfg_star.filetemplate); // Function that ensure that Hmax and Wplateau is at numax
	width_l0=width_height_l0.vecXd1;
	height_l0=width_height_l0.vecXd2;
	noise_l0.resize(nu_l0.size());
	noise_l0.setZero();
	noise_l0=harvey1985(noise_params_harvey1985, nu_l0, noise_l0, 1); // Iterate on Noise_l0 to update it by putting the noise profile with one harvey profile
		
	c=1; // This is the ratio of HNR between the reference star and the target simulated star: maxHNR_l0/maxHNR_ref.
	hmax_l0=cfg_star.maxHNR_l0*noise_l0*c;
	height_l0=height_l0.cwiseProduct(hmax_l0); // height_l0 being normalised to 1 on width_height_load_rescale, getting the desired hmax_l0 requires just to multiply height_l0 by hmax_l0

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
			std::cout << "nu_l0 = " << nu_l0 << std::endl;
			std::cout << "hmax_l0 = " << hmax_l0 << std::endl;
			std::cout << "Height_l0: " << height_l0 << std::endl;
			std::cout << "cfg_star.H0_spread: " << cfg_star.H0_spread << std::endl;
			exit(EXIT_FAILURE);
		}
	}
	width_l0=width_l0*cfg_star.Gamma_max_l0;
	// ------- l=1 modes ------
	// Use the solver to get mixed modes
	el=1;
	delta0l_star=-el*(el + 1) * cfg_star.delta0l_percent_star / 100.;	
	freqs=solve_mm_asymptotic_O2p_new(cfg_star.Dnu_star, cfg_star.epsilon_star, el, delta0l_star, cfg_star.alpha_p_star, cfg_star.nmax_star, cfg_star.DPl_star, 
								  cfg_star.alpha_g_star, cfg_star.q_star, cfg_star.sigma_p, cfg_star.fmin, cfg_star.fmax, resol, true, false);
	// Filter solutions that endup at frequencies higher/lower than the nu_l0 because we will need to extrapolate height/widths otherwise...
	posOK=where_in_range(freqs.nu_m, nu_l0.minCoeff(), nu_l0.maxCoeff(), false);
	nu_m_l1.resize(posOK.size());
	for (int i=0; i<posOK.size();i++)
	{
		nu_m_l1[i]=freqs.nu_m[posOK[i]];
		if (cfg_star.sigma_m !=0) // If requested, we add a random gaussian qty to the mixed mode solution
		{
			r = distrib_m(gen_m);
			nu_m_l1[i]=nu_m_l1[i]+r;
		}
	}
	
	// Generating widths profiles for l=1 modes using the ksi function
	Dnu_p=freqs.dnup;
	DPl=freqs.dPg; 
	ksi_pg=ksi_fct2_new(nu_m_l1, freqs.nu_p, freqs.nu_g, Dnu_p, DPl, cfg_star.q_star, "precise"); //"precise" // assume Dnu_p, DPl and q constant
	h1_h0_ratio=h_l_rgb(ksi_pg, cfg_star.Hfactor); // WARNING: Valid assummption only not too evolved RGB stars (below the bump, see Kevin mail 10 August 2019). Hfactor Added on May 2, 2022
	height_l1p.resize(nu_m_l1.size());
	for (int i=0; i<nu_m_l1.size();i++)
	{
		tmp=lin_interpol(nu_l0, height_l0, nu_m_l1[i]);
		height_l1p[i]=tmp;
	}
	
	height_l1p=height_l1p*cfg_star.Vl[0];
	height_l1=h1_h0_ratio.cwiseProduct(height_l1p);
	width_l1=gamma_l_fct2(ksi_pg, nu_m_l1, nu_l0, width_l0, h1_h0_ratio, el, cfg_star.Wfactor); //Wfactor Added on May 2, 2022

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
		{
			std::cout << " The option of determining the rotation rate through Teff_star is not anymore supported in this C++ only code version." << std::endl;
			std::cout << " For that purpose, please use the older version that integrates calls to python to handle mixed modes" << std::endl;
			std::cout << " The program will be terminated" << std::endl;
			exit(EXIT_SUCCESS);
			//rot_env, rot_c, rot_env_true, rot_c_true=rot_2zones_v1(rot_env_input, cfg_star.numax_star, Teff_star, sigma_logg=0.1, randomize_core_rot=True, output_file_rot=output_file_rot)
		}
	}

	a1_l1=dnu_rot_2zones(ksi_pg, rot2data.rot_env, rot2data.rot_core);
	// ------- l=2 modes -----
	el=2;
	delta0l_star=-el*(el + 1) * cfg_star.delta0l_percent_star / 100.;
	nu_l2.resize(np_max-np_min);
	for (int en=np_min; en< np_max;en++)
	{
		tmp=asympt_nu_p(cfg_star.Dnu_star, en, cfg_star.epsilon_star, el, delta0l_star, cfg_star.alpha_p_star, cfg_star.nmax_star);
		nu_l2[en-np_min]=tmp;
	}
	// Filter solutions that endup at frequencies higher/lower than the nu_l0 because we will need to extrapolate height/widths otherwise...
	posOK=where_in_range(nu_l2, nu_l0.minCoeff(), nu_l0.maxCoeff(), false);
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
	a1_l2.setConstant(rot2data.rot_env);
	
	// ------ l=3 modes ----
	el=3;
	delta0l_star=-el*(el + 1) * cfg_star.delta0l_percent_star / 100.;
	nu_l3.resize(np_max-np_min);
	for (int en=np_min; en<np_max;en++)
	{
		tmp=asympt_nu_p(cfg_star.Dnu_star, en, cfg_star.epsilon_star, el, delta0l_star, cfg_star.alpha_p_star, cfg_star.nmax_star);
		nu_l3[en-np_min]=tmp;
	}
	posOK=where_in_range(nu_l3, nu_l0.minCoeff(), nu_l0.maxCoeff(), false);
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
	a1_l3.setConstant(rot2data.rot_env);//=numpy.repeat(rot_env, len(nu_l3))

	//-----  ADDED ON 13 Sept -----
	// Implementation of the latitudinal differential rotation for outer layers
	// This uses the new substructure env_lat_dif_rot that has all its values initialised to 0 
	// by default. This dummy value is used to identify the different scenarios.
	// Implementation of a3 and a5 is possible in various ways. 
	// However, we recommend using only two situations due to the quality of the available data
	// (unless your a3,a5 values come from eg a rotational model). that is Inside cfg_star.env_lat_dif_rot Either:
	//      - a3_l2, a3_l3 and a5_l3 are set to 0, then this is a case without differential rotation
	//      - a3_l2 = a3_l3 set to some values ~ few percent of a1. And keep a5_l3 =0
	a3_l2.resize(nu_l2.size());
	a3_l3.resize(nu_l3.size());
	a5_l3.resize(nu_l3.size());
	a3_l2.setConstant(cfg_star.env_lat_dif_rot.a3_l2);
	a3_l3.setConstant(cfg_star.env_lat_dif_rot.a3_l3);
	a5_l3.setConstant(cfg_star.env_lat_dif_rot.a5_l3);
	// Implementation of asphericity parameters. As for the differential rotation, the recommendation is 
	// to keep it to 0 (default value), unless you have a model of eg activity and magnetic effects
	a2_l1.resize(nu_m_l1.size());
	a2_l2.resize(nu_l2.size());
	a2_l3.resize(nu_l3.size());
	a4_l2.resize(nu_l2.size());
	a4_l3.resize(nu_l3.size());
	a6_l3.resize(nu_l3.size());
	a2_l1.setConstant(cfg_star.env_aspher.a2_l1);
	a2_l2.setConstant(cfg_star.env_aspher.a2_l2);
	a2_l3.setConstant(cfg_star.env_aspher.a2_l3);
	a4_l2.setConstant(cfg_star.env_aspher.a4_l2);
	a4_l3.setConstant(cfg_star.env_aspher.a4_l3);
	a6_l3.setConstant(cfg_star.env_aspher.a6_l3);
	
	// ----- 

	params_out.nu_l0=nu_l0;
	params_out.nu_p_l1=freqs.nu_p;
	params_out.nu_g_l1=freqs.nu_g;
	params_out.nu_m_l1=nu_m_l1;
	params_out.nu_l2=nu_l2;
	params_out.nu_l3=nu_l3;
	params_out.width_l0=width_l0;
	params_out.width_l1=width_l1;
	params_out.width_l2=width_l2;
	params_out.width_l3=width_l3;
	params_out.height_l0=height_l0;
	params_out.height_l1=height_l1;
	params_out.height_l2=height_l2;
	params_out.height_l3=height_l3;
	params_out.a1_l1=a1_l1;
	params_out.a1_l2=a1_l2;
	params_out.a1_l3=a1_l3;
	params_out.a2_l1=a2_l1;
	params_out.a2_l2=a2_l2;
	params_out.a2_l3=a2_l3;
	params_out.a3_l2=a3_l2;
	params_out.a3_l3=a3_l3;
	params_out.a4_l2=a4_l2;
	params_out.a4_l3=a4_l3;
	params_out.a5_l3=a5_l3;
	params_out.a6_l3=a6_l3;

	return params_out;
}
