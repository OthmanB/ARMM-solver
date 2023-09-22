#include <Eigen/Dense>
#include <iostream>
#include "../unchanged_code/interpol.h"
#include "../unchanged_code/bump_DP.h"
#include "ksi2_fct_new.h"

#ifdef _OPENMP
   #include <omp.h>
#else
   #define omp_get_thread_num() 0
#endif

/*
# Variant of ksi_fct that deals with arrays for nu_p, nu_g, Dnu_p, DPl
# This requires that nu_p and Dnu_p have the same dimension
# Also nu_g and DPl must have the same dimension
# Additional parameter:
#  - norm-method: When set to "fast", normalise by the max of the ksi_pg calculated at the
#				   Frequencies nu given by the user
#				   When set to 'exact', normalise by the max of a heavily interpolated function
#				   of ksi_pg. Allows a much a higher precision, but will be slower
#				   This could be usefull as in case of low ng, the norm is badly estimated in
#				   "fast" mode. Then we need to use a more continuous function to evaluate the norm
*/
Eigen::VectorXd ksi_fct2_new(const Eigen::VectorXd& nu, const Eigen::VectorXd& nu_p, const Eigen::VectorXd& nu_g, const Eigen::VectorXd& Dnu_p, const Eigen::VectorXd& DPl, const long double q, const std::string norm_method)
{
    const int Lp = nu_p.size();
    const int Lg = nu_g.size();
    const long double resol = 1e6 / (4 * 365. * 86400.); // Fix the grid resolution to 4 years (converted into microHz)
    Eigen::VectorXd ksi_tmp, ksi_pg(nu.size()), nu_highres, ksi_highres;
    int Ndata;
    long double norm_coef, fmin, fmax;
    ksi_pg.setZero();
    if (norm_method == "fast"){
		for (int np = 0; np < Lp; np++){
			for (int ng = 0; ng < Lg; ng++){
				ksi_tmp = ksi_fct1(nu, nu_p[np], nu_g[ng], Dnu_p[np], DPl[ng], q);
				ksi_pg = ksi_pg + ksi_tmp;
			}
		}
        norm_coef = ksi_pg.maxCoeff();
		ksi_pg = ksi_pg / norm_coef;
        // Ensuring that round-off errors don't lead to values higher than 1...
        for (int i = 0; i < ksi_pg.size(); i++){
            if (ksi_pg[i] > 1){
                ksi_pg[i] = 1;
            }
        }
    } else{
        ksi_pg=ksi_fct2_precise(nu, nu_p, nu_g, Dnu_p, DPl, q);
    }
    return ksi_pg;
}

Eigen::VectorXd ksi_fct2_precise(const Eigen::VectorXd& nu, const Eigen::VectorXd& nu_p, const Eigen::VectorXd& nu_g, const Eigen::VectorXd& Dnu_p, const Eigen::VectorXd& DPl, const long double q)
{
	// A slightly more optimized version since 17 Sept 2023. 10% performance increase + proper omp implementation
    const int Lp = nu_p.size();
    const int Lg = nu_g.size();
    const long double resol = 1e6 / (4 * 365. * 86400.); // Fix the grid resolution to 4 years (converted into microHz)
    const long double fmin = (nu_p.minCoeff() >= nu_g.minCoeff()) ? nu_g.minCoeff() : nu_p.minCoeff();
    const long double fmax = (nu_p.maxCoeff() >= nu_g.maxCoeff()) ? nu_p.maxCoeff() : nu_g.maxCoeff();
    const int Ndata = int((fmax - fmin) / resol);
    const Eigen::VectorXd nu_highres = Eigen::VectorXd::LinSpaced(Ndata, fmin, fmax);
    Eigen::VectorXd ksi_pg(nu.size());//, ksi_tmp(nu.size());
    ksi_pg.setZero();
    #pragma omp parallel for shared(ksi_pg)
    for (int np = 0; np < Lp; np++)
    {
        Eigen::VectorXd ksi2_local(nu.size());
        ksi2_local.setZero();
        for (int ng = 0; ng < Lg; ng++){           
            // The function of interest
            ksi2_local += ksi_fct1(nu, nu_p[np], nu_g[ng], Dnu_p[np], DPl[ng], q); //ksi_fct1(nu, nu_p, nu_g, Dnu_p, DPl, q, np, ng);
        }
        #pragma omp critical
        {
            ksi_pg += ksi2_local;
        }  
    }
    Eigen::VectorXd ksi_highres(nu_highres.size());
    ksi_highres.setZero();
    #pragma omp parallel for shared(ksi_highres)
    for (int np = 0; np < Lp; np++)
    {
        Eigen::VectorXd ksi2_highres_local(nu_highres.size());
        ksi2_highres_local.setZero();
        for (int ng = 0; ng < Lg; ng++){           
            // The function of interest
            ksi2_highres_local += ksi_fct1(nu_highres, nu_p[np], nu_g[ng], Dnu_p[np], DPl[ng], q);//ksi_fct1(nu_highres, nu_p, nu_g, Dnu_p, DPl, q, np, ng);
        }
        #pragma omp critical
        {
            ksi_highres += ksi2_highres_local;
        }
    }
    //#pragma omp barrier # Removed on 22 Sept 2023: When integrated in CPP TAMCMC, this causes a stall of the MCMC. And it is in fact not usefull
    const long double norm_coef = ksi_highres.maxCoeff();
    ksi_pg=ksi_pg/norm_coef;
    
	#pragma omp parallel for
    for (int i = 0; i < nu.size(); i++)
    {
        if (ksi_pg[i] > 1)
        {
            ksi_pg[i] = 1;
        }
    }  
    return ksi_pg;
}
