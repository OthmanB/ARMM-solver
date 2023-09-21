#define BOOST_TEST_MODULE ZetaTests
#include <boost/test/included/unit_test.hpp>

#include <Eigen/Dense>
#include "colors.hpp"
#include <chrono>
#include <cmath>
#include "../../unchanged_code/bump_DP.h"
#include "../../changed_code/ksi2_fct_new.h"

using Eigen::VectorXd;
using namespace std::chrono;

BOOST_AUTO_TEST_CASE(ZetaTest)
{  
    const double tolerance = 1e-6; // in percent
    const int numTests = 100;
    const int numPoints = 20;
    // 
	const int el=1;
    Eigen::VectorXd nu(numPoints);
    std::random_device rd;
    std::mt19937 gen(rd());
    
    long duration_total1=0, duration_total2=0;
    std::cout << "Testing output difference between ksi_fct2 and ksi_fct2_old..." << std::endl;
    for (int i = 0; i < numTests; i++) { 
        // p modes
        long double Dnu_p=std::uniform_real_distribution<>(20., 60.)(gen); //60
        long double epsilon=0.4;
        long double delta0l=0.;//-el*(el + 1) * delta0l_percent / 100.;
        long double beta_p=0.0076;
        // g modes
        long double DPl=std::uniform_real_distribution<>(100., 400.)(gen); // 400
        long double alpha_g=0.;
        long double q=std::uniform_real_distribution<>(0.05, 1.)(gen); // 0.15

        long double fmin=500;
        long double fmax=1000;

        std::uniform_real_distribution<> dis(fmin, fmax);
        // Generate random frequencies nu where the zeta1 function is evaluated
        for (int i = 0; i < numPoints; ++i) {
            nu(i) = dis(gen);
        }

    // Define global Pulsation parameters
        // Parameters for p modes that follow exactly the asymptotic relation of p modes
        // Define the frequency range for the calculation by (1) getting numax from Dnu and (2) fixing a range around numax
        long double beta0=0.263; // according to Stello+2009, we have Dnu_p ~ 0.263*numax^0.77 (https://arxiv.org/pdf/0909.5193.pdf)
        long double beta1=0.77; // according to Stello+2009, we have Dnu_p ~ 0.263*numax^0.77 (https://arxiv.org/pdf/0909.5193.pdf)
        long double nu_max=std::pow(10, log10(Dnu_p/beta0)/beta1);
        long double nmax=nu_max/Dnu_p - epsilon;
        long double alpha_p=beta_p/nmax;
        
        int np_min=int(floor(fmin/Dnu_p - epsilon - el/2 - delta0l));
        int np_max=int(ceil(fmax/Dnu_p - epsilon - el/2 - delta0l));
        np_min=int(floor(np_min - alpha_p*std::pow(np_min - nmax, 2) /2.));
        np_max=int(ceil(np_max + alpha_p*std::pow(np_max - nmax, 2) /2.));
        int ng_min=int(floor(1e6/(fmax*DPl) - alpha_g));
        int ng_max=int(ceil(1e6/(fmin*DPl) - alpha_g));
        // --------------------------------------------
        
        VectorXd nu_p(np_max-np_min);
        VectorXd Dnu_p_vec(np_max - np_min);
        for (int en=np_min; en<np_max;en++){
            nu_p[en-np_min]=asympt_nu_p(Dnu_p, en, epsilon, 1, delta0l, alpha_p, nmax);
            Dnu_p_vec[en-np_min]=Dnu_p;
        }
        VectorXd nu_g(ng_max-ng_min);
        VectorXd DPl_vec(ng_max-ng_min);
        for (int ng=ng_min; ng<ng_max;ng++)
        {
            nu_g[ng-ng_min]=asympt_nu_g(DPl, ng, alpha_g);
            DPl_vec[ng-ng_min]=DPl;
        }
        auto start1 = high_resolution_clock::now();
        VectorXd zeta1 = ksi_fct2_new(nu, nu_p, nu_g, Dnu_p_vec, DPl_vec, q, "precise");
        auto end1 = high_resolution_clock::now();
        auto duration1 = duration_cast<microseconds>(end1 - start1).count();
        duration_total1=duration_total1 + static_cast<long>(duration1); // result in seconds
        auto start2 = high_resolution_clock::now();
        VectorXd zeta2 = ksi_fct2(nu, nu_p, nu_g, Dnu_p_vec, DPl_vec, q, "precise");
        auto end2 = high_resolution_clock::now();
        auto duration2 = duration_cast<microseconds>(end2 - start2).count();
        duration_total2=duration_total2 + static_cast<long>(duration2); // result in seconds
        // Check if the solutions are the same within tolerance
        std::cout << colors::red  << "[" << i << "]" << colors::white << std::flush;
        if ((zeta1 - zeta2).norm() > tolerance/100){
            std::cout << "      Dnu   = " << Dnu_p << std::endl;
            std::cout << "      DPl   = " << DPl << std::endl;
            std::cout << "      q     = " << q << std::endl;
            std::cout << "      zeta1 = " << zeta1.transpose() << std::endl;
            std::cout << "      zeta2 = " << zeta2.transpose() << std::endl;
            std::cout << "      delta = " << (zeta1 - zeta2).norm() << std::endl;    
            BOOST_FAIL("Error: delta greater than the specified tolerance="+dbl_to_str(tolerance)+" % ");
        }
    }
 
    // Compare execution times
    std::cout << "NEW ksi_fct2 total execution time: " << duration_total1/1e6 << " seconds" << std::endl;
    std::cout << "OLD ksi_fct2 total execution time: " << duration_total2/1e6 << " seconds" << std::endl;
}
