#define BOOST_TEST_MODULE SolverTests
#include <boost/test/included/unit_test.hpp>

#include <Eigen/Dense>
#include "colors.hpp"
#include <chrono>
#include <random>
#include <iostream>
#include "solver_mm_new.h"

using Eigen::VectorXd;
using namespace std::chrono;

BOOST_AUTO_TEST_CASE(Asymptotic_O2p_Test)
{  
    const double tolerance = 1e-6; // in percent
    const int numTests = 10;
    const int numPoints = 30;
    // 
	const int el=1;
    const double resol=1e6/(4.*365.*86400.);
    const double factor=0.05;
    std::random_device rd;
    std::mt19937 gen(rd());
    
    long duration_total1=0, duration_total2=0;
    std::cout << "Testing output difference between solver_mm_asymptotic_O2 new and old..." << std::endl;
    for (int i = 0; i < numTests; i++) { 
        // p modes
        long double Dnu_p=std::uniform_real_distribution<>(3., 10.)(gen); //60
        long double epsilon=0.4;
        long double delta0l=0.;//-el*(el + 1) * delta0l_percent / 100.;
        long double beta_p=0.0076;
        // g modes
        long double DPl=std::uniform_real_distribution<>(100., 400.)(gen); // 400
        long double alpha_g=0.;
        long double q=std::uniform_real_distribution<>(0.05, 1.)(gen); // 0.15

        long double fmin=50;
        long double fmax=300;

        // Define global Pulsation parameters
        // Parameters for p modes that follow exactly the asymptotic relation of p modes
        // Define the frequency range for the calculation by (1) getting numax from Dnu and (2) fixing a range around numax
        long double beta0=0.263; // according to Stello+2009, we have Dnu_p ~ 0.263*numax^0.77 (https://arxiv.org/pdf/0909.5193.pdf)
        long double beta1=0.77; // according to Stello+2009, we have Dnu_p ~ 0.263*numax^0.77 (https://arxiv.org/pdf/0909.5193.pdf)
        long double nu_max=std::pow(10, log10(Dnu_p/beta0)/beta1);
        long double nmax=nu_max/Dnu_p - epsilon;
        long double alpha_p=beta_p/nmax;
        // --------------------------------------------
        
        auto start1 = high_resolution_clock::now();
        Data_eigensols sols_new =solve_mm_asymptotic_O2p_new(Dnu_p,  epsilon, el, delta0l,  alpha_p, nmax, 
               DPl, alpha_g, q, 0, fmin, fmax, resol);
        auto end1 = high_resolution_clock::now();
        auto duration1 = duration_cast<microseconds>(end1 - start1).count();
        duration_total1=duration_total1 + static_cast<long>(duration1); // result in seconds
        auto start2 = high_resolution_clock::now();
        Data_eigensols sols_old =solve_mm_asymptotic_O2p(Dnu_p,  epsilon, el, delta0l,  alpha_p, nmax, 
                DPl, alpha_g, q, 0, fmin, fmax, resol);
        auto end2 = high_resolution_clock::now();
        auto duration2 = duration_cast<microseconds>(end2 - start2).count();
        duration_total2=duration_total2 + static_cast<long>(duration2); // result in seconds
        // Check if the solutions are the same within tolerance
       /*
        std::cout << colors::red  << "[" << i << "]" << colors::white << std::endl;
        std::cout << "      Dnu   = " << Dnu_p << std::endl;
        std::cout << "      DPl   = " << DPl << std::endl;
        std::cout << "      q     = " << q << std::endl;
        std::cout << "      sols_new.nu_m = " << sols_new.nu_m.transpose() << std::endl;
        std::cout << "      sols_old.nu_m = " << sols_old.nu_m.transpose() << std::endl;
        std::cout << "      delta = " << (sols_new.nu_m - sols_old.nu_m).norm() << std::endl;
        */
        if ((sols_new.nu_m - sols_old.nu_m).norm() > tolerance/100){
            BOOST_FAIL("Error: delta greater than the specified tolerance="+dbl_to_str(tolerance)+" % ");
        }
        }
    // Compare execution times
    std::cout << "Solver new total execution time: " << duration_total1/1e6 << " seconds" << std::endl;
    std::cout << "Solver old total execution time: " << duration_total2/1e6 << " seconds" << std::endl;
}

BOOST_AUTO_TEST_CASE(Asymptotic_O2from_l0_Test)
{  
    const double tolerance = 1e-6; // in percent
    const int numTests = 10;
    const int numPoints = 30;
    // 
	const int el=1;
    const double resol=1e6/(4.*365.*86400.);
    const double factor=0.05;
    std::random_device rd;
    std::mt19937 gen(rd());
    
    long duration_total1=0, duration_total2=0;
    std::cout << colors::yellow << "Testing output difference between solver_mm_asymptotic_O2from_l0 new and old..." << colors::white << std::endl;
    for (int i = 0; i < numTests; i++) { 
        // p modes
        long double Dnu_p=std::uniform_real_distribution<>(3., 10.)(gen); //60
        long double epsilon=0.4;
        long double delta0l=0.;//-el*(el + 1) * delta0l_percent / 100.;
        long double beta_p=0.0076;
        // g modes
        long double DPl=std::uniform_real_distribution<>(100., 400.)(gen); // 400
        long double alpha_g=0.;
        long double q=std::uniform_real_distribution<>(0.05, 1.)(gen); // 0.15

        long double fmin=50;
        long double fmax=300;

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
        // --------------------------------------------
        VectorXd nu_l0(np_max-np_min);
        for (int en=np_min; en<np_max;en++){
            nu_l0[en-np_min]=asympt_nu_p(Dnu_p, en, epsilon, 0, 0, alpha_p, nmax);
        }

        auto start1 = high_resolution_clock::now();
        Data_eigensols sols_new =solve_mm_asymptotic_O2from_l0_new(nu_l0, el, delta0l, DPl, alpha_g,  q,
                0, resol, false, false, fmin, fmax);
        auto end1 = high_resolution_clock::now();
        auto duration1 = duration_cast<microseconds>(end1 - start1).count();
        duration_total1=duration_total1 + static_cast<long>(duration1); // result in seconds
        auto start2 = high_resolution_clock::now();
     
        Data_eigensols sols_old=solve_mm_asymptotic_O2from_l0(nu_l0, el, delta0l, DPl, alpha_g,  q,
                0, resol, false, false, fmin, fmax);
        auto end2 = high_resolution_clock::now();
        auto duration2 = duration_cast<microseconds>(end2 - start2).count();
        duration_total2=duration_total2 + static_cast<long>(duration2); // result in seconds
        // Check if the solutions are the same within tolerance
        /*
        std::cout << colors::red  << "[" << i << "]" << colors::white << std::endl;
        std::cout << "      Dnu   = " << Dnu_p << std::endl;
        std::cout << "      DPl   = " << DPl << std::endl;
        std::cout << "      q     = " << q << std::endl;
        std::cout << "      sols_new.nu_m = " << sols_new.nu_m.transpose() << std::endl;
        std::cout << "      sols_old.nu_m = " << sols_old.nu_m.transpose() << std::endl;
        std::cout << "      delta = " << (sols_new.nu_m - sols_old.nu_m).norm() << std::endl;
        */
        if ((sols_new.nu_m - sols_old.nu_m).norm() > tolerance/100){
            BOOST_FAIL("Error: delta greater than the specified tolerance="+dbl_to_str(tolerance)+" % ");
        }
        }
    // Compare execution times
    std::cout << "Solver new total execution time: " << duration_total1/1e6 << " seconds" << std::endl;
    std::cout << "Solver old total execution time: " << duration_total2/1e6 << " seconds" << std::endl;
}


BOOST_AUTO_TEST_CASE(Asymptotic_O2from_l0_nupl_Test)
{  
    const double tolerance = 1e-6; // in percent
    const int numTests = 10;
    const int numPoints = 30;
    // 
	const int el=1;
    const double resol=1e6/(4.*365.*86400.);
    const double factor=0.05;
    std::random_device rd;
    std::mt19937 gen(rd());
    
    long duration_total1=0, duration_total2=0;
    std::cout << colors::yellow << "Testing output difference between solver_mm_asymptotic_O2from_nupl new and old..." << colors::white << std::endl;
    for (int i = 0; i < numTests; i++) { 
        // p modes
        long double Dnu_p=std::uniform_real_distribution<>(3., 10.)(gen); //60
        long double epsilon=0.4;
        long double delta0l=0.;//-el*(el + 1) * delta0l_percent / 100.;
        long double beta_p=0.0076;
        // g modes
        long double DPl=std::uniform_real_distribution<>(100., 400.)(gen); // 400
        long double alpha_g=0.;
        long double q=std::uniform_real_distribution<>(0.05, 1.)(gen); // 0.15

        long double fmin=50;
        long double fmax=300;

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
        // --------------------------------------------
        VectorXd nu_p_all(np_max-np_min);
        for (int en=np_min; en<np_max;en++){
            nu_p_all[en-np_min]=asympt_nu_p(Dnu_p, en, epsilon, el, 0, alpha_p, nmax);
        }

        auto start1 = high_resolution_clock::now();
        Data_eigensols sols_new= solve_mm_asymptotic_O2from_nupl_new(nu_p_all, el,
                    DPl, alpha_g,  q, 0, resol, false, false, fmin,  fmax);
        auto end1 = high_resolution_clock::now();
        auto duration1 = duration_cast<microseconds>(end1 - start1).count();
        duration_total1=duration_total1 + static_cast<long>(duration1); // result in seconds
        auto start2 = high_resolution_clock::now();
     
        Data_eigensols sols_old= solve_mm_asymptotic_O2from_nupl(nu_p_all, el,
                    DPl, alpha_g,  q, 0, resol, false, false, fmin,  fmax);
        auto end2 = high_resolution_clock::now();
        auto duration2 = duration_cast<microseconds>(end2 - start2).count();
        duration_total2=duration_total2 + static_cast<long>(duration2); // result in seconds
        // Check if the solutions are the same within tolerance
        /*
        std::cout << colors::red  << "[" << i << "]" << colors::white << std::endl;
        std::cout << "      Dnu   = " << Dnu_p << std::endl;
        std::cout << "      DPl   = " << DPl << std::endl;
        std::cout << "      q     = " << q << std::endl;
        std::cout << "      sols_new.nu_m = " << sols_new.nu_m.transpose() << std::endl;
        std::cout << "      sols_old.nu_m = " << sols_old.nu_m.transpose() << std::endl;
        std::cout << "      delta = " << (sols_new.nu_m - sols_old.nu_m).norm() << std::endl;
        */
       if ((sols_new.nu_m - sols_old.nu_m).norm() > tolerance/100){
            BOOST_FAIL("Error: delta greater than the specified tolerance="+dbl_to_str(tolerance)+" % ");
        }
        }
    // Compare execution times
    std::cout << "Solver new total execution time: " << duration_total1/1e6 << " seconds" << std::endl;
    std::cout << "Solver old total execution time: " << duration_total2/1e6 << " seconds" << std::endl;
}
