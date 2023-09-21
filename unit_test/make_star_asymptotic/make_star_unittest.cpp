#define BOOST_TEST_MODULE Make_Star_Tests
#include <boost/test/included/unit_test.hpp>

#include <Eigen/Dense>
#include "colors.hpp"
#include <chrono>
#include <cmath>
#include <random>

#include "../../unchanged_code/configure_make_star.h"
#include "../../unchanged_code/readparams_job.h"
#include "../../unchanged_code/data_solver.h"
#include "../../unchanged_code/bump_DP.h"
#include "../../changed_code/make_asymptotic_star_new.h"
#include "../../unchanged_code/writeparams_job.h"


using Eigen::VectorXd;
using namespace std::chrono;
BOOST_AUTO_TEST_CASE(Make_Star_Test)
{  

    const double tolerance = 1e-6; // in percent
    const int numTests = 100;
    const int numPoints = 20;
    const std::string cfg_file="../config/make_star_test.cfg";
    const std::string params1_fileout="../debug_files/newcode_debug.out";
    const std::string params2_fileout="../debug_files/oldcode_debug.out";
    
    // 
	const int el=1;
    Eigen::VectorXd nu(numPoints);
    std::random_device rd;
    std::mt19937 gen(rd());

    std::unordered_map<std::string, std::string> input_params=readParameterFile(cfg_file);
    Cfg_synthetic_star cfg_star=configure_make_star(input_params);
    
    long duration_total1=0, duration_total2=0;
    std::cout << "Testing output difference between make_star with and without all improvments..." << std::endl;
    for (int i = 0; i < numTests; i++) { 
        // p modes
        cfg_star.Dnu_star=std::uniform_real_distribution<>(20., 60.)(gen); //60
        // g modes
        cfg_star.DPl_star=std::uniform_real_distribution<>(100., 400.)(gen); // 400
        cfg_star.q_star=std::uniform_real_distribution<>(0.05, 1.)(gen); // 0.15
      
        // --------------------------------------------
        auto start1 = high_resolution_clock::now();
        Params_synthetic_star params1=make_synthetic_asymptotic_star_new(cfg_star);
        auto end1 = high_resolution_clock::now();
        auto duration1 = duration_cast<microseconds>(end1 - start1).count();
        duration_total1=duration_total1 + static_cast<long>(duration1); // result in microseconds
        auto start2 = high_resolution_clock::now();
        Params_synthetic_star params2=make_synthetic_asymptotic_star(cfg_star);
        auto end2 = high_resolution_clock::now();
        auto duration2 = duration_cast<microseconds>(end2 - start2).count();
        duration_total2=duration_total2 + static_cast<long>(duration2); // result in microseconds
        // Check if the solutions are the same within tolerance
        double norm_l0=(params1.nu_l0 - params2.nu_l0).norm();
        double norm_pl1=(params1.nu_p_l1 - params2.nu_p_l1).norm();
        double norm_gl1=(params1.nu_g_l1 - params2.nu_g_l1).norm();
        double norm_ml1=(params1.nu_m_l1 - params2.nu_m_l1).norm();
        double norm_pl2=(params1.nu_l2 - params2.nu_l2).norm();
        double norm_pl3=(params1.nu_l3 - params2.nu_l3).norm();
        double norm_Wl0=(params1.width_l0 - params2.width_l0).norm();
        double norm_Wl1=(params1.width_l1 - params2.width_l1).norm();
        double norm_Wl2=(params1.width_l2 - params2.width_l2).norm();
        double norm_Wl3=(params1.width_l3 - params2.width_l3).norm();
        double norm_Hl0=(params1.height_l0 - params2.height_l0).norm();
        double norm_Hl1=(params1.height_l1 - params2.height_l1).norm();
        double norm_Hl2=(params1.height_l2 - params2.height_l2).norm();
        double norm_Hl3=(params1.height_l3 - params2.height_l3).norm();
        double norm_a11=(params1.a1_l1 - params2.a1_l1).norm();
        double norm_a12=(params1.a1_l2 - params2.a1_l2).norm();
        double norm_a13=(params1.a1_l3 - params2.a1_l3).norm();
        double norm_a21=(params1.a2_l1 - params2.a2_l1).norm();
        double norm_a22=(params1.a2_l2 - params2.a2_l2).norm();
        double norm_a23=(params1.a2_l3 - params2.a2_l3).norm();
        double norm_a32=(params1.a3_l2 - params2.a3_l2).norm();
        double norm_a33=(params1.a3_l3 - params2.a3_l3).norm();
        double norm_a42=(params1.a4_l2 - params2.a4_l2).norm();
        double norm_a43=(params1.a4_l3 - params2.a4_l3).norm();
        double norm_a53=(params1.a5_l3 - params2.a5_l3).norm();
        double norm_a63=(params1.a6_l3 - params2.a6_l3).norm();
        const std::vector<std::string> norm_names ={"norm_l0","norm_pl1","norm_gl1","norm_ml1","norm_pl2","norm_pl3",
             "norm_Wl0", "norm_Wl1","norm_Wl2","norm_Wl3","norm_Hl0", "norm_Hl1","norm_Hl2","norm_Hl3",
            "norm_a11","norm_a12","norm_a13","norm_a21","norm_a13","norm_a21",
            "norm_a22","norm_a23","norm_a32","norm_a33","norm_a42","norm_a43",
            "norm_a53","norm_a63"};
        VectorXd norm_all(28);
        norm_all << norm_l0,norm_pl1,norm_gl1,norm_ml1,norm_pl2,norm_pl3,
            norm_Wl0, norm_Wl1,norm_Wl2,norm_Wl3,norm_Hl0, norm_Hl1,norm_Hl2,norm_Hl3,
            norm_a11,norm_a12,norm_a13,norm_a21,norm_a13,norm_a21,
            norm_a22,norm_a23,norm_a32,norm_a33,norm_a42,norm_a43,
            norm_a53,norm_a63;
        std::cout << colors::red  << "[" << i << "]" << colors::white << std::flush;
        for(int k=0;k<norm_all.size();k++){
            if (norm_all[k] > tolerance/100){
                std::cout << std::endl << colors::yellow << "Error with delta=norm_all["<<k<<"] :  " << norm_names[k] << std::endl;
                // Saving outputs of params1 into a file for debug
                MatrixXd mode_params=bumpoutputs_2_MatrixXd_with_aj(params1, cfg_star.inclination); // get the output in a format that can be written with the writting function
                write_range_modes(cfg_star, params1, params1_fileout,false);
                write_star_l1_roots(params1, params1_fileout, true);
                //el, nu, h, w, a1, a2, a3, a4, a5, a6, asym, inc
                write_star_mode_params_asympt_model(mode_params, params1_fileout, true);
                // Saving outputs of params2 into a file for debug
                mode_params=bumpoutputs_2_MatrixXd_with_aj(params2, cfg_star.inclination); // get the output in a format that can be written with the writting function
                write_range_modes(cfg_star, params2, params2_fileout,false);
                write_star_l1_roots(params2, params2_fileout, true);
                //el, nu, h, w, a1, a2, a3, a4, a5, a6, asym, inc
                write_star_mode_params_asympt_model(mode_params, params2_fileout, true);
                std::cout << "      Dnu   = " << cfg_star.Dnu_star << std::endl;
                std::cout << "      DPl   = " << cfg_star.DPl_star << std::endl;
                std::cout << "      q     = " << cfg_star.q_star << std::endl;
                std::cout << "      delta = " << norm_all[k] << std::endl;
                std::cout << "      Read ../debug_files/*.out for further visualisation of discrepancies " << std::endl;
                BOOST_FAIL("Error: delta greater than the specified tolerance="+dbl_to_str(tolerance)+" % ");
            }
        }
    }
 
    // Compare execution times
    std::cout << "New make_star_asymptotic total execution time: " << duration_total1/1e6 << " seconds" << std::endl;
    std::cout << "Old make_star_asymptotic total execution time: " << duration_total2/1e6 << " seconds" << std::endl;
}
