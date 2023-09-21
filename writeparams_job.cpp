
// Functions adapted or taken from io_star_params.cpp in the Spectrum Simulator
#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include "writeparams_job.h"
using Eigen::VectorXd;
using Eigen::VectorXi;
using Eigen::MatrixXd;


void write_star_mode_params_asympt_model(MatrixXd mode_params, std::string file_out, bool append){

	VectorXi Nchars(17), precision(17);

	std::ofstream outfile;
	Nchars << 5, 20, 20, 20, 16, 16, 16 , 16, 16, 16, 16, 16, 16, 16, 16, 16, 10;
	precision << 1, 10, 10, 10, 10, 10, 10, 10 , 10, 10, 8, 8, 8, 8, 8, 2, 3;
	if(append == false){
		outfile.open(file_out.c_str());	
	} else{
		outfile.open(file_out.c_str(), std::ios::app);	
	}
	if(outfile.is_open()){
		outfile << "# Configuration of mode parameters. This file was generated by write_star_mode_params_aj (write_star_params.cpp)" << std::endl;
		outfile << "# Input mode parameters. degree / freq / H / W / a1  / a2  /  a3  /  a4  /  a5  /  a6  / asymetry / inclination" << std::endl;
		for(int i=0; i<mode_params.rows(); i++){
			for(int j=0;j<mode_params.cols(); j++){
				outfile << std::setw(Nchars[j]) << std::setprecision(precision[j]) << mode_params(i,j);
			}
			outfile << std::endl;
		}
		outfile.close();
	}  
	else {
		file_read_error(file_out);
	}
}

MatrixXd bumpoutputs_2_MatrixXd_with_aj(Params_synthetic_star params, double inc){
	int i, cpt=0;
	int Nt=params.nu_l0.size() + params.nu_m_l1.size() + params.nu_l2.size() + params.nu_l3.size();
	MatrixXd mode_params(Nt, 12);

	for (i=0;i<params.nu_l0.size();i++){
		mode_params(cpt, 0)=0;
		mode_params(cpt, 1)=params.nu_l0[i];
		mode_params(cpt, 2)=params.height_l0[i];
		mode_params(cpt, 3)=params.width_l0[i];
		mode_params(cpt, 4)=0; // a1 is 0 for l=0
		mode_params(cpt, 5)=0; // a2 
		mode_params(cpt, 6)=0; // a3 
		mode_params(cpt, 7)=0; // a4 
		mode_params(cpt, 8)=0; // a5 
		mode_params(cpt, 9)=0; // a6 
        mode_params(cpt, 10)=0; // asym 
		mode_params(cpt, 11)=inc; // inc	
  		cpt=cpt+1;
	}
	for (i=0;i<params.nu_m_l1.size();i++){
		mode_params(cpt, 0)=1;
		mode_params(cpt, 1)=params.nu_m_l1[i];
		mode_params(cpt, 2)=params.height_l1[i];
		mode_params(cpt, 3)=params.width_l1[i];
		mode_params(cpt, 4)=params.a1_l1[i]; // a1 
		mode_params(cpt, 5)=params.a2_l1[i]; // a2 
		mode_params(cpt, 6)=0; // a3 not available for l=1
		mode_params(cpt, 7)=0;// a4 not available for l=1
		mode_params(cpt, 8)=0; // a5 not available for l=1
		mode_params(cpt, 9)=0; // a6 not available for l=1
        mode_params(cpt, 10)=0; // asym is 0 in simulations
		mode_params(cpt, 11)=inc; // inc	
       cpt=cpt+1;
	}
	for (i=0;i<params.nu_l2.size();i++){
		mode_params(cpt, 0)=2;
		mode_params(cpt, 1)=params.nu_l2[i];
		mode_params(cpt, 2)=params.height_l2[i];
		mode_params(cpt, 3)=params.width_l2[i];
		mode_params(cpt, 4)=params.a1_l2[i]; // a1 
		mode_params(cpt, 5)=params.a2_l2[i]; // a2
		mode_params(cpt, 6)=params.a3_l2[i]; // a3
		mode_params(cpt, 7)=params.a4_l2[i]; // a4 
		mode_params(cpt, 8)=0; // a5 not available for l=2
		mode_params(cpt, 9)=0; // a6 not available for l=2
        mode_params(cpt, 10)=0; // asym is 0 in simulations
		mode_params(cpt, 11)=inc; // inc	

		cpt=cpt+1;
	}
	for (i=0;i<params.nu_l3.size();i++){
		mode_params(cpt, 0)=3;
		mode_params(cpt, 1)=params.nu_l3[i];
		mode_params(cpt, 2)=params.height_l3[i];
		mode_params(cpt, 3)=params.width_l3[i];
		mode_params(cpt, 4)=params.a1_l3[i]; // a1 is 0 for l=0
		mode_params(cpt, 5)=params.a2_l3[i]; // a2 
		mode_params(cpt, 6)=params.a3_l3[i]; // a3
		mode_params(cpt, 7)=params.a4_l3[i]; // a4 
		mode_params(cpt, 8)=params.a5_l3[i]; // a5 
		mode_params(cpt, 9)=params.a6_l3[i]; // a6 
        mode_params(cpt, 10)=0; // asym is 0 in simulations
		mode_params(cpt, 11)=inc; // inc	
		cpt=cpt+1;
	}
	return mode_params;
}

void write_range_modes(Cfg_synthetic_star cfg_star, Params_synthetic_star params, std::string output_file, bool append){

	const int Nchars=20;
	const int precision=6;
    std::ofstream outfile;
	if(append == false){
		outfile.open(output_file.c_str());	
	} else{
		outfile.open(output_file.c_str(), std::ios::app);	
	}
	if (outfile.is_open()){
		outfile << "# numax and min and max frequencies for the relevant modes. The third parameter is nmax_star, the position of the curvature for the 2nd order equation of Freqs. This file was generated by io_star_params.cpp" << std::endl;
		outfile << std::setw(Nchars) << std::setprecision(precision) << cfg_star.numax_star;
		outfile << std::setw(Nchars) << std::setprecision(precision) << cfg_star.fmin;
		outfile << std::setw(Nchars) << std::setprecision(precision) << cfg_star.fmax;
		outfile << std::setw(Nchars) << std::setprecision(precision) << cfg_star.nmax_star << std::endl; 
        outfile.close();
	}
	else {
		file_read_error(output_file);
	}
}


void file_read_error(std::string file_out){
		std::cout << " Unable to open file " << file_out << std::endl;	
		std::cout << " Check that the full path exists" << std::endl;
		std::cout << " The program will exit now" << std::endl;
		exit(EXIT_FAILURE);
}

void write_star_noise_params(MatrixXd noise_params, std::string file_out, bool append){

	VectorXi Nchars(3), precision(3);

	std::ofstream outfile;

	Nchars << 16, 16, 16;
	precision << 6, 6, 6;

	if (append == false){
		outfile.open(file_out.c_str());
	} else{
		outfile.open(file_out.c_str(), std::ios::app);
	}
	if(outfile.is_open()){

		outfile << "# Configuration of mode parameters. This file was generated by write_star_mode_params (write_star_params.cpp)" << std::endl;
		outfile << "# Input mode parameters. H0 , tau_0 , p0 / H1, tau_1, p1 / N0. Set at -1 if not used. -2 means that the parameter is not even written on the file (because irrelevant)." << std::endl;
		
		for(int i=0; i<noise_params.rows(); i++){
			for(int j=0;j<noise_params.cols(); j++){
				outfile << std::setw(Nchars[j]) << std::setprecision(precision[j]) << noise_params(i,j);
			}
			outfile << std::endl;
		}
		outfile.close();
	}  
	else {
		file_read_error(file_out);
	}

}

void write_star_l1_roots(Params_synthetic_star params_out, std::string file_out, bool append){
	std::ofstream outfile;

	int Nchars = 16;
	int precision = 8;

	if (append == false){
		outfile.open(file_out.c_str());
	} else{
		outfile.open(file_out.c_str(), std::ios::app);
	}
	if(outfile.is_open()){

		outfile << "# l=1 p and g modes used for the mixed mode computation. This file was generated" << std::endl;
            // ---- Show outputs ----
        for (int en=0; en<params_out.nu_p_l1.size(); en++ )
        {
            outfile <<  "p   1" << std::setw(Nchars) << std::setprecision(precision) << params_out.nu_p_l1[en] << std::endl;
        }
        for (int en=0; en<params_out.nu_g_l1.size(); en++ )
        {
            outfile << "g   1" << std::setw(Nchars) << std::setprecision(precision) << params_out.nu_g_l1[en] << std::endl;
        }
    }
}

void copy_cfg(const std::string& filename, const std::string& outputFilename, bool append) {
    std::ifstream inputFile(filename);
    std::ofstream outputFile(outputFilename, append ? std::ios::app : std::ios::trunc);

    if (inputFile.is_open() && outputFile.is_open()) {
		outputFile << "# Configuration used to generate the stellar pulsations" << std::endl;
        std::string line;
        while (std::getline(inputFile, line)) {
            // Remove trailing white space and tabulation
            //line.erase(line.find_last_not_of(" \t") + 1);
			line =strtrim(line);
            // Check if line starts with "#"
            if (!line.empty() && line[0] != '#') {
                outputFile << line << std::endl;
            }
        }

        inputFile.close();
        outputFile.close();
    } else {
        std::cout << "Failed to open input or output file." << std::endl;
    }
}