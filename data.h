/*
 * data.h
 *
 * Header file that contains all kind of class/structures
 * used to process and/or encapsulate data
 * 
 *  Created on: 22 Feb 2016
 *      Author: obenomar
 */

#pragma once
#include <Eigen/Dense>
#include <string>
#include <vector>

//using namespace std;

using Eigen::MatrixXd;
using Eigen::VectorXi;
using Eigen::VectorXd;


struct Data_coresolver{
	VectorXd nu_m, ysol, nu,pnu, gnu; //
};

struct Data_eigensols{
	VectorXd nu_p, nu_g, nu_m;
};