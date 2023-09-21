
/* 
 *	Simple linear fit algorithm 
 * 
 *  Created on: 11 Oct 2017
 *      Author: obenomar
*/

# pragma once

# include <Eigen/Dense>

using Eigen::VectorXd;

VectorXd linfit_new(const VectorXd& x, const VectorXd& y); // A more optimal way to perform a linear fit