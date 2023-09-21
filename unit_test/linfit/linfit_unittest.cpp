#define BOOST_TEST_MODULE LinearFitTests
#include <boost/test/included/unit_test.hpp>

#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include "../../unchanged_code/linfit.h"
#include "../../changed_code/linfit.h"


using Eigen::VectorXd;
using namespace std::chrono;

BOOST_AUTO_TEST_CASE(LinearFitTest)
{
    const double tolerance = 1e-6;
    const int numTests = 500;
    const int numPoints = 200;
    std::cout << "Testing output difference between linfit and linfit_old..." << std::endl;
    for (int i = 0; i < numTests; i++) {
        // Generate random x and y values
        VectorXd x = VectorXd::Random(numPoints);
        VectorXd y = VectorXd::Random(numPoints);
        VectorXd fit1 = linfit_new(x, y);
        VectorXd fit2 = linfit(x, y);
        // Check if the solutions are the same within tolerance
        BOOST_CHECK_CLOSE((fit1 - fit2).norm(), 0.0, tolerance);
    }
    // Measure execution time for linfit
    // Generate random x and y values
    VectorXd x = VectorXd::Random(numPoints);
    VectorXd y = VectorXd::Random(numPoints);
    auto start1 = high_resolution_clock::now();
    for (int i = 0; i < numTests; i++) {
        VectorXd fit1 = linfit_new(x, y);
    }
    auto end1 = high_resolution_clock::now();
    auto duration1 = duration_cast<microseconds>(end1 - start1).count();

    // Measure execution time for linfit_old
    auto start2 = high_resolution_clock::now();
    for (int i = 0; i < numTests; i++) {    
        VectorXd fit2 = linfit(x, y);
    }
    auto end2 = high_resolution_clock::now();
    auto duration2 = duration_cast<microseconds>(end2 - start2).count();
    // Compare execution times
    std::cout << "linfit execution time: " << duration1 << " microseconds" << std::endl;
    std::cout << "linfit_old execution time: " << duration2 << " microseconds" << std::endl;;
}
