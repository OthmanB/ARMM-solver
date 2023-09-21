#define BOOST_TEST_MODULE LinearFitTests
#include <boost/test/included/unit_test.hpp>

#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include "../../unchanged_code/interpol.h"
#include "../../changed_code/interpol.h"

using Eigen::VectorXd;
using namespace std::chrono;

BOOST_AUTO_TEST_CASE(LinearInterpTest)
{
    long duration_total1=0, duration_total2=0;

    const double tolerance = 1e-6;
    const int numTests = 50000;
    const int numPoints = 200;
    std::cout << "Testing output difference between lin_interpol_new and lin_interpol..." << std::endl;
    VectorXd x = VectorXd::LinSpaced(numPoints, 0, 99);
    VectorXd y = x.array().square();
    VectorXd y_int1(numPoints), y_int2(numPoints);
    for (int i = 0; i < numTests; i++) {
        VectorXd x_int = VectorXd::Random(numPoints);
        auto start1 = high_resolution_clock::now();
        for(int j=0; j< numPoints;j++){
            y_int1[j] = lin_interpol_new(x, y, x_int[j]);
        }
        auto end1 = high_resolution_clock::now();
        auto duration1 = duration_cast<microseconds>(end1 - start1).count();
        duration_total1=duration_total1 + static_cast<long>(duration1); // result in seconds

        auto start2 = high_resolution_clock::now();        
        for (int j=0; j<numPoints;j++){
            y_int2[j] = lin_interpol(x, y, x_int[j]);
        }
        auto end2 = high_resolution_clock::now();
        auto duration2 = duration_cast<microseconds>(end2 - start2).count();
        duration_total2=duration_total2 + static_cast<long>(duration2); // result in seconds
        
        // Check if the solutions are the same within tolerance
        BOOST_CHECK_CLOSE((y_int1 - y_int2).norm(), 0.0, tolerance);
    }   
    // Compare execution times
    std::cout << "lin_interpol execution time: " << duration_total1 << " microseconds" << std::endl;
    std::cout << "lin_interpol_old execution time: " << duration_total2 << " microseconds" << std::endl;;
}
