# What is it?

This is unit tests that evaluate the gain in performance with parallelisation or refactor of several key functions of the ARMM.
Significant gains can be achieved for ksi2_fct and the asymptotic_O2 solvers thanks to the parallelisation. Typically, on a 4 thread, 
a gain of x5 for ksi2_fct and x2 for asymptotic_O2.
The total gain on a full modeled star can be evaluated using the unit_test make_star_test. Gains are:

On a MAC OS 12.5 Apple M1 Pro Chip (10 core max):
Threads       1       2       4       6       8       10
Time (s)    47.66    25.73   13.52   11.62   9.43     9.25 

Reference code without parallelisation and optimisation of key functions: 47.50 seconds.
The refactoring for parallelisation came at an extra cost, which is largely compensated by the gains in optimising linfit(). 
Thus, single-core performance remain the same as without OpenMP, while extra cores show drastic improvments. However, more than 4 threads
may not be worth it.

