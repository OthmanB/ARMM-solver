### 0.5 ###
	Fully operational version with exhaustive testing.

### 0.35alpha [0%] [100%]
	Take back 0.3alpha and test the new implementations regarding random number generators for randomness of nu_p and nu_m 

### 0.3alpha [100%]
	Implementation of all relevant function from bump_DP.py into bump_DP.cpp [100%]
	Testing all the functions internally in the c++ and check the behavior of of the master code creating values [100%]
	Testing comparatively with python code the master functions... BEWARE THAT SMALL BUGS WERE FOUND THAT MAY CHANGE THE RESULTS [100%]
	Note:  I found out that the 'fast' method used to compute ksi_fct2() is not accurate enough. While this inaccuracy could be acceptable
		   for RGB (as the spectrum resolution does not allow us to resolve the modes that are mostly affected by the approximation), it is not
		   the case for SG. After some thought, I decided to just use the 'precise' method, assuming a resolutino of 4 years in the data for defining
		   the normalisation constant. It slower than the fast method, but remains managable in the C++ implementation.
	
### 0.2alpha [100%]
	Implementation of solve_mm_asymptotic_O2p() [100%]
	Implementation of a testing function for solve_mm_asymptotic_O2p() [100%]
	Testing comparatively with the python code: [100%] 

	Code Improvements: The python version of solver_mm had an error: The curvature of the p modes is not properly handled in the python code. The global large separation is used to determine the solutions of tan(\thetap) = tan(\thetag) while it should be the local one, see Eq. 3.31 in Charlotte Gehan thesis (https://tel.archives-ouvertes.fr/tel-02128409/document). Frequencies are barely unchanged during my test but this might be important if large curvatures exists in p modes. 

	Performance Improvements: Limiting the use of conservativeResize by using static arrays.

### 0.1alpha [DONE]
	Implementation of solver_mm()
	Implementation of a testing function for sg
	Implementation of a testing function for RGB
	Testing comparatively with the python code: Works well (actually may be even better in terms of precision)
