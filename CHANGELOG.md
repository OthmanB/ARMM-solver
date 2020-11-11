### 0.3alpha [99%]
	Implementation of all relevant function from bump_DP.py into bump_DP.cpp [100%]
	Testing all the functions internally in the c++ and check the behavior of of the master code creating values [80%]
	Testing comparatively with python code the master functions... BEWARE THAT SMALL BUGS WERE FOUND THAT MAY CHANGE THE RESULTS
	
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
