### 0.2alpha [0%]
	Implementation of solve_mm_asymptotic_O2p() [100%]
	Implementation of a testing function for solve_mm_asymptotic_O2p() [100%]
	Testing comparatively with the python code: [70%] : Crash with Dnu=10 and DP=70 in the C++ code
				Other comment: I think there could be an error on the solver_mm: It takes account of only a single nu_p or nu_g... Might need to account for the supperposed effect of all through gnu() and pnu()... Investigate after being sure that C++ and python have same results (stable reference coding): Could be the source of the phase problem. 

### 0.1alpha [DONE]
	Implementation of solver_mm()
	Implementation of a testing function for sg
	Implementation of a testing function for RGB
	Testing comparatively with the python code: Works well (actually may be even better in terms of precision)
