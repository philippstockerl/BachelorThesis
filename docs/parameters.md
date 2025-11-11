# Parameters Documentation

This document provides a detailed overview of all parameters used for the GRF generator and model. The parameters are grouped by their functionality and impact on map generation, uncertainty modeling, and optimization processes.

---

## generator.py
### General Parameters
- **batch_mode** | *boolean*    : Enables batch scenario generation. When **true**, the generator will automatically loop over all combinations of kernels, length scales, variances, and seeds to create multiple scenarios.
 When **false**, it only creates one scenario using the single parameter combination provided.
- **seed** | *int* : Base random seed for reproducibility.
- **num_seeds** | *int*  : Number of different random seeds to test when batch_mode is enabled. 
- **scenario_name** | *string* : When set to **auto**, the generator constructs a name in the following pattern: /Data/scenarios/scenario_{KERNEL}_{SEED}_{GRID_SIZE}x{GRID_SIZE}. Custom scenario name when set to any different **string** value.
- **output_dir** | *string* : Directory where all scenario subfolders are saved to. Standard directory is /Data/scenarios
- **visualize** | *boolean*     : Controls whether a .png of forecast and actual map are to be exported to the set directory.

### Grid Parameters
- **size** | *int*  :   Defines the grid dimension squared. Directly determines spatial resolution of generated maps. Also impacts path lengths and computational complexity (wouldn't try anything over 200!; MacBook Air M2)
- **resolution** | *float* :    Defines distance between grid points. Acts as cell size or unit scale. Implemented for testing. Standard is set to 1.0
- **normalize** | *boolean* : If set to **true**, generated GRF values are normalized to a fixed range or standardized to zero mean. Ensures comparability across kernels, seeds, and parameters sets. If set to **false**, raw GRF data output is used for further processing.
- **range** | *float* : Target cost range after normalization. Lower bound ensures all edge costs remain positive (important for pathfinding). !Test for further: negative to encourage route? e.g. back winds
- **padding_factor** | *float* : Adds extra grid "padding" for FFT-based GRF generation. This should help avoiding wrap-around artifacts ("edge hugging"). Padding factor of 0.2 for example extends the domain by 20% in each direction. The field is then generated on this larger domain and cropped back to the original grid size.

### Field Parameters
- **kernels** | *string* : List of covaraince (kernel) types to test. Each defines how spatial correlation decays with distance. Control the texture of the map.
- **len_scales** | *float* : Length scale determines distance over which two points in space remain correlated. Defines "weather-cell" structure. Test proposal: fine/medium/coarse correlation patterns.
- **variances** | *float* : Sets amplitude of variation in the generated field. Controls how much the field fluctuates around the mean. Flat/ nearly uniform maps vs. expensive/ cheap regions. 
- **means** | *float* : Mean value of the Gaussian field before normalization. Used to shift entire field up or down (simulate baseline costs like wind drag). Kept at 0.0 before rescaling.
- **nu** | *float* : Smoothness parameter for Matern kernel. Defines differentiability (roughness/ smoothness). Intermediate values yield physically realistic textures like turbulent fields.
- **alpha** | *float* : Stability parameter for Stable kernels. Controls tail heaviness and smoothness.
- **beta** | *float* : Shape parameter for Rational Quadratic kernel. Governs multi-scale correlation. 
- **anisotropy** | *boolean* : Enables anisotropic field generation: correlation depends on direction. When **false**, fields are isotropic (same correlation in all directions).
- **ansiotropy_ratio** | *float* : Defines ratio of correlation lengths between x- and y-directions if anisotropy=true. 
- **fft_method** | *boolean* : If **true**, FFT-based spectral synthesis method is used for GRF generation (fast, memory efficient, supports large grids). If **false**, uses direction covariance-based sampling (Cholesky or turning bands). NOTE: thesis is build around FFT. Cholesky requires work around! 
- **cv_levels** | *float* : Defines coefficient of variation (CV) levels. Variance/ Mean. Used to scale the variance adaptively to control uncertainty across scenarios. Each CV defines a seperate scenario level that can be cross-compared between kernels and grid sizes.

### Forecast Parameters
- **blurr_sigma** | *float* : Controls Gaussian blur (smoothing kernel) applied to base GRF to create the forecast map. This creates the deviation costs used in the robust model. Represents limited spatial resolution of forecasts. Actual map with high-frequency variation removed.
- **deviation_factor** | *flaot* : Defines expected fractional deviation between the forecast and the actual values. Scales the perturbations applied to the forecast map to generate the deviation values. 
- **use_difference_based_deviation** | *boolean* :   When **true**, the GRF generator will compute deviations as abs(actual - forecast) instead of using the parametric deviation_factor * nominal_forecast.
- **normalize** | *boolean* : Wether to re-normalize the forecast map after blurring and deviation scaling. Ensures forecast cost map stays comparable to actual map, even after smoothing.
- **cv_adjustment** | *boolean* : When enabled, adjusts forecast field´s variance to maintain the same coefficient of variation (CV) as the base GRF. Keeps relative uncertainty level consistent even after blurring. 
 
### Actual Parameters
- **white_noise_sigma** | *float* : Adds white Gaussian noise to the base GRF to simulate small-scale unpredictable variations. Creates a slightly different but correlated "true" map that D* lite can discover incrementally.
- **normalize** | *boolean* : Re-normalizes the acutal map (after adding white noise) so that both forecast and actual map are in the same cost range.
- **cv_target** | *float* : Defines a target coefficient of variation for the actual field. The generator will scale the actual field´s variance to reach this target CV. Used to maintain controlled uncertainty levels.


## robust.py
- **robustness_factor**: A parameter controlling the robustness of the model against uncertainty.
- **robust_optimization**: Indicates whether robust optimization techniques are enabled.
- **robust_threshold**: Threshold value used to determine robustness constraints.

## dynamic.py
- **dstar_epsilon**: Inflation factor for the heuristic in the D* Lite path planning algorithm.
- **dstar_replan_interval**: Time interval or condition for replanning the path.
- **dstar_max_iterations**: Maximum number of iterations allowed for the D* Lite algorithm.

## evaluation.py
- **evaluation_metrics**: List of metrics used to evaluate the model performance (e.g., RMSE, MAE).
- **evaluation_interval**: Frequency at which evaluations are conducted during experiments.
- **evaluation_seed**: Seed used for evaluation randomness to ensure reproducibility.

---

This structured parameter documentation is intended to facilitate understanding and tuning of the GRF generator and associated models for improved map generation, uncertainty quantification, and optimization performance.
