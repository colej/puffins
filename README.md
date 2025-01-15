# puffins
A module for modelling strictly periodic signals in time series, based on [Hogg & Villar, 2021](https://arxiv.org/abs/2101.07256). Most of the linear algebra that is presented here for the regression work is expanded upon in the above text, so we refer the reader there.


### Authors
C. Johnston - Royal Society | University of Surrey / KU Leuven / Max Planck Institute for Astrophysics
D.W. Hogg - New York University / Center for Computational Astrophysics | Flatiron Institute / Max Planck Institute for Astronomy
N.L. Eisner - Tatari Inc. / Center for Computational Astrophysics | Flatiron Institute / Princeton University


## Intro
Periodic (seasonal) signals are ubiquitous to time series data. However, there are a wide variety of (well motivated) methods used by various teams for modelling different periodic signals that make the results often difficult to compare. What's more is that many of these methods are very computationally expensive. __Puffins__ is designed to be a method for modelling the (strictly) periodic signals present in time series data by way of a linear regression model imbued with a feature embedding and weighting. By utilizing a feature embedding and weighting, __Puffins__ models a time series using a Fourier basis and solves for a data driven model that only retains important features due to the feature weighting. 

By providing a uniform basis on which to flexibly model different types of periodic signals, __Puffins__ naturally provides a uniform bases on which we can compare and classify the signals based on the modelled regression coefficients. In addition to providing a modelling frame work, we will also provide a classification framework based on the modelling output that can either used in conjunction with the regression model, independently of it, or not at all.


## Astrophysics application
Developing this package was motivated by modelling the signatures of eclipsing binary stars in astronomical photometric time series data. While there are highly sophisticated, physically motivated, parameteric modelling codes built specifically for modelling eclipsing binary stars to an extremely high precision, they are often high dimensional models, with strong parameter degeneracies, and are imensely computationally costly. While some science cases demand extremely high precision parameter estimates, other workflows such as classification and residual modelling are hampered by the high computation cost of modelling the high amplitude eclipsing binary signal. Thus, we have a clear motive to build __effective__ models to model and remove the eclipse signal from astronomical time series data.


## Mathematical overview
We are using a linear regression to model our signals of interest for a time series of N observations $y_i$; $i\in (0...N-1)$ with $e_i$ associated uncertainties, taken at $t_i$ times. The typical formulation of the regression problem to y is given by:  __Y__ = __X__ __&beta;__, where __X__ is our N x p design matrix, with N predictors and p features, __&beta;__ is the p x 1 coefficient matrix, and __Y__ is the Nx1 target matrix.

Because we know that we're modelling strictly periodic signals, we will embed our features in a Fourier space - thus, our design matrix is no longer N x p, but N x 2K + 1, where K is the number of harmonics used in our harmonic series to model the signal.

Given the disparity of types of astronomical data, we will be dealing with regimes where __i)__ have far more data points than features (N >> p), and __ii)__ in regimes where we have fewer data points than peatures (N < p). As such, we have two forms of the OLS to solve. In the two cases, we use different formulations for the optimal regression:
 - $\mathbf{\hat{\beta}} = \left( \mathbf{X}^T \mathbf{X} \right)^{-1} \mathbf{X}^T \mathbf{Y}$
 - $\mathbf{\hat{\beta}} = \mathbf{X}^T \left( \mathbf{X} \mathbf{X}^T \right)^{-1} \mathbf{Y}$
since the matrix $\left( \mathbf{X}^T \mathbf{X} \right)$ isn't invertable when p >> N.
 
In some situations, we will want to include weights for each data point in the form of inverse square uncertainties, contained in the diagonal matrix __C__. This will lead to:
 - $\mathbf{\hat{\beta}} = \left( \mathbf{X}^T \mathbf{C}^{-1} \mathbf{X} \right)^{-1} \mathbf{X}^T \mathbf{C}^{-1} \mathbf{Y}$
 - $\mathbf{\hat{\beta}} = \left( \mathbf{X}^T \mathbf{C}^{-1} \mathbf{X} \right)^{+} \mathbf{X}^T \mathbf{C}^{-1} \mathbf{Y}$

 Where $^+$ denotes the pseudo inverse.

## Latent space modelling


## Why did we call it puffins?
Well, the naming logic is as follows: Who doesn't like Puffins? No one.