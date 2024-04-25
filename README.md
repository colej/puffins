# puffins
Functions and pipelines for disentangling multiple strictly periodic signals in photometric time series

## Intro
Some nonsense routines that we (Cole Johnston, Nora Eisner, David Hogg) are working on.
The goal is to be able to look for pulsations __under__ the harmonic signature 
introduced by highly non-sinusoidal eclipse signatures in binary systems. 

The first approach is to use a simple fourier series to reconstruct the eclipse signal
with K harmonic components, given some input frequency corresponding to the orbit.
Then, we construct a magic design matrix which is basically (1 cos(x) sin(x)) by 
N data points, and regress ourselves the solution. Turns out, this works well enough
as a first pass!

Next, we want to expand the code to include an arbitrary number of harmonic components
that are determined via Ridge regression. 

Finally, we want to stick this all into the covariance term in a periodogram 
calculation so it happens in the background and you don't have to care about it.
