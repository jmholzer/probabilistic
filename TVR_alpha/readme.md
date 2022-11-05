See https://github.com/jmholzer/probabilistic-pdfs/issues/6

*So the strategy is:*

1. perform numerical differentiation on the raw data to get 1st derivative, as usual. This will result in a noisy 1st derivative
2. use TVR to differentiate the noisy 1st derivative, to hopefully get a well behaved 2nd derivative
3. We will probably want to fit another model onto the 2nd derivative, for example a natural cubic spline under some constraints. This will ensure that the pdf behaves 'correctly' (e.g. total area = 1, pdf >=0 for all domain)

Note that the results of TVR differentiation is a function of the alpha parameter, which controls the smoothness.
The data for different assets will surely result in different amount of noise in their 1st derivative, so the alpha for one asset may not be applicable to another asset.
However in practice, visually it seems like increasing alpha doesn't have any downsides (see pics). So perhaps we can just bump alpha up to a ridiculously high number.

Actually, alpha=100 is definitely too much as it noticeably shifted the pdf downwards. But alpha 0 -10 doesn't seem to affect the pdf, only smoothes out the jaggedness.
