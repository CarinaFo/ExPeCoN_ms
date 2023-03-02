import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.stats import norm
from scipy.optimize import curve_fit

# Plot Bayesian model

#x-axis ranges from -5 and 5 with .001 steps
x = np.arange(-5, 5, 0.001)

#define multiple normal distributions
plt.plot(x, norm.pdf(x, 0, 2), label='likelihood')
plt.plot(x, norm.pdf(x, -2, 1), label='prior')
plt.plot(x, norm.pdf(x, -1, 1.5), label='posterior')

#add legend to plot
plt.legend()
plt.show()


# Plot psychometric curve with fake data

intensity = np.array([60, 65, 68, 70, 75, 77, 80, 80, 90, 95, 100, 105, 110,115], dtype=float)
intensity = [a/60 for a in intensity]
prob = np.array([2, 2, 4, 6, 8, 12,20, 25, 25, 29, 29, 29, 30, 30], dtype=float)
p = prob/max(prob)

# psychometric function

def pf(x, alpha, beta):
    return 1. / (1 + np.exp( -(x-alpha)/beta ))

# fitting
par0 = scipy.array([0., 1.]) # use some good starting values, reasonable default is [0., 1.]
par, mcov = curve_fit(pf, intensity, p, par0)
print(par)
plt.plot(intensity, p, 'ro')
plt.plot(intensity, pf(intensity, par[0], par[1]))
plt.axvspan(1.25, 1.5, alpha=0.2)
plt.axhline(0.5, linestyle='dotted')
plt.xlabel('intensity in mA.')
plt.ylabel('detection rate')
plt.show()