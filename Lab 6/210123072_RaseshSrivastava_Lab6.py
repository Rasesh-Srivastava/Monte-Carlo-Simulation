import numpy as np
from statistics import NormalDist
from scipy.integrate import quad
PossibleValuesOfM = [100, 1000, 10000, 100000]
ConfidenceInterval = 0.95
def f(u):
    return np.exp(np.sqrt(u))

ExactValueOfI, _ = quad(f, 0, 1)
for m in PossibleValuesOfM:
    print(f"For M = {m}:")
    ObtainedValues = []
    for i in range(m):
        u = np.random.uniform(0, 1)
        ObtainedValues.append(f(u))
    EstimatedValueOfI = np.mean(ObtainedValues)
    print(f"Estimated Value of I: {EstimatedValueOfI}")
    sm = 0
    for i in ObtainedValues:
        sm = sm+(i-EstimatedValueOfI)**2
    sm /= m*(m-1)
    delta = NormalDist().inv_cdf((1+ConfidenceInterval)/2)
    LowerBound = EstimatedValueOfI-(delta*np.sqrt(sm))
    UpperBound = EstimatedValueOfI+(delta*np.sqrt(sm))
    print(f"95% Confidence Interval: ({LowerBound},{UpperBound})\n")

print(f"Exact Value of I: {ExactValueOfI}")
