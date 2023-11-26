import numpy as np
import matplotlib.pyplot as GraphPlotter
import random
import math
np.random.seed()
# function to generate random variable with exponential distribution with corresponding mean for each process
def T(n):
    mean = 0
    if n == 1:
        mean = 4
    if n == 2:
        mean = 4
    if n == 3:
        mean = 2
    if n == 4:
        mean = 5
    if n == 5:
        mean = 2
    if n == 6:
        mean = 3
    if n == 7:
        mean = 2
    if n == 8:
        mean = 3
    if n == 9:
        mean = 2
    if n == 10:
        mean = 2
    u = random.random()
    x = -mean*math.log(u)
    return x


# array to store the values of E10
values = []

# variable to keep track of how many times E10>70
count2 = 0

# geneerating value of E10 10,000 times
for i in range(0, 10000):
    E1 = T(1)
    E2 = E1+T(2)
    E3 = E1+T(3)
    E4 = E2+T(4)
    E5 = E2+T(5)
    E6 = E3+T(6)
    E7 = E3+T(7)
    E8 = E3+T(8)
    E9 = max(E5, E6, E7)+T(9)
    E10 = max(E4, E8, E9)+T(10)
    values.append(E10)
    if E10 > 70:
        count2 += 1


def calculate_prob(E):
    count = 0
    for i in E:
        if i > 70:
            count += 1
    return count / len(E)


def calculate_std(E):
    mu = calculate_prob(E)
    var = 0
    for i in E:
        y = 0
        if i > 70:
            y = 1
        var += (y - mu) ** 2
    var /= (len(E) - 1)
    std = np.sqrt(var)
    return std


def GenerateExponential(theta):
    lambda_parameter = 1/theta
    u = np.random.uniform()
    X = (-np.log(1-u)/lambda_parameter)
    return X


def generate_E10(theta, n):
    E_10 = list()
    count = 0
    for i in range(n):
        T1 = GenerateExponential(theta[0])
        T2 = GenerateExponential(theta[1])
        T3 = GenerateExponential(theta[2])
        T4 = GenerateExponential(theta[3])
        T5 = GenerateExponential(theta[4])
        T6 = GenerateExponential(theta[5])
        T7 = GenerateExponential(theta[6])
        T8 = GenerateExponential(theta[7])
        T9 = GenerateExponential(theta[8])
        T10 = GenerateExponential(theta[9])

        E1 = T1
        E2 = T1 + T2
        E3 = T1 + T3
        E4 = T1 + T2 + T4
        E5 = T1 + T2 + T5
        E6 = T1 + T3 + T6
        E7 = T1 + T3 + T7
        E8 = T1 + T3 + T8
        E9 = max(E5, E6, E7) + T9
        E10 = max(E4, E8, E9) + T10

        E_10.append(E10)
        if E10 > 70:
            count += 1
    return E_10, count


n = 10000
theta = [4, 4, 2, 5, 2, 3, 2, 3, 2, 2]

E_10, count = generate_E10(theta, n)
mean_E10 = np.mean(E_10)
var_E10 = np.var(E_10)
variance_E10_monte = var_E10
# Part (b)
print(
    f'Taking n = 10000, The sample mean of E_10 using simple Monte Carlo = {mean_E10}')

# Part (c)
GraphPlotter.hist(E_10, bins=80, alpha=0.6, color='red',
                  label='Histogram of E_10 Values')
GraphPlotter.ylabel('Frequency')
GraphPlotter.xlabel('Values of E_10')
GraphPlotter.title(f'Histogram of generated values of E_10')
GraphPlotter.legend()
GraphPlotter.show()

# Part (d)
print(f'The observed Standard Deviation of E_10 = {np.sqrt(var_E10)}')
print(
    f'Approximate value of the probability that the project misses the deadline using simple Monte Carlo = {count/n}')
print("Standard deviation of probability = ", calculate_std(E_10))
print()

# Part (e)
lambda_ = list()

for i in theta:
    lambda_.append(i*4)

E_10, count = generate_E10(lambda_, n)
mean_E10 = np.mean(E_10)
var_E10 = np.var(E_10)
sample_size = variance_E10_monte/var_E10 * n
print('Using Importance Sampling Technique:')
print(f'The observed mean of E_10 = {mean_E10}')
print(f'The observed standard deviation of E_10 = {np.sqrt(var_E10)}')
print(
    f'An approximate value of the probability that the project misses the deadline using importance sampling technique = {count/n}')
print(f'Effective Sample Size = {sample_size}')
print()

# Part (f)
lamda_ = theta
k_values = [3, 4, 5]
mini_ss = n**4
mini_k = 0
mean_ = 0
var_ = 0
for k in k_values:
    print(f'for k = {k}')
    for i in range(len(theta)):
        if (i == 0 or i == 1 or i == 3 or i == 9):
            lamda_[i] = k*theta[i]
    E_10, count = generate_E10(lambda_, n)
    mean_E10 = np.mean(E_10)
    var_E10 = np.var(E_10)
    sample_size = variance_E10_monte/var_E10 * n
    print(f'The observed mean of E10 = {mean_E10}')
    print(f'The observed standard deviation of E10 = {np.sqrt(var_E10)}')
    print(
        f'An approximate value of the probability that the project misses the deadline using importance sampling = {count/n}')
    print(f'Effective Sample Size = {sample_size}\n')
    if sample_size < mini_ss:
        mini_ss = sample_size
        mini_k = k
        mean_ = mean_E10
        var_ = var_E10
    print()

# Part (h)
print(f"k = {mini_k} has the minimum effective sample size.")
L = mean_ - 2.58*np.sqrt(var_)/np.sqrt(n)
U = mean_ + 2.58*np.sqrt(var_)/np.sqrt(n)
print(f"99% Confidence interval {(L, U)}")
