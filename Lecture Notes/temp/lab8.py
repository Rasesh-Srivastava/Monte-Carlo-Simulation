import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma # to get cdf of gamma distribution



def lcg(a,b,m,x,n):
    seq= []
    for i in range(n):
        x = a*x+b
        x%=m
        seq.append(x/m)
    return seq

def fibonacci_generator(u):
    while(len(u)<1000000):
        ui = u[-17]-u[-5]
        if(ui<0):
            ui+=1
        u.append(ui)

def uniform_number(seq):
    if(len(seq)==17):
        fibonacci_generator(seq)
        seq= seq[:50]
    temp = seq[-1]
    seq.pop()
    return temp

uniform_seq = lcg(731987, 97839941, 1101345, 1, 10000)
fibonacci_generator(uniform_seq)
uniform_seq = uniform_seq[50:]


def gen_poisson(unif, x):
    cutoff = np.exp(-x)
    count = 0
    temp =uniform_number(unif)
    while(temp >= cutoff):
        temp *=uniform_number(unif)
        count += 1
    return count 

def poi_pmf(x):
    lam=2.9
    return np.power(lam,x)*np.power(np.e,-lam)/np.math.factorial(x)

def Box_Muller():
    U1 = uniform_number(uniform_seq)
    U2 = uniform_number(uniform_seq)

    R = np.sqrt(-2.0 * np.log(U1))
    theta = 2.0 * np.pi * U2

    Z1 = R * np.cos(theta)
    Z2 = R * np.sin(theta)
    return Z1

def gen_gamma(n):
    y=0
    while(n>0):
        ui = uniform_number(uniform_seq)
        x=-np.log(ui)
        y+=x
        n-=1
    return y

def q1_simple(n):
    l = 1/3
    k=0.8
    # poi= np.random.poisson(2.9, n)
    poi=[]
    for i in range(n):
        poi.append(gen_poisson(uniform_seq,2.9))
    y=[]
    for j in range (n):
        rain=0
        for i in range(poi[j]):
            ui = uniform_number(uniform_seq)
            xi = np.power(-np.log(1-ui),1/k)/l  #inverse of cdf of weibull
            rain = rain+xi
        if(rain<5):
            y.append(1)
        else:
            y.append(0)
    un = np.mean(y)
    temp = np.sqrt(np.sum((y-un)**2)/(n*(n-1)))
    lb = un - 2.58*temp
    ub = un+2.58*temp
    print(f"Estimated value of I is {un}")
    print(f"99% confidence interval for n = {n} is ( {lb}  , {ub} )")
    return y



def q1_stratified(n):
    l = 1/3
    k=0.8
    probab = [poi_pmf(0),poi_pmf(1),poi_pmf(2),poi_pmf(3),poi_pmf(4),poi_pmf(5)]
    probab.append(1-sum(probab))
    nj = []
    for i in range(6):
        nj.append(n*probab[i])
    value=0
    var=0
    for i in range (6):
        y=[]
        for j in range (int(nj[i])):
            rain=0
            for m in range(i):
                ui = uniform_number(uniform_seq)
                xi = np.power(-np.log(1-ui),1/k)/l  #inverse of cdf of weibull
                rain = rain+xi
            if(rain<5):
                y.append(1)
            else:
                y.append(0)
        if(len(y)>0):
            un = np.mean(y)
            value = value +probab[i]*un
            temp = np.sqrt(np.sum((y-un)**2)/(j*(j-1)))
            temp = temp*probab[i]*probab[i]
            var+=temp
    r = n*probab[6]
    for j in range (int(r)):
        rain=0
        x=gen_poisson(uniform_seq,2.9)
        while(x<6):
            x=gen_poisson(uniform_seq,2.9)
        for m in range(x):
            ui = uniform_number(uniform_seq)
            xi = np.power(-np.log(1-ui),1/k)/l  #inverse of cdf of weibull
            rain = rain+xi
        if(rain<5):
            y.append(1)
        else:
            y.append(0)
    if(len(y)>0):
        un = np.mean(y)
        value = value +probab[i]*un
        # print("Hello")
        # print(j)
        temp = np.sqrt(np.sum((y-un)**2)/(j*(j-1)))
        temp = temp*probab[i]*probab[i]
        var+=temp
    lb = value - 2.58*var
    ub = value+2.58*var
    print(f"Estimated value of I is {value}")
    print(f"99% confidence interval for n = {n} is ( {lb}  , {ub} )")


def q2(n):
    alpha = [2082, 1999, 2008, 2047, 2199, 2153, 1999, 2136, 2053, 2121, 1974, 2110, 2110, 2168, 2035, 2019, 2044, 2191, 2284, 1912, 2196, 2099, 2041, 2192, 2188, 1984, 2158, 2019, 2032, 2051, 2192, 2133, 2142, 2113, 2150, 2221, 2046, 2127]
    h_values =[]
    for j in range(n):
        # y19 = np.random.gamma(shape = alpha[18], scale = 1, size = 1)
        y19= gen_gamma(alpha[18])
        h = 1
        for i in range(38):
            if(i!=18):
                cdf_value = gamma.cdf(y19, a=alpha[i], scale=1)
                h*=cdf_value
        h_values.append(h)
    print(np.mean(h_values))



def q3(n):
    f=[]
    h=[]
    for j in range(n):
        # random_samples = np.random.randn(5)
        random_samples=[]
        for i in range (5):
            random_samples.append(Box_Muller())
        s=0
        x=1
        for i in range(5):
            s=s+np.power(np.e,random_samples[i])
            x*=np.power(np.e,random_samples[i])
        s/=5 # f(x)
        x/=5 # h(x)
        f.append(s)
        h.append(x)
    u_hat = np.mean(f)
    theta_hat = np.mean(h)
    print(f'u_hat = {u_hat} ')
    print(f'theta_hat = {theta_hat}')
    f_ = f-u_hat
    h_ = h - theta_hat
    beta_opt = 0
    h__=0
    for j in range(n):
        beta_opt = beta_opt + f_[j]*h_[j]
        h__ += np.power(h_[j],2)
    beta_opt/=h__
    print(f'optimal value of beta = {beta_opt}')
    theta = 0.2*np.power(np.e,2.5)
    u = u_hat + beta_opt*(theta - theta_hat)
    print(f'value of the regression estimator = {u}')


# q1_simple(100)
# q1_simple(10000)

q1_stratified(100)
# q1_stratified(10000)

# q2(5000)

# q3(10000)

