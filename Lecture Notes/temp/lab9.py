import numpy as np
import matplotlib.pyplot as plt

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

def q1b(n):
    lam = [4,4,2,5,2,3,2,3,2,2]
    e10=[]
    for i in range (n):
        t=[]
        for j in range (10):
            ui = uniform_number(uniform_seq)
            t.append(-lam[j]*np.log(ui))
        evalue = t[0]+t[9]+ max(max(t[1]+t[3],t[2]+t[7]),t[8]+max(t[4]+t[1],max(t[5],t[6])+t[2]))
        e10.append(evalue)
    print(f"The approximate value of E10 is {np.mean(e10)}")
    return e10


def q1c():
    bins = np.arange(0,60,1)
    plt.hist(e10,bins=bins)
    plt.show()

def q1d():
    y=[]
    n=len(e10)
    for i in range(n):
        if(e10[i]>70):
            y.append(1)
        else:
            y.append(0)
    un=np.mean(y)
    temp = np.sqrt(np.sum((y-un)**2)/((n-1)))
    print(f"The approximate value of probability  is {un}")
    print(f"The std deviation is {temp}")

def q1e(n):
    lam = [4,4,2,5,2,3,2,3,2,2]
    e10=[]
    y=[]
    ys=[]
    e10s=[]
    ratios=[]
    ratiosq=[]
    h= []
    for i in range (n):
        ratio=1
        t=[]
        ts=[]
        for j in range (10):
            ui = uniform_number(uniform_seq)
            t.append(-lam[j]*np.log(ui))
            ts.append(4*t[j])
            ratio = ratio*4*np.power(np.e,-0.75*ts[j]/lam[j])
        evalue = t[0]+t[9]+ max(max(t[1]+t[3],t[2]+t[7]),t[8]+max(t[4]+t[1],max(t[5],t[6])+t[2]))
        evalues = ts[0]+ts[9]+ max(max(ts[1]+ts[3],ts[2]+ts[7]),ts[8]+max(ts[4]+ts[1],max(ts[5],ts[6])+ts[2]))
        e10.append(evalue)
        e10s.append(evalues)
        ratios.append(ratio)
        ratiosq.append(ratio*ratio)
        if(evalue>70):
           y.append(1)
        else:
            y.append(0)
        if(evalues>70):
           ys.append(1)
        else:
            ys.append(0)
        h.append(ys[i]*ratios[i])
    un=np.mean(h)
    temp = np.sqrt(np.sum((h-un)**2)/((n-1)))
    print(f"The approximate value of probability  is {un}")
    print(f"The std deviation is {temp}")
    ne=n*np.mean(ratios)*np.mean(ratios)/np.mean(ratiosq)
    print(f"effective sample size is {ne}")

def q1f(n,k):
    lam = [4,4,2,5,2,3,2,3,2,2]
    e10=[]
    y=[]
    ys=[]
    e10s=[]
    ratios=[]
    ratiosq=[]
    h= []
    for i in range (n):
        ratio=1
        t=[]
        ts=[]
        for j in range (10):
            ui = uniform_number(uniform_seq)
            t.append(-lam[j]*np.log(ui))
            if(j==0 or j==1 or j== 3 or j==9):
             ts.append(k*t[j])
             ratio = ratio*k*np.power(np.e,-(1-1/k)*ts[j]/lam[j])
            else:
                ts.append(t[j])
        evalue = t[0]+t[9]+ max(max(t[1]+t[3],t[2]+t[7]),t[8]+max(t[4]+t[1],max(t[5],t[6])+t[2]))
        evalues = ts[0]+ts[9]+ max(max(ts[1]+ts[3],ts[2]+ts[7]),ts[8]+max(ts[4]+ts[1],max(ts[5],ts[6])+ts[2]))
        e10.append(evalue)
        e10s.append(evalues)
        ratios.append(ratio)
        ratiosq.append(ratio*ratio)
        if(evalue>70):
           y.append(1)
        else:
            y.append(0)
        if(evalues>70):
           ys.append(1)
        else:
            ys.append(0)
        h.append(ys[i]*ratios[i])
    un = np.mean(h)
    temp = np.sqrt(np.sum((h-un)**2)/(n-1))
    print(f"For value of k = {k}")
    print(f"The approximate value of probability  is {un}")
    print(f"The std deviation is {temp}")
    ne=n*np.mean(ratios)*np.mean(ratios)/np.mean(ratiosq)
    print(f"effective sample size is {ne}")
    lb = un - 1.96*temp/np.sqrt(n)
    ub = un+1.96*temp/np.sqrt(n)
    print(f"95% confidence interval for n = {n} is ( {lb}  , {ub} )")
# least sample size is for k = 5

e10 = q1b(10000)
q1c()
q1d()
q1e(10000)

q1f(10000,3)
q1f(10000,4)
q1f(10000,5)