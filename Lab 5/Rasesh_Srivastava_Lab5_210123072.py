# Rasesh Srivastava
# Roll Number - 210123072

# MA323 Monte Carlo Simulation
# Lab 05

# Question 1
import plotly.express as expressSo
import plotly.graph_objects as PlotlyGraphPlotter
from scipy.stats import gaussian_kde
import numpy as npy
npy.random.seed(42)
import matplotlib.pyplot as GraphPlotter
import math
def Box_Muller_Method(n):
    u1 = npy.random.rand(n)
    u2 = npy.random.rand(n)
    z1 = npy.sqrt(-2 * npy.log(u1)) * npy.cos(2 * npy.pi * u2)
    z2 = npy.sqrt(-2 * npy.log(u1)) * npy.sin(2 * npy.pi * u2)
    return z1, z2
N_Values = 10000
a_Values=[-0.5,0,0.5,1]
MulSama=[]
MulSamb=[]
MulSamc=[]
MulSamd=[]
a11_of_sigma=1
a22_of_sigma=4
a11_of_mean_meu=5
a21_of_mean_meu=8
Mul_Sam1,Mul_Sam2=Box_Muller_Method(10000)
NPY_Samples = npy.array(Mul_Sam1)
FirstSamples = NPY_Samples * a11_of_sigma + a11_of_mean_meu
NPY_Samples = npy.array(Mul_Sam2)
SecondSamples = NPY_Samples * a22_of_sigma + a21_of_mean_meu
for a in a_Values:
    stdDev=[[1,2*a],[2*a,4]]
    CorrelationCoefficient=a
    for z1,z2 in zip(FirstSamples,SecondSamples):
        x1=a11_of_mean_meu+a11_of_sigma*z1
        x2=a21_of_mean_meu+a22_of_sigma*(z1)*(CorrelationCoefficient) +(math.sqrt(1-CorrelationCoefficient*CorrelationCoefficient)*a22_of_sigma*z2)
        if(a==-0.5):
            MulSama.append((x1,x2))
        if(a==0):
            MulSamb.append((x1,x2))
        if(a==0.5):
            MulSamc.append((x1,x2))
        if(a==1):
            MulSamd.append((x1,x2))
MulSama=npy.array(MulSama)
MulSamb=npy.array(MulSamb)
MulSamc=npy.array(MulSamc)
MulSamd=npy.array(MulSamd)
x=MulSama[:,0]
y=MulSama[:,1]
def ContourOnHistogramPlotter(x, y, title):
    fig = PlotlyGraphPlotter.Figure(PlotlyGraphPlotter.Histogram2d(
        x=x,
        y=y
    ))
    kde = gaussian_kde(npy.vstack([x, y]))
    x_range = npy.linspace(min(x), max(x), 100)
    y_range = npy.linspace(min(y), max(y), 100)
    X, Y = npy.meshgrid(x_range, y_range)
    Z = kde(npy.vstack([X.ravel(), Y.ravel()]))
    Z = Z.reshape(X.shape)
    fig.add_trace(PlotlyGraphPlotter.Contour(
        x=x_range,
        y=y_range,
        z=Z,
        colorscale='Viridis',
        colorbar=dict(title='Density'),
        contours=dict(showlines=True),
        opacity=0.4
    ))
    fig.update_layout(
        title=title,
        xaxis_title='X1',
        yaxis_title='X2'
    )
    fig.show()
ContourOnHistogramPlotter(MulSama[:, 0], MulSama[:, 1], 'Contour Plot for a=-0.5 on the 2D-histogram.')
ContourOnHistogramPlotter(MulSamb[:, 0], MulSamb[:, 1], 'Contour Plot for a=0 on the 2D-histogram.')
ContourOnHistogramPlotter(MulSamc[:, 0], MulSamc[:, 1], 'Contour Plot for a=0.5 on the 2D-histogram.')
ContourOnHistogramPlotter(MulSamd[:, 0], MulSamd[:, 1], 'Contour Plot for a=1 on the 2D-histogram.')
npy.random.seed(42)
a_Values=[-0.5,0,0.5,1]
MulSama=[]
MulSamb=[]
MulSamc=[]
MulSamd=[]
Mul_Sam1,Mul_Sam2=Box_Muller_Method(10000)
NPY_Samples = npy.array(Mul_Sam1)
FirstSamples = NPY_Samples * a11_of_sigma + a11_of_mean_meu
NPY_Samples = npy.array(Mul_Sam2)
SecondSamples = NPY_Samples * a22_of_sigma + a21_of_mean_meu
for a in a_Values:
    stdDev=[[1,2*a],[2*a,4]]
    CorrelationCoefficient=a
    for z1,z2 in zip(FirstSamples,SecondSamples):
        x1=a11_of_mean_meu+a11_of_sigma*z1
        x2=a21_of_mean_meu+a22_of_sigma*(z1)*(CorrelationCoefficient) +(math.sqrt(1-CorrelationCoefficient*CorrelationCoefficient)*a22_of_sigma*z2)
        if(a==-0.5):
            MulSama.append((x1,x2))
        if(a==0):
            MulSamb.append((x1,x2))
        if(a==0.5):
            MulSamc.append((x1,x2))
        if(a==1):
            MulSamd.append((x1,x2))
MulSama=npy.array(MulSama)
MulSamb=npy.array(MulSamb)
MulSamc=npy.array(MulSamc)
MulSamd=npy.array(MulSamd)
x=MulSama[:,0]
y=MulSama[:,1]
fig = PlotlyGraphPlotter.Figure(PlotlyGraphPlotter.Histogram2d(
        x=x,
        y=y
    ))
fig.show()
x=MulSamb[:,0]
y=MulSamb[:,1]
fig = PlotlyGraphPlotter.Figure(PlotlyGraphPlotter.Histogram2d(
        x=x,
        y=y
    ))
fig.show()
x=MulSamc[:,0]
y=MulSamc[:,1]
fig = PlotlyGraphPlotter.Figure(PlotlyGraphPlotter.Histogram2d(
        x=x,
        y=y
    ))
fig.show()
x=MulSamd[:,0]
y=MulSamd[:,1]
fig = PlotlyGraphPlotter.Figure(PlotlyGraphPlotter.Histogram2d(
        x=x,
        y=y
    ))
fig.show()
def ContourOnHistogramPlotter2(x, y, title):
    fig = PlotlyGraphPlotter.Figure(PlotlyGraphPlotter.Histogram2d(
        x=x,
        y=y
    ))
    kde = gaussian_kde(npy.vstack([x, y]))
    x_range = npy.linspace(min(x), max(x), 100)
    y_range = npy.linspace(min(y), max(y), 100)
    X, Y = npy.meshgrid(x_range, y_range)
    Z = kde(npy.vstack([X.ravel(), Y.ravel()]))
    Z = Z.reshape(X.shape)
    fig.add_trace(PlotlyGraphPlotter.Contour(
        x=x_range,
        y=y_range,
        z=Z,
        colorscale='Viridis',
        colorbar=dict(title='Density'),
        contours=dict(showlines=True)
    ))
    fig.update_layout(
        title=title,
        xaxis_title='X1',
        yaxis_title='X2'
    )
    fig.show()
ContourOnHistogramPlotter2(MulSama[:, 0], MulSama[:, 1], 'Contour Plot for a=-0.5')
ContourOnHistogramPlotter2(MulSamb[:, 0], MulSamb[:, 1], 'Contour Plot for a=0')
ContourOnHistogramPlotter2(MulSamc[:, 0], MulSamc[:, 1], 'Contour Plot for a=0.5')
ContourOnHistogramPlotter2(MulSamd[:, 0], MulSamd[:, 1], 'Contour Plot for a=1')
mu = npy.array([5, 8])
a_values = [-0.5, 0, 0.5, 1]
num_samples = 10000
samples_list = []
for a in a_values:
    sigma = npy.array([[1, 2 * a], [2 * a, 4]])
    if npy.linalg.det(sigma) == 0:
        print(f"Singular matrix encountered for a = {a}. Skipping contour plot.")
    else:
        samples = npy.random.multivariate_normal(mu, sigma, num_samples)
        samples_list.append(samples)
        x, y = npy.meshgrid(npy.linspace(0, 10, 100), npy.linspace(0, 16, 100))
        pos = npy.dstack((x, y))
        density = (
            npy.exp(-0.5 * ((pos - mu).dot(npy.linalg.inv(sigma)) * (pos - mu)).sum(axis=2))
            / (2 * npy.pi * npy.sqrt(npy.linalg.det(sigma)))
        )
        GraphPlotter.figure(figsize=(8, 6))
        GraphPlotter.contourf(y,x, density, cmap='viridis', levels=20)
        GraphPlotter.colorbar(label='Density')
        GraphPlotter.title(f'Contour Plot of Actual Density (a = {a})')
        GraphPlotter.ylabel('X1')
        GraphPlotter.xlabel('X2')
        GraphPlotter.show()
for i, a in enumerate(a_values):
    if npy.linalg.det(sigma) != 0:
        fig = expressSo.density_heatmap(x=samples_list[i][:, 0], y=samples_list[i][:, 1], title=f'Density Heatmap (a = {a})')
        fig.update_traces(showscale=False)
        fig.show()