---
layout: post
title: "Disciplined Estimation of Time Series- Residual Test 1"
author: Aleksandar Haber
categories: machine_learning
comments: true
---
This post is the first post in a series of posts on the estimation of time series. Here we will learn how to test if a time sequence is a white noise. This is important for validating the performance of a method for estimating time series. In this post, we consider scalar time series. 

A time series, denoted by $$e_{k}$$, where $$k$$ is a discrete-time instant, is a white noise if the sequence $$\{e_{k} \}$$ is a sequence of independent and identically distributed random variables that have finite mean and variance.

So our problem can be formulated as follows. Given an observed random sequence $$\{e_{k} \}$$, $$k=1,2,\ldots, n$$, determine if this sequence is white-noise.

How to tackle this fundamental problem? First, we introduce a sample mean:

$$
\begin{align}
\bar{e}_{n}=\sum_{i=1}^{n} e_{i}.
\label{sampleMean}
\end{align}
$$

Next, we need to introduce a sample autocovariance function:

$$
\begin{align}
\hat{\gamma}(h)=\frac{1}{n} \sum_{k=1}^{n-h}  \big(e_{k}-\bar{e}_{n} \big) \big(e_{k+h}-\bar{e}_{n} \big), \;\; 0\le h \le n-1
\label{sampleAutoCovariance}
\end{align}
$$
and the sample autocorrelation function
$$
\begin{align}
\hat{\rho}(h)=\frac{\hat{\gamma}(h)}{\hat{\gamma}(0)}
\label{sampleAutoCorrelation}
\end{align}
$$
It should be kept in mind that $\hat{\rho}(h)$ is biased estimate of the "exact" autocorrelation function $$\hat{\rho}(h)$$. The bias is in the order of $1/n$, and for relatively large $$n$$, this bias is relatively small. 

Next, we recall Theorem  from (Brockwell&Davis, (1991), page 222)

Theorem 1. If $$\{ x_{k}\}$$ is the stationary process
$$
\begin{align}
x_{k}-\mu = \sum_{j=-\infty}^{\infty} a_{j} z_{k-j}
\label{thm1}
\end{align}
$$
where $$\{z_{k}\}~IID(0,\sigma^2)$$, $$\sum_{j=-\infty}^{\infty} |a_{j}| < \infty $$, and $$\sum_{j=-\infty}^{\infty} a_{j}^{2} |j| < \infty $$, then for each $$j\in \{1,2,\ldots \}$$
$$
\begin{align}
\hat{\boldsymbol{\rho}}\;\; \text{is} \;\; \text{AN}\big( \boldsymbol{\rho}, \frac{1}{n} W  \big)
\label{mainResultThm1}
\end{align}
$$
where 
$$
\begin{align}
\hat{\boldsymbol{\rho}}(h)=\begin{bmatrix} \hat{\rho}(1) \\  \hat{\rho}(2) \\ \vdots \\  \hat{\rho}(h)   \end{bmatrix},\;\;
\boldsymbol{\rho}(h)=\begin{bmatrix} \rho(1) \\  \rho(2) \\ \vdots \\  \rho(h)   \end{bmatrix}  
\label{explanation1}
\end{align}
$$

and $$W$$ is the covariance matrix whose $(i,j)$ entry, denoted by $$w_{ij}$$ is given by the Bartlett's formula
$$
\begin{align}
w_{ij}=\sum_{k=1}^{\infty} \big(\rho(k+i)+\rho(k-i)-2\rho(i)\rho(k) \big) \big(\rho(k+j)+\rho(k-j)-2\rho(j)\rho(k) \big)
\end{align}
$$

In Theorem 1, $$\rho$$ denotes the "exact" autocorrelation function. The conditions of this theorem are satisfied by every AutoRegressive Moving Average (ARMA) process, that is driven by an IID sequence $$z_{k}$$. Also in Eq.~\eqref{mainResultThm1} the notation $$AN$$ stands for asymptotic normality. 

So let us use the results of Theorem 1, to test if a sequence is a white noise. Let us assume that the sequence $$\{ \e_{k}\}$$ is IID(0,\sigma^2) Namely, for a white noise, we know that the exact autocorrelation function is 
$$
\rho(0)=1,\;\; \rho(k)=0, \;\; k=1,2,\ldots,
$$
and consequently, the matrix $W$ is an identity matrix. Furthermore, for large $$n$$, the sample autocorrelation functions $$\hat{\rho}(1),\hat{\rho}(2),\ldots, \hat{\rho}(n)$$ are approximatelly independent and identically distributed normal random variables with the mean of $0$ and the variance of $1/n$.
So, we need to perform hypothesis testing: $$H_{0}: \rho(l)=0, \;\; l>0$$ against $$H_{1}: \rho(l) \ne 0, \;\; l>0$$.

Since $\hat{\rho}(k)$ is approximately normally distributed for large $n$ and zero mean, we can write for $l>0$
$$
\begin{align}
P\Big(-1.96\frac{1}{\sqrt{n}} \le \hat{\rho}(l) \le 1.96\frac{1}{\sqrt{n}} \Big) =1-0.05=95 \%
\end{align}
$$


### References 
Brockwell, P. J., & Davis, R. A. (1991). Time series: Theory and methods (2nd ed.).
New York: Springer.

