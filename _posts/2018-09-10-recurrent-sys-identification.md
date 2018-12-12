---
layout: post
title: "Using Recurrent NNs and Keras to Reproduce the Input-Output Behaviour of a State-Space Model of a Dynamical System"
categories: misc
---


This post is motivated by an [interesting question](https://github.com/keras-team/keras/issues/8579) posted on the Keras Github page.
The post in which the question is stated starts as follows: "For a final thesis I am working with Keras on Tensorflow and have to investigate how LSTM Networks can be used to identify dynamical systems."

Feedforward neural networks have been extensively used for system identification of nonlinear dynamical systems and state-space models. However, it is interesting to investigate the potential of different recurrent neural network architectures for the identification of state-space models. 
In this post, I will report my progress on this interesting topics and I will post codes and my thoughts. 

So let us start with a problem definition. So let us start with a problem definition. Various dynamical systems, such as robotic arms, oscillators, and even economic systems, 
can be mathematically described by a linearized state-space model having the following form:

$$
\begin{align}
& \dot{\mathbf{x}}(t)=A\mathbf{x}(t)+B\mathbf{u}(t) \label{eq1:state} \\
& \mathbf{y}(t)=C\mathbf{x}(t)+ D\mathbf{u}(t)\label{eq1:output}  
\end{align}
$$
where $\mathbf{x}\in \mathbb{R}^{n}$ is the state vector, $\mathbf{u}\in \mathbb{R}^{m}$ is the input vector, and $\mathbf{y}\in \mathbb{R}^{r}$ is an output vector.
The matrices $A\in \mathbb{R}^{n\times n}$, $B\in \mathbb{R}^{n\times m}$, $\mathbf{C}\in \mathbb{R}^{r \times n}$ and $D\in \mathbb{R}^{r\times m}$ are the system matrices.



In \eqref{eq1:state} and \eqref{eq1:output}
