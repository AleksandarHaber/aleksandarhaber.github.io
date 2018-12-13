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

The vector $\mathbf{u}$ represents external influences on the system. The vector $\mathbf{y}$ is the vector of the observed variables,  and the vector $\mathbf{x}$ is the state vector.
Loosely speaking, this vector represents the internal system memory. 

The identification problem for the system \eqref{eq1:state}-\eqref{eq1:output}. Given the sequence of the input-output data $$\{ \big(\mathbf{u}(t_{0}),\mathbf{y}(t_{0})\big), \big(\mathbf{u}(t_{1}),\mathbf{y}(t_{1})\big), \ldots, \big(\mathbf{u}(t_{N}),\mathbf{y}(t_{N})\big) \}$$, 
where $N$ is the length of the sequence, estimate the system matrices $A$, $B$, $C$, and $D$ up to (unknown) similarity transformation. The estimated state-space model should accurately reproduce the input-output behavior of the original system.

From a practical point of view, it is easier to estimate a discrete-time version of the original system. The discrete-time system has the following form:

$$
\begin{align}
& \mathbf{x}_{k+1}=A\mathbf{x}_{k}+B\mathbf{u}_{k} \label{eq1:stateDiscrete} \\
& \mathbf{y}_{k}=C\mathbf{x}_{k}+ D\mathbf{u}_{k}\label{eq1:outputDiscrete}  
\end{align}
$$
where $k$ is the discrete time instant. That is, the state and other vectors are approximations of the original state at the time $t=kT$, where $T$ is a sampling period.

Obviously, the recurrent neural network architectures such as Long Short-Term Memory networks do not have the exact structure of the state-space model. However, we can still 
employ them to learn to reproduce the input-outpur behavior of the discrete-time state-space model \eqref{eq1:stateDiscrete}-\eqref{eq1:outputDiscrete}.
That is, when we apply the sequence of inputs $$\{\mathbf{u}_{0},\mathbf{u}_{1},\ldots, \mathbf{u}_{N} \}$$ to the neural network, it should produce the sequence of outputs 
$$\{\hat{\mathbf{y}}_{0},\hat{\mathbf{y}_{1}},\ldots, \hat{\mathbf{y}}_{N} \}$$ that accurately approximates the output sequence $$\{\mathbf{y}_{0},\mathbf{y}_{1},\ldots, \mathbf{y}_{N} \}$$ of the real-system. 

In the sequel, we will show how to perform this using the Keras machine learning toolbox. For presentation clarity, this will be explained using the example of a mass-spring system. The codes can be found on my GitHub repository.

The mass-spring system is shown in Fig. 1.

|![Mass-spring system](/assets/images/mass_spring.jpg){: .center-image } |
|:--:|
| Figure 1: Mass spring system.|


The mathematical model is given by the following ordinary differential equation:
\begin{align}
m\ddot{d}+k_{d}\dot{d}+k_{s}d=F
\label{eq2:ode}
\end{align}

where $d$ is the distance from the equilibrium point, $m$ is the mass, $k_{d}$ and $k_{s}$ are the damping and spring constants, and $F$ is the external force. 
By introducing the state-space variables $x_{1}=d$ and $x_{2}=\dot{d}$, the model \eqref{eq2:ode} can be written in the state-space form
$$
\begin{align}
\underbrace{\begin{bmatrix}
\dot{x}_{1} \\ \dot{x}_{2}
\end{bmatrix}}_{\dot{\mathbf{x}}}= \underbrace{\begin{bmatrix}
0 & 1 \\ -\frac{k_{s}}{m} & -\frac{k_{d}}{m}\end{bmatrix}}_{A} \underbrace{\begin{bmatrix}
x_{1} \\ x_{2} 
\end{bmatrix}}_{\mathbf{x}}+\underbrace{\begin{bmatrix} 0 \\ \frac{1}{m} \end{bmatrix}}_{B}\underbrace{F}_{\mathbf{u}} \label{state:Mass_spring}
\end{align}
$$
We assume that only the position vector $d$ (that is, the state variable $x_{1}$) can be observed. Consequently, the output equation takes the following form:
$$
\begin{align}
\mathbf{y}=\underbrace{\begin{bmatrix}1 & 0\end{bmatrix}}_{C}\mathbf{x}
\label{output:Mass_spring}
\end{align}
$$
The state-space model \eqref{state:Mass_spring}-\eqref{output:Mass_spring} of the spring-mass system is in the continuous-time domain. To transform it into the 
discrete-time domain we use the [Backward Euler](https://en.wikipedia.org/wiki/Backward_Euler_method) method. Using this approximation, we obtain:
$$
\begin{align}
\frac{\mathbf{x}_{k}-\mathbf{x}_{k-1}}{h}=A\mathbf{x}_{k}+B\mathbf{u}_{k-1}
\label{discreteTime1}
\end{align}
$$
where $h$ is the discretization time step. The last equation leads to 

$$
\begin{align}
\mathbf{x}_{k}=\tilde{A}\mathbf{x}_{k-1}+\tilde{B}\mathbf{u}_{k-1}
\label{discreteTime2}
\end{align}
$$
where $\tilde{A}=(I-hA)^{-1}$ and $\tilde{B}=h\tilde{A}B$, and the output equation remains unchanged:
$$
\begin{align}
\mathbf{y}_{k}=C\mathbf{x}_{k}
\label{output:Mass_springDiscrete}
\end{align}
$$

Next, we present a Python code for computing the system response on the basis of the backward Euler method. 
The code will be used to generate training, validation, and test data for the recurrent neural networks.


{% highlight python linenos %}
import numpy as np
import matplotlib.pyplot as plt

# define the continuous-time system matrices

A=np.matrix([[0, 1],[- 0.1, -0.05]])
B=np.matrix([[0],[1]])
C=np.matrix([[1, 0]])
#define an initial state for simulation
x0=np.random.rand(2,1)

#define the number of time-samples used for the simulation and the sampling time for the discretization
time=300
sampling=0.5

#define an input sequence for the simulation
#input_seq=np.random.rand(time,1)
input_seq=np.ones(time)
#plt.plot(input_sequence)


# the following function simulates the state-space model using the backward Euler method
# the input parameters are:
#    -- A,B,C              - continuous time system matrices 
#    -- initial_state      - the initial state of the system 
#    -- time_steps         - the total number of simulation time steps 
#    -- sampling_perios    - the sampling period for the backward Euler discretization 
# this function returns the state sequence and the output sequence
# they are stored in the vectors Xd and Yd respectively
def simulate(A,B,C,initial_state,input_sequence, time_steps,sampling_period):
    from numpy.linalg import inv
    I=np.identity(A.shape[0]) # this is an identity matrix
    Ad=inv(I-sampling_period*A)
    Bd=Ad*sampling_period*B
    Xd=np.zeros(shape=(A.shape[0],time_steps+1))
    Yd=np.zeros(shape=(C.shape[0],time_steps+1))
    
    for i in range(0,time_steps):
       if i==0:
           Xd[:,[i]]=initial_state
           Yd[:,[i]]=C*initial_state
           x=Ad*initial_state+Bd*input_sequence[i]
       else:
           Xd[:,[i]]=x
           Yd[:,[i]]=C*x
           x=Ad*x+Bd*input_sequence[i]
    Xd[:,[-1]]=x
    Yd[:,[-1]]=C*x
    return Xd, Yd
    
state,output=simulate(A,B,C,x0,input_seq, time ,sampling)    


plt.plot(output[0,:])
plt.xlabel('Discrete time instant-k')
plt.ylabel('Position- d')
plt.title('System step response')
{% endhighlight %}








![Mass-spring system](/assets/images/step_response.png)














