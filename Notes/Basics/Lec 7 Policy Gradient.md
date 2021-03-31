# Lec 7 Policy Gradient

[toc]

- Last Lecture
  - we approximated the value or action-value function using parameters θ,

  - $$
    \begin{aligned} V_{\theta}(s) & \approx V^{\pi}(s) \\ Q_{\theta}(s, a) & \approx Q^{\pi}(s, a) \end{aligned}
    $$

  -  And policy is generated directly from the value function (e.g. using ε-greedy)

- This Lecture

  - We will focus on **model-free** reinforcement learning

  - $$
    \pi_{\theta}(s, a)=\mathbb{P}[a \mid s, \theta]
    $$



## Introduction

Value-Based v.s. Policy-Based

<img src="https://raw.githubusercontent.com/hanmochen/Pictures/master/20210322170757.png" style="zoom:33%;" />

Advantages of Policy-Based RL

- better convergence
- Effective in high-dimensional or continuous action spaces
- Can learn stochastic policies

Disadvantages

- Local Optimum
- Evaluating a policy is inefficient and high variance



### Policy Search

Goal: give policy $π_θ(s,a)$ with parameters $θ$, find best $θ$

How do we measure the quality of $π_θ$

- In episodic environments we use the start value:
  - $J_{1}(\theta)=V^{\pi_{\theta}}\left(s_{1}\right)=\mathbb{E}_{\pi_{\theta}}\left[v_{1}\right]$
- In continuing environments we use average value
  - $J_{a v V}(\theta)=\sum_{s} d^{\pi_{\theta}}(s) V^{\pi_{\theta}}(s)$
  - or average reward per time-step
  - $J_{a v R}(\theta)=\sum_{s} d^{\pi_{\theta}}(s) \sum_{a} \pi_{\theta}(s, a) \mathcal{R}_{s}^{a}$
  - where $d^{π_θ}(s)$ is the stationary distribution of Markov Chain



Given $π_θ(s,a)$ and $J(θ)$, policy-based reinforcement learning is an optimization problem.

Optimization methods

- Non-gradient methods
- Gradient methods
  - gradient descent



## Finite Difference Policy Gradient



- To evaluate policy gradient of $π_θ(s,a)$
- For each dimension $k\in [1,n]$
  - $\frac{\partial J(\theta)}{\partial \theta_{k}} \approx \frac{J\left(\theta+\epsilon u_{k}\right)-J(\theta)}{\epsilon}$
- Simple, noisy, inefficient



## Monte-Carlo Policy Gradient



### Score Function

Assume policy $π_θ$ is differentiable whenever it is non-zero
$$
\begin{aligned} \nabla_{\theta} \pi_{\theta}(s, a) &=\pi_{\theta}(s, a) \frac{\nabla_{\theta} \pi_{\theta}(s, a)}{\pi_{\theta}(s, a)} \\ &=\pi_{\theta}(s, a) \nabla_{\theta} \log \pi_{\theta}(s, a) \end{aligned}
$$
Score Function $\nabla_{\theta} \log \pi_{\theta}(s, a)$





For softmax policy
$$
\pi_{\theta}(s, a) \propto e^{\phi(s, a)^{\top} \theta}
$$
So the score function is
$$
\nabla_{\theta} \log \pi_{\theta}(s, a)=\phi(s, a)-\mathbb{E}_{\pi_{\theta}}[\phi(s, \cdot)]
$$


For Gaussian Policy(in continuous action spaces) $a \sim \mathcal{N}\left(\mu(s)=\phi(s)^T \theta, \sigma^{2}\right)$
$$
\nabla_{\theta} \log \pi_{\theta}(s, a)=\frac{(a-\mu(s)) \phi(s)}{\sigma^{2}}
$$



### Policy Gradient Theorem



Consider a simple one-step MDP

- Starting state $s\sim d(s)$
- Terminating after one time-step $r=R_{s,a}$

$$
\begin{aligned} J(\theta) &=\mathbb{E}_{\pi_{\theta}}[r] \\ &=\sum_{s \in \mathcal{S}} d(s) \sum_{a \in \mathcal{A}} \pi_{\theta}(s, a) \mathcal{R}_{s, a} \\ \nabla_{\theta} J(\theta) &=\sum_{s \in \mathcal{S}} d(s) \sum_{a \in \mathcal{A}} \pi_{\theta}(s, a) \nabla_{\theta} \log \pi_{\theta}(s, a) \mathcal{R}_{s, a} \\ &=\mathbb{E}_{\pi_{\theta}}\left[\nabla_{\theta} \log \pi_{\theta}(s, a) r\right] \end{aligned}
$$

**Policy Gradient Theorem**

For any diﬀerentiable policy $π_θ(s,a)$

the policy gradient is $\nabla_{\theta} J(\theta)=\mathbb{E}_{\pi_{\theta}}\left[\nabla_{\theta} \log \pi_{\theta}(s, a) Q^{\pi_{\theta}}(s, a)\right]$



### REINFORCE

- Update parameters by stochastic gradient descent
- Using return $v_t$ as an unbiased sample of $Q^{π_θ}(s_t,a_t)$
- $\Delta \theta_{t}=\alpha \nabla_{\theta} \log \pi_{\theta}\left(s_{t}, a_{t}\right) v_{t}$



<img src="https://raw.githubusercontent.com/hanmochen/Pictures/master/20210323135921.png" alt="image-20210323135834474" style="zoom: 50%;" />



## Actor-Critic Policy Gradient



Why do we need a critic?

- Problem of MC-PG: high variance due to $v_t$ as an estimator of $Q^{π_θ}(s,a)$
- Instead we use a critic to estimate it $Q_{w}(s, a) \approx Q^{\pi_{\theta}}(s, a)$

Actor-critic algorithm maintains 

- Critic: Updates action-value function parameters $w$
- Actor: Updates policy parameters $θ$, in direction suggested by critic

$$
\nabla_{\theta} J(\theta) \approx \mathbb{E}_{\pi_{\theta}}\left[\nabla_{\theta} \log \pi_{\theta}(s, a) Q_{w}(s, a)\right]
$$



### Critic Design



- How to evaluate action-value function (Policy Evaluation)
  - MC learning
  - TD learning
  - function approxmation
- Example: QAC
  - Using linear function approximation $Q_{w}(s, a)=\phi(s, a)^{\top} w$
  - Critic: Updates $w$ by linear TD(0)
  - Actor: Update $θ$ by policy gradient
  - <img src="https://raw.githubusercontent.com/hanmochen/Pictures/master/20210323141632.png" alt="image-20210323141632601" style="zoom:50%;" />



### Bias in Actor-critic

- Bias comes from value function approximation
- A biased policy gradient may lead to the wrong direction
- Luckily, if we choose function approximation carefully, we can avoid the bias.



**Compatible Function Approximation Theorem**

If the following conditions are satisfied,

- Value Function Approximator is compatible to the policy $\nabla_{w} Q_{w}(s, a)=\nabla_{\theta} \log \pi_{\theta}(s, a)$
- Value Function parameters minimize the mean-squared error $\varepsilon=\mathbb{E}_{\pi_{\theta}}\left[\left(Q^{\pi_{\theta}}(s, a)-Q_{w}(s, a)\right)^{2}\right]$

Then the policy gradient is exact,
$$
\nabla_{\theta} J(\theta)=\mathbb{E}_{\pi_{\theta}}\left[\nabla_{\theta} \log \pi_{\theta}(s, a) Q_{w}(s, a)\right]
$$


### Advantage Funtion

**Reducing Variance using baseline**

- We want to find a baseline function $B(s)$ only related to state $s$, which satisfies $\mathbb{E}_{\pi_{\theta}}\left[\nabla_{\theta} \log \pi_{\theta}(s, a) B(s)\right]=0$
- A good baseline is $B(s)=V^{π_θ}(s)$
- Then we can define the advantage function 

$$
\begin{aligned} A^{\pi_{\theta}}(s, a) &=Q^{\pi_{\theta}}(s, a)-V^{\pi_{\theta}}(s) \\ \nabla_{\theta} J(\theta) &=\mathbb{E}_{\pi_{\theta}}\left[\nabla_{\theta} \log \pi_{\theta}(s, a) A^{\pi_{\theta}}(s, a)\right] \end{aligned}
$$



**Estimating Advantage Function**

- We can estimate both $V^{π_θ}(s)$ and $Q^{\pi_{\theta}}(s, a)$

$$
\begin{aligned} V_{v}(s) & \approx V^{\pi_{\theta}}(s) \\ Q_{w}(s, a) & \approx Q^{\pi_{\theta}}(s, a) \\ A(s, a) &=Q_{w}(s, a)-V_{v}(s) \end{aligned}
$$

- Notice that the TD-error $δ^{π_θ}$ is an unbiased estimator of the advantage function

$$
\begin{aligned} \mathbb{E}_{\pi_{\theta}}\left[\delta^{\pi_{\theta}} \mid s, a\right] &=\mathbb{E}_{\pi_{\theta}}\left[r+\gamma V^{\pi_{\theta}}\left(s^{\prime}\right) \mid s, a\right]-V^{\pi_{\theta}}(s) \\ &=Q^{\pi_{\theta}}(s, a)-V^{\pi_{\theta}}(s) \\ &=A^{\pi_{\theta}}(s, a) \end{aligned}
$$

- We can only use TD error to compute policy gradient 

$$
\nabla_{\theta} J(\theta)=\mathbb{E}_{\pi_{\theta}}\left[\nabla_{\theta} \log \pi_{\theta}(s, a) \delta^{\pi_{\theta}}\right]
\\
\delta_{v}=r+\gamma V_{v}\left(s^{\prime}\right)-V_{v}(s)
$$

- And this way we only use parameters $v$.



### Policy Gradient with Eligibility Traces 



- In Critic, we can use MC, TD(0), forward-view/backward-view TD(λ)

- In Actor, we can also use these methods

  - Monte-Carlo policy gradient $\Delta \theta=\alpha\left(v_{t}-V_{v}\left(s_{t}\right)\right) \nabla_{\theta} \log \pi_{\theta}\left(s_{t}, a_{t}\right)$

  - Actor-Critic Policy Gradient $\Delta \theta=\alpha\left(r+\gamma V_{v}\left(s_{t+1}\right)-V_{v}\left(s_{t}\right)\right) \nabla_{\theta} \log \pi_{\theta}\left(s_{t}, a_{t}\right)$

  - Policy Gradient with Eligibility Traces 

  - $$
    \begin{aligned} \delta &=r_{t+1}+\gamma V_{v}\left(s_{t+1}\right)-V_{v}\left(s_{t}\right) \\ e_{t+1} &=\lambda e_{t}+\nabla_{\theta} \log \pi_{\theta}(s, a) \\ \Delta \theta &=\alpha \delta e_{t} \end{aligned}
    $$



### Natural Policy Gradient

- Vanilla gradient descent is ususlly sensitive to parameters.

- To avoid this, we can use Natural Policy Gradient

- $$
  \nabla_{\theta}^{n a t} \pi_{\theta}(s, a)=G_{\theta}^{-1} \nabla_{\theta} \pi_{\theta}(s, a)
  $$

- where $G_θ$ is the Fisher information matrix

$$
G_{\theta}=\mathbb{E}_{\pi_{\theta}}\left[\nabla_{\theta} \log \pi_{\theta}(s, a) \nabla_{\theta} \log \pi_{\theta}(s, a)^{T}\right]
$$


$$
\begin{aligned} \nabla_{\theta} J(\theta) &=\mathbb{E}_{\pi_{\theta}}\left[\nabla_{\theta} \log \pi_{\theta}(s, a) A^{\pi_{\theta}}(s, a)\right] \\ &=\mathbb{E}_{\pi_{\theta}}\left[\nabla_{\theta} \log \pi_{\theta}(s, a) \nabla_{\theta} \log \pi_{\theta}(s, a)^{T} w\right] \\ &=G_{\theta} w \\ \nabla_{\theta}^{n a t} J(\theta) &=w \end{aligned}
$$


### Summary

$$
\begin{aligned} \nabla_{\theta} J(\theta) &=\mathbb{E}_{\pi_{\theta}}\left[\nabla_{\theta} \log \pi_{\theta}(s, a) v_{t}\right] &\quad \text{REINFORCE}\\ 
&=\mathbb{E}_{\pi_{\theta}}\left[\nabla_{\theta} \log \pi_{\theta}(s, a) Q^{w}(s, a)\right] &\quad \text{Q actor-critic}\\ 
&=\mathbb{E}_{\pi_{\theta}}\left[\nabla_{\theta} \log \pi_{\theta}(s, a) A^{w}(s, a)\right] &\quad \text{Advantage actor-critic}\\
&=\mathbb{E}_{\pi_{\theta}}\left[\nabla_{\theta} \log \pi_{\theta}(s, a) \delta\right]  &\quad \text{TD actor-critic}\\ 
&=\mathbb{E}_{\pi_{\theta}}\left[\nabla_{\theta} \log \pi_{\theta}(s, a) \delta e\right] &\quad \text{TD(λ) actor-critic}\\ 
G_{\theta}^{-1} \nabla_{\theta} J(\theta) &=w  & \quad \text{Natural Actor-Critic}

\end{aligned}
$$

