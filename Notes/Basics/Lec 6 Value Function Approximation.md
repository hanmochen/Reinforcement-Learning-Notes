

[toc]

## Introduction

### RL for large-scale Problems

How does Reinforcement Learning solve large problems?

 For small-scale problems, we use a lookup table

- $s\to V(s)$
- $(s,a) \to V(s,a)$

Problems with large MDPs

- Too many states or state-action pairs to store in memory
- Too slow to learn the value of each state or state-action pair

Solutions:

- Esitimate value function with a function approximation $\hat{v}(s, w) \approx v_{\pi}(s)$ or $\hat{q}(s, a, \mathbf{w}) \approx q_{\pi}(s, a)$
- Generalize from seen states to unseen states
- Use parameter $\mathbf{w}$ using MC or TD learning



### Types of Value Function Approximation

<img src="https://raw.githubusercontent.com/hanmochen/Pictures/master/20210308140423.png" alt="image-20210308140423249" style="zoom:33%;" />



Function Approximators:

- **Linear Regression**
- **Decision Tree**
- Nearest Neighbor
- Fourier/wavelet bases
- Neural Network

In RL, we prefer the function approximator to be **differentiable**. 

Furthermore, it requires a training method for **non-stationary, non-iid** data.



## Incremental Methods

### Gradient Descent

**Gradient Descent**

- Let $J(\mathbf{w})$ be a differentiable function 
- The gradient is 

$$
\nabla_{\mathbf{w}} J(\mathbf{w})=\left(\begin{array}{c}\frac{\partial J(\mathbf{w})}{\partial \mathbf{w}_{1}} \\ \vdots \\ \frac{\partial J(\mathbf{w})}{\partial \mathbf{w}_{n}}\end{array}\right)
$$

- $\Delta \mathbf{w}=-\frac{1}{2} \alpha \nabla_{\mathbf{w}} J(\mathbf{w})$



**Value Function Approximation By Stochastic Gradient Descent**

- Goal: find parameter vector $\mathbf{w}$ minimizing mean-squared error between target function $\hat{v}(s,\mathbf{w})$ and true value function $v_π(s)$
  $$
  J(\mathbf{w})=\mathbb{E}_{\pi}\left[\left(v_{\pi}(S)-\hat{v}(S, \mathbf{w})\right)^{2}\right]
  $$

- Stochastic Gradient Descent
  $$
  \Delta \mathbf{w}=\alpha\left(v_{\pi}(S)-\hat{v}(S, \mathbf{w})\right) \nabla_{\mathbf{w}} \hat{v}(S, \mathbf{w})
  $$
  



### Linear Function Approximation

**Feature Vectors**

- Represent state by a feature vector
  $$
  \mathbf{x}(S)=\left(\begin{array}{c}\mathbf{x}_{1}(S) \\ \vdots \\ \mathbf{x}_{n}(S)\end{array}\right)
  $$
  
- For example
  - Distance of robot from landmarks
  - Trends in the stock market
  - Piece and pawn configurations in chess



**Linear Function Approxmation**
$$
\hat{v}(S, \mathbf{w})=\mathbf{x}(S)^{\top} \mathbf{w}=\sum_{j=1}^{n} \mathbf{x}_{j}(S) \mathbf{w}_{j}
$$
Objective Function
$$
J(\mathbf{w})=\mathbb{E}_{\pi}\left[\left(v_{\pi}(S)-\mathbf{x}(S)^{\top} \mathbf{w}\right)^{2}\right]
$$
Stochastic Gradient Descent
$$
\Delta \mathbf{w}=\alpha\left(v_{\pi}(S)-\hat{v}(S, \mathbf{w})\right) \mathbf{x}(S)
$$
Table Lookup is a special case of linear function approximation
$$
\mathbf{x}^{\text {table }}(S)=\left(\begin{array}{c}\mathbf{1}\left(S=s_{1}\right) \\ \vdots \\ \mathbf{1}\left(S=s_{n}\right)\end{array}\right)
$$

$$
\hat{v}(S, \mathbf{w})=\left(\begin{array}{c}\mathbf{1}\left(S=s_{1}\right) \\ \vdots \\ \mathbf{1}\left(S=s_{n}\right)\end{array}\right) \cdot\left(\begin{array}{c}\mathbf{w}_{1} \\ \vdots \\ \mathbf{w}_{n}\end{array}\right)
$$



### Incremental Prediction Algorithms

- Problem: we don't know the true $v_π(s)$

- Solution: find a target to replace $v_π(s)$

- In MC, the target is the return $G_t$
  $$
  \Delta \mathbf{w}=\alpha\left(G_{t}-\hat{v}\left(S_{t}, \mathbf{w}\right)\right) \nabla_{\mathbf{w}} \hat{v}\left(S_{t}, \mathbf{w}\right)
  $$

  - $G_t$ is unbiased
  - Apply supervised learning to training data $\left\langle S_{1}, G_{1}\right\rangle,\left\langle S_{2}, G_{2}\right\rangle, \ldots,\left\langle S_{T}, G_{T}\right\rangle$
  - Linear model converges to global optimum
  - Non-linear model converges to local optimum

- In TD(0), the target is the TD target $R_{t+1}+\gamma \hat{v}\left(S_{t+1}, \mathbf{w}\right)$
  $$
  \Delta \mathbf{w}=\alpha\left(R_{t+1}+\gamma \hat{v}\left(S_{t+1}, \mathbf{w}\right)-\hat{v}\left(S_{t}, \mathbf{w}\right)\right) \nabla_{\mathbf{w}} \hat{v}\left(S_{t}, \mathbf{w}\right)
  $$

  - TD target is biased
  - Apply supervised learning to training data $\left\langle S_{1}, R_{2}+\gamma \hat{v}\left(S_{2}, \mathbf{w}\right)\right\rangle,\left\langle S_{2}, R_{3}+\gamma \hat{v}\left(S_{3}, \mathbf{w}\right)\right\rangle, \ldots,\left\langle S_{T-1}, R_{T}\right\rangle$
  - Linear model converges close to global optimum

- In TD(λ), the target is the λ-return $G_t^λ$
  $$
  \Delta \mathbf{w}=\alpha\left(G_{t}^{\lambda}-\hat{v}\left(S_{t}, \mathbf{w}\right)\right) \nabla_{\mathbf{w}} \hat{v}\left(S_{t}, \mathbf{w}\right)
  $$

  - λ-return is biased

  - Apply supervised learning to training data $\left\langle S_{1}, G_{1}^{\lambda}\right\rangle,\left\langle S_{2}, G_{2}^{\lambda}\right\rangle, \ldots,\left\langle S_{T-1}, G_{T-1}^{\lambda}\right\rangle$

  - Forward view TD(λ)
    $$
    \Delta \mathbf{w}=\alpha\left(G_{t}^{\lambda}-\hat{v}\left(S_{t}, \mathbf{w}\right)\right) \nabla_{\mathbf{w}} \hat{v}\left(S_{t}, \mathbf{w}\right)
    $$

  - Backward View TD(λ)
    $$
    \begin{aligned} \delta_{t} &=R_{t+1}+\gamma \hat{v}\left(S_{t+1}, \mathbf{w}\right)-\hat{v}\left(S_{t}, \mathbf{w}\right) \\ E_{t} &=\gamma \lambda E_{t-1}+ \nabla_{\mathbf{w}} \hat{v}\left(S_{t}, \mathbf{w}\right)\\ \Delta \mathbf{w} &=\alpha \delta_{t} E_{t} \end{aligned}
    $$
    



### Incremental Control Algorithms



<img src="https://raw.githubusercontent.com/hanmochen/Pictures/master/20210309140640.png" alt="image-20210309140640386" style="zoom:33%;" />

- Policy Evaluation: Approxmate Policy Evaluation
- Policy Improvement: ε-greedy policy improvement



**Action-value function approximation**

- Goal: Find $\hat{q}(S, A, \mathbf{w}) \approx q_{\pi}(S, A)$

- Objective Function: $J(\mathbf{w})=\mathbb{E}_{\pi}\left[\left(q_{\pi}(S, A)-\hat{q}(S, A, \mathbf{w})\right)^{2}\right]$

- SGD: $\Delta \mathbf{w}=\alpha\left(q_{\pi}(S, A)-\hat{q}(S, A, \mathbf{w})\right) \nabla_{\mathbf{w}} \hat{q}(S, A, \mathbf{w})$

- Linear model: $\nabla_{\mathbf{w}} \hat{q}(S, A, \mathbf{w})=\mathbf{x}(S, A)$

- Also, we should find a target to replace $q_π(s,a)$

  - In MC, $\Delta \mathbf{w}=\alpha\left(G_{t}-\hat{q}\left(S_{t}, A_{t}, \mathbf{w}\right)\right) \nabla_{\mathbf{w}} \hat{q}\left(S_{t}, A_{t}, \mathbf{w}\right)$

  - In TD(0), $\Delta \mathbf{w}=\alpha\left(R_{t+1}+\gamma \hat{q}\left(S_{t+1}, A_{t+1}, \mathbf{w}\right)-\hat{q}\left(S_{t}, A_{t}, \mathbf{w}\right)\right) \nabla_{\mathbf{w}} \hat{q}\left(S_{t}, A_{t}, \mathbf{w}\right)$

  - In Forward-View TD(λ): $\Delta \mathbf{w}=\alpha\left(q_{t}^{\lambda}-\hat{q}\left(S_{t}, A_{t}, \mathbf{w}\right)\right) \nabla_{\mathbf{w}} \hat{q}\left(S_{t}, A_{t}, \mathbf{w}\right)$

  - In Backward-View TD(λ): 
    $$
    \begin{aligned} \delta_{t} &=R_{t+1}+\gamma \hat{q}\left(S_{t+1}, A_{t+1}, \mathbf{w}\right)-\hat{q}\left(S_{t}, A_{t}, \mathbf{w}\right) \\ E_{t} &=\gamma \lambda E_{t-1}+\nabla_{\mathbf{w}} \hat{q}\left(S_{t}, A_{t}, \mathbf{w}\right) \\ \Delta \mathbf{w} &=\alpha \delta_{t} E_{t} \end{aligned}
    $$
    





### Convergence



<img src="https://raw.githubusercontent.com/hanmochen/Pictures/master/20210309142053.png" alt="image-20210309142053007" style="zoom: 33%;" />



Problem: TD doesn't follow the gradient of any objective function → Gradient TD

<img src="https://raw.githubusercontent.com/hanmochen/Pictures/master/20210309142251.png" alt="image-20210309142251811" style="zoom:33%;" />

**Convergence of Control Algorithms**

<img src="https://raw.githubusercontent.com/hanmochen/Pictures/master/20210309142325.png" alt="image-20210309142325490" style="zoom:33%;" />



## Batch Methods

- Problem: Low Sample Efficiency
- Solution: Given the agent's experience as input("training data"), find the best fitting value function



### Least Square Prediction

- Given value function approximation model $\hat{v}(s, \mathbf{w}) \approx v_{\pi}(s)$

- And experience $\mathcal{D}$ consisting of <state,value> pairs $\mathcal{D}=\left\{\left\langle s_{1}, v_{1}^{\pi}\right\rangle,\left\langle s_{2}, v_{2}^{\pi}\right\rangle, \ldots,\left\langle s_{T}, v_{T}^{\pi}\right\rangle\right\}$

- Least Sqaures Algorithms find parameters minimizing 
  $$
  \begin{aligned} L S(\mathbf{w}) &=\sum_{t=1}^{T}\left(v_{t}^{\pi}-\hat{v}\left(s_{t}, \mathbf{w}\right)\right)^{2} \\ &=\mathbb{E}_{\mathcal{D}}\left[\left(v^{\pi}-\hat{v}(s, \mathbf{w})\right)^{2}\right] \end{aligned}
  $$
  



**Stochastic Gradient Descent with Experience Replay**

Repeat

1. Sample state, value form experience $\left\langle s, v^{\pi}\right\rangle \sim \mathcal{D}$
2. Apply stochastiv gradient descent update $\Delta \mathbf{w}=\alpha\left(v^{\pi}-\hat{v}(s, \mathbf{w})\right) \nabla_{\mathbf{w}} \hat{v}(s, \mathbf{w})$

Converges to least squares solution $\mathbf{w}^{\pi}=\underset{\mathbf{w}}{\operatorname{argmin}} L S(\mathbf{w})$



**Experience Replay in Deep Q-Networks**(DQN)

DQN uses experience replay and fixed Q-targets

- Take action $a_t$ according to ε-greedy policy

- Store transition $(s_t,a_t,r_{t+1}，s_{t+1})$ in replay memory $\mathcal{D}$

- Sample mini-batch of transitions $(s,a,r,s')$ form $\mathcal{D}$

- Compute Q-learning targets w.r.t. old,fixed parameters $w^-$

- Optimize MSE between Q-network and Q-learning targets
  $$
  \mathcal{L}_{i}\left(w_{i}\right)=\mathbb{E}_{s, a, r, s^{\prime} \sim \mathcal{D}_{i}}\left[\left(r+\gamma \max _{a^{\prime}} Q\left(s^{\prime}, a^{\prime} ; w_{i}^{-}\right)-Q\left(s, a ; w_{i}\right)\right)^{2}\right]
  $$
  

**Linear Least Square Prediction**

- Experience replay may take many iterations

- If we use linear model to approximate value function $\hat{v}(s, \mathbf{w})=\mathbf{x}(s)^{\top} \mathbf{w}$, it has a closed form solution
  $$
  \mathbf{w}=\left(\sum_{t=1}^{T} \mathbf{x}\left(s_{t}\right) \mathbf{x}\left(s_{t}\right)^{\top}\right)^{-1} \sum_{t=1}^{T} \mathbf{x}\left(s_{t}\right) v_{t}^{\pi}
  $$
  
- Direct solution is $O(N^3)$, incremental solution time is $O(N^2)$ using Shermann-Morrison
- LSMC: $\mathbf{w}=\left(\sum_{t=1}^{T} \mathbf{x}\left(S_{t}\right) \mathbf{x}\left(S_{t}\right)^{\top}\right)^{-1} \sum_{t=1}^{T} \mathbf{x}\left(S_{t}\right) G_{t}$
- LSTD: $\mathbf{w}=\left(\sum_{t=1}^{T} \mathbf{x}\left(S_{t}\right)\left(\mathbf{x}\left(S_{t}\right)-\gamma \mathbf{x}\left(S_{t+1}\right)\right)^{\top}\right)^{-1} \sum_{t=1}^{T} \mathbf{x}\left(S_{t}\right) R_{t+1}$



**Convergence**

<img src="https://raw.githubusercontent.com/hanmochen/Pictures/master/20210309152602.png" alt="image-20210309152601899" style="zoom: 33%;" />



### Least Square Control



<img src="https://raw.githubusercontent.com/hanmochen/Pictures/master/20210309153023.png" alt="image-20210309153022935" style="zoom: 33%;" />

- Policy Evaluation: Least Square Q-learning
- Policy Improvement: Greedy Policy Improvement



**Least Square Action-Value Function Approximation**



- Using linear model to approximate action-value function $\hat{q}(s, a, \mathbf{w})=\mathbf{x}(s, a)^{\top} \mathbf{w} \approx q_{\pi}(s, a)$
- From experience generated under policy π: $\mathcal{D}=\left\{\left\langle\left(s_{1}, a_{1}\right), v_{1}^{\pi}\right\rangle,\left\langle\left(s_{2}, a_{2}\right), v_{2}^{\pi}\right\rangle, \ldots,\left\langle\left(s_{T}, a_{T}\right), v_{T}^{\pi}\right\rangle\right\}$



**Least Square Control**

- For policy evaluation, wa want to use all experience
- But experience is generated form many policies
- So we must learn off-policy
  - Use experience generated by old policy $S_{t}, A_{t}, R_{t+1}, S_{t+1} \sim \pi_{\text {old }}$
  - Consider alternative successor action $A'=π_{new}(S_{t+1})$
  - Update $\hat{q}\left(S_{t}, A_{t}, \mathbf{w}\right) = R_{t+1}+\gamma \hat{q}\left(S_{t+1}, A^{\prime}, \mathbf{w}\right)$



**Least-Square Q-learning**

- Linear Q-learning update
  $$
  \begin{aligned} \delta &=R_{t+1}+\gamma \hat{q}\left(S_{t+1}, \pi\left(S_{t+1}\right), \mathbf{w}\right)-\hat{q}\left(S_{t}, A_{t}, \mathbf{w}\right) \\ \Delta \mathbf{w} &=\alpha \delta \mathbf{x}\left(S_{t}, A_{t}\right) \end{aligned}
  $$

- LSTDQ algorithm
  $$
  \mathbf{w}=\left(\sum_{t=1}^{T} \mathbf{x}\left(S_{t}, A_{t}\right)\left(\mathbf{x}\left(S_{t}, A_{t}\right)-\gamma \mathbf{x}\left(S_{t+1}, \pi\left(S_{t+1}\right)\right)\right)^{\top}\right)^{-1} \sum_{t=1}^{T} \mathbf{x}\left(S_{t}, A_{t}\right) R_{t+1}
  $$



**Least Square Policy Iteration Algorithm**

<img src="https://raw.githubusercontent.com/hanmochen/Pictures/master/20210310140244.png" alt="image-20210310140244567" style="zoom: 33%;" />



**Convergence of Control algorithms**

<img src="https://raw.githubusercontent.com/hanmochen/Pictures/master/20210310140408.png" alt="image-20210310140408711" style="zoom:33%;" />

