# Lecture 3 Dynamic Programming

[toc]

## Recap: MDP

### Markov Process

**A Markov Process ( or Markov Chain ) is a tuple $<\mathcal{S},\mathcal{P}>$**

#### Markov Property

$$
\mathbb{P}\left[S_{t+1} \mid S_{t}\right]=\mathbb{P}\left[S_{t+1} \mid S_{1}, \ldots, S_{t}\right]
$$

#### Transition Matrix

$$
\mathcal{P}_{s s^{\prime}}=\mathbb{P}\left[S_{t+1}=s^{\prime} \mid S_{t}=s\right]
\\
\mathcal{P} = \left[\begin{array}{ccc}\mathcal{P}_{11} & \ldots & \mathcal{P}_{1 n} \\ \vdots & & \\ \mathcal{P}_{n 1} & \ldots & \mathcal{P}_{n n}\end{array}\right]
$$



### Markov Reward Process

Add a reward function in a markov process 





#### Definition

**A Markov Reward Process is a tuple $<\mathcal{S},\mathcal{P},\mathcal{R},\gamma>$**

- $\mathcal{R}$ is a reward function, $\mathcal{R}_s = \mathbb{E}[R_{t+1} \mid S_t = s]$
- $γ$ is a discount factor

Return:
$$
G_{t}=R_{t+1}+\gamma R_{t+2}+\ldots=\sum_{k=0}^{\infty} \gamma^{k} R_{t+k+1}
$$
Value Function:
$$
v(s)=\mathbb{E}\left[G_{t} \mid S_{t}=s\right]
$$


#### Bellman Equation 


$$
\begin{aligned}
v(s) &= \mathbb{E}\left[G_{t} \mid S_{t}=s\right] = \mathbb{E}\left[R_{t+1}+\gamma G_{t+1} \mid S_{t}=s\right] 
\\ &= \mathcal{R}_s + \gamma \mathbb{E}\left[G_{t+1} \mid S_{t}=s\right] 
\\ &= \mathcal{R}_s + \gamma \sum_{s'} \mathbb{P}(S_{t+1} = s' \mid S_t = s ) \mathbb{E}\left[G_{t+1} \mid S_{t}=s,S_{t+1} = s'\right] 
\\ & = \mathcal{R}_s + \gamma \sum_{s'} \mathcal{P}_{ss'}v(s')
\end{aligned}
$$

$$
\mathbf{v} = \mathcal{R} + \gamma \mathcal{P} \mathbf{v}
$$



### Markov Decision Process



Add actions in a Markov Decision Process



#### Definition



**A Markov Decision Process  is a tuple $<\mathcal{S},\mathcal{A},\mathcal{P},\mathcal{R},\gamma>$**

- $\mathcal{A}$ is a finite set of actions, which determines the transition probability and the reward
- $ \mathcal{P}_{s s^{\prime}}^{a}=\mathbb{P}\left[S_{t+1}=s^{\prime} \mid S_{t}=s, A_{t}=a\right] $
- $\mathcal{R}_{s}^{a}=\mathbb{E}\left[R_{t+1} \mid S_{t}=s, A_{t}=a\right]$

Policy: 

- $π\in Π = \{π:\mathcal{S}\mapsto \mathcal{A} \} $
- $\pi(a \mid s)=\mathbb{P}\left[A_{t}=a \mid S_{t}=s\right]$



Given $π$,  MDP is a Markov Decision Process with $<\mathcal{S},\mathcal{P^{π }},\mathcal{R}^{π},\gamma>$

In the same way, we can define the **value function.**
$$
v_{\pi}(s)=\mathbb{E}_{\pi}\left[G_{t} \mid S_{t}=s\right]
$$
Moreover, we can define the **action-value function**
$$
q_{\pi}(s, a)=\mathbb{E}_{\pi}\left[G_{t} \mid S_{t}=s, A_{t}=a\right]
$$
The relationship between value function and action-value function
$$
v_{\pi}(s) = \sum_{a\in \mathcal{A}}\pi(a|s) q_{\pi}(s,a)
$$



#### Bellman Expectation Equation



Substitute $\mathbf{v},\mathcal{R},\mathcal{P}$ with $\mathbf{v}^{π},\mathcal{R}^{π},\mathcal{P}^{π}$


$$
\mathbf{v}^{π} = \mathcal{R}^{π} + \gamma \mathcal{P}^{π} \mathbf{v}^{π}
$$




#### Optimal Policy and Value Function



- How to evaluate a state?
  - Its value function

- How to evaluate a policy?
  - Its value function of every state



Define a partial ordering over policy 
$$
\pi \geq \pi^{\prime} \text { if } v_{\pi}(s) \geq v_{\pi^{\prime}}(s), \forall s
$$


- There exists an optimal policy $π_*$
- All optimal policies achieve the optimal value function and action-value function
- There is always a deterministic optimal policy.



#### Bellman Optimality Equation


$$
v_{*}(s)=\max _{a} q_{*}(s, a)
$$

$$
q_{*}(s, a)=\mathcal{R}_{s}^{a}+\gamma \sum_{s^{\prime} \in \mathcal{S}} \mathcal{P}_{s s^{\prime}}^{a} v_{*}\left(s^{\prime}\right)
$$

To seperate $v(s)$ and $q(s,a)$
$$
v_{*}(s)=\max _{a} \mathcal{R}_{s}^{a}+\gamma \sum_{s^{\prime} \in \mathcal{S}} \mathcal{P}_{s s^{\prime}}^{a} v_{*}\left(s^{\prime}\right)
$$

$$
q_{*}(s, a)=\mathcal{R}_{s}^{a}+\gamma \sum_{s^{\prime} \in \mathcal{S}} \mathcal{P}_{s s^{\prime}}^{a} \max _{a^{\prime}} q_{*}\left(s^{\prime}, a^{\prime}\right)
$$

Non-linear and no closed-form solution.



## Introduction to Dynamic Programming



A paradigm for solving problems by breaking into subproblems



### When to use DP



- Optimal substructure
  - Bellman Optimality Equation
- Overlapping subproblems
  - Bellman Expectation Equation



### DP in Markov Decision Process

- DP requires full knowledge of MDP
- model-based method

#### Prediction: Policy Evaluation

- Given policy $π$, the MDP $\langle\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, γ \rangle$ is a MRP $\left\langle\mathcal{S}, \mathcal{P}^{\pi}, \mathcal{R}^{\pi}, \gamma\right\rangle$
- Prediction focuses on evaluating the policy $π$ and calculating the value function $\mathbf{v}_{π}$
- Using Bellman Expectation Equation

#### Control: Policy Improvement

- Control focuses on how to improve the policy and find the optimal policy and its value function
  - Policy Iteration
  - Value Iteration
- Using both Bellman Expectation Equation and Bellman Optimality Equation



## Policy Evaluation





- Input: $\langle\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, γ \rangle$  and policy $π $
  -   $\left\langle\mathcal{S}, \mathcal{P}^{\pi}, \mathcal{R}^{\pi}, \gamma\right\rangle$
- Output: $\mathbf{v}_{π }$



Using Bellman Expectation Equation
$$
\mathbf{v}_{π} = \mathcal{R}^{π} + \gamma \mathcal{P}^{π} \mathbf{v}_{π}
$$

- Non-iterative method $\mathbf{v}_{π} = (I-γ\mathcal{P}^{π})^{-1} \mathcal{R}^{π}$

- Iterative method
  $$
  \mathbf{v}_{π}^{k+1} = \mathcal{R}^{π} + \gamma \mathcal{P}^{π} \mathbf{v}_{π}^{k}
  $$
  

### Algorithm

- Input: $\left\langle\mathcal{S}, \mathcal{P}^{\pi}, \mathcal{R}^{\pi}, \gamma\right\rangle$ 
- Initial: $\mathbf{v}_{π}^0$
- At each Iteration $k$
  - $\mathbf{v}_{π}^{k+1} = \mathcal{R}^{π} + \gamma \mathcal{P}^{π} \mathbf{v}_{π}^{k}$
- Output: $\mathbf{v}_{π}$

Convergence can be guaranteed because all the eigenvalues of $γ\mathcal{P}^{π}$ are between $-1$ and $1$



## Policy Iteration



Policy Iteration can be divided into two parts 

- Policy evaluation
- Policy Improvement



### Policy Improvement



Question: Given $π$ and $\mathbf{v}_{π}$, find a better policy $π'$



Strategy: Greedy Search

- Given $π$ and $\mathbf{v}_{π}$, $q_{π}(s,a)$ 
- Consider a deterministic policy $π'$
  - Because $v(s) = \sum_{a} π(a\mid s)q_{π}(s,a) \leqslant \max_{a} q_{π}(s,a) $
  - By setting $π'(s) = \text{argmax}_{a} q_{π}(s,a)$,
  - $v_{π'}(s) = \max_{a} q_{π}(s,a) \geqslant v_{π}(s)$



### Algorithm



- Input:  MDP $\langle\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, γ \rangle$
- Initial: $π_0$
- At each iter $k$
  - Evaluate the policy $π_k$, calculate $q_{π_k}(s,a)$
  - Update policy $π_{k+1}(s) = \text{argmax}_{a} q_{π_k}(s,a)$
- Output: Optimal policy and value function $π_*$ and $\mathbf{v}_*$



### Notes



- Proof of Convergence

  - The value function is improved at each iteration

  - $$
    \begin{aligned} v_{\pi}(s) & \leq q_{\pi}\left(s, \pi^{\prime}(s)\right)=\mathbb{E}_{\pi^{\prime}}\left[R_{t+1}+\gamma v_{\pi}\left(S_{t+1}\right) \mid S_{t}=s\right] \\ & \leq \mathbb{E}_{\pi^{\prime}}\left[R_{t+1}+\gamma q_{\pi}\left(S_{t+1}, \pi^{\prime}\left(S_{t+1}\right)\right) \mid S_{t}=s\right] \\ & \leq \mathbb{E}_{\pi^{\prime}}\left[R_{t+1}+\gamma R_{t+2}+\gamma^{2} q_{\pi}\left(S_{t+2}, \pi^{\prime}\left(S_{t+2}\right)\right) \mid S_{t}=s\right] \\ & \leq \mathbb{E}_{\pi^{\prime}}\left[R_{t+1}+\gamma R_{t+2}+\ldots \mid S_{t}=s\right]=v_{\pi^{\prime}}(s) \end{aligned}
    $$

  - And the improvement stops only if it achieves the optimal.

- Discussion of stopping condition
- The algorithm contains two levels of iteration
  - The policy iteration 
    - policy evaluation
  - Why not update policy after each iteration of evaluation
    - which leads to **value iteration**
- Generalization:
  - <img src="https://raw.githubusercontent.com/hanmochen/Pictures/master/20200924100029.png" style="zoom:25%;" />
  - <img src="https://raw.githubusercontent.com/hanmochen/Pictures/master/20200924100112.png" alt="image-20200924100112341" style="zoom:25%;" />







## Value Iteration



Problem: Try to directly find the optimal value function instead of the optimal policy.





### Recap: Bellman Optimality Equation



It consists of two parts:

- Maximisation:

$$
v_{*}(s)=\max _{a} q_{*}(s, a)
$$

- Expectation:

$$
q_{*}(s, a)=\mathcal{R}_{s}^{a}+\gamma \sum_{s^{\prime} \in \mathcal{S}} \mathcal{P}_{s s^{\prime}}^{a} v_{*}\left(s^{\prime}\right)
$$





### Algorithm



- Input:  MDP $\langle\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, γ \rangle$
- Initial: $\mathbf{v}_0$ and $q_{0}(s,a)$  
- At each iter $k$
  - $v_{k+1}(s) = \max_a q_k(s,a)$
  - $q_{k+1}(s,a) =\mathcal{R}_{s}^{a}+\gamma \sum_{s^{\prime} \in \mathcal{S}} \mathcal{P}_{s s^{\prime}}^{a} v_{k+1}\left(s^{\prime}\right)$

- Output: the optimal value function $\mathbf{v}_{*}$

## Discussion



### Async Dynamic Programming

- In-place Dynamic Programming

  -  Sync: 

  - $$
    v_{n e w}(s) \leftarrow \max _{a \in \mathcal{A}}\left(\mathcal{R}_{s}^{a}+\gamma \sum_{s^{\prime} \in \mathcal{S}} \mathcal{P}_{s s^{\prime}}^{a} v_{o l d}\left(s^{\prime}\right)\right)
    $$

  - Async:

  - $$
    v(s) \leftarrow \max _{a \in \mathcal{A}}\left(\mathcal{R}_{s}^{a}+\gamma \sum_{s^{\prime} \in \mathcal{S}} \mathcal{P}_{s s^{\prime}}^{a} v\left(s^{\prime}\right)\right)
    $$

- Prioritised Sweeping:

  - Use magnitude of Bellman error to guide state selection:

    - $$
      \left|\max _{a \in \mathcal{A}}\left(\mathcal{R}_{s}^{a}+\gamma \sum_{s^{\prime} \in \mathcal{S}} \mathcal{P}_{s s^{\prime}}^{a} v\left(s^{\prime}\right)\right)-v(s)\right|
      $$

  - Update the state with bigggest error

- Real-Time Dynamic Programming
- Full-Width Backups and Sample Backups



### Convergence: contraction mapping



>  Contraction Mapping Theorem:
>
> For any metric space $\mathcal{V}$ that is complete (i.e. closed) under an operator $T ( v )$ , where $T$ is a $γ$-contraction,
>
> 1. $T$ converges to a unique ﬁxed point
> 2. At a linear convergence rate of $γ$





Bellman Expectation Operator

- Deﬁne the Bellman expectation backup operator $T^{\pi}$

  - $T^{\pi}(v)=\mathcal{R}^{\pi}+\gamma \mathcal{P}^{\pi} v$

- This operator is a $γ$-contraction

  - $$
    \begin{aligned}\left.\left|T^{\pi}(u)-T^{\pi}(v)\right|\right|_{\infty} &=\left\|\left(\mathcal{R}^{\pi}+\gamma \mathcal{P}^{\pi} u\right)-\left(\mathcal{R}^{\pi}+\gamma \mathcal{P}^{\pi} v\right)\right\|_{\infty} \\ &=\left\|\gamma \mathcal{P}^{\pi}(u-v)\right\|_{\infty} \\ & \leq\left\|\gamma \mathcal{P}^{\pi}\right\| u-v\left\|_{\infty}\right\|_{\infty} \\ & \leq \gamma\|u-v\|_{\infty} \end{aligned}
    $$

- 







Bellman Optimality Operator

- Deﬁne the Bellman optimality backup operator $T^*$
  - $T^{*}(v)=\max _{a \in \mathcal{A}} \mathcal{R}^{a}+\gamma \mathcal{P}^{a} v$
- This operator is a $γ$ -contraction
  - $\left\|T^{*}(u)-T^{*}(v)\right\|_{\infty} \leq \gamma\|u-v\|_{\infty}$





### Summary



- MDP and Bellman Equation

- Use DP for prediction and control

- Policy Evaluation

- Policy Iteration

- Value Iteration

  