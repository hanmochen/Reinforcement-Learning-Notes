# Reinforcement Learning Notes

[toc]

# Lec 1 Introduction



## The RL settings

### Rewards

- A reward $R_t$ is a **scalar** feedback signal
- **Reward Hypothesis**:
  - All goals can be described by the maximization of expected culumative reward
  - Notes:
    - **maximization**
    - **Expectation**
    - **Cumulative**
    - How to deal with multiple goals?
- Sequential Decision Making: to maximize total future reward
  - Reward may be delayed, which implies that sometimes immediate reward may be sacrificed got more long-term reward



### Agent and Environment

<img src="https://raw.githubusercontent.com/hanmochen/Pictures/master/20210227162725.png" alt="image-20210227162659124" style="zoom:25%;" />

At each timestep $t$

the agent:

- Do action $a_t$
- Observe $o_t$
- Receice $r_t$

the env:

- Receive $a_t$
- Emits $o_t$ and $r_t$



### State



**History**:

- the sequence of observations, actions, rewards $o_1,r_1,a_1,o_2,r_2,a_2,\cdots,o_t,r_t$
- The agent uses history to decide $a_t$

**State**:

- Generally, the agent don't use all the information in the history to make decisions
- **State** is the information used to determine what happens next
- i.e. The state is a **suﬃcient statistic** for the future
- Formally, state is a function of the history
  - $S_t = f(H_t)$

**Environment State and Agent State**:

- environment state $S_t^e$: environment's private representation
  - usually not visible to the agent
- agent state $S_t^a$: agent's internal representation

**Information State**:

- contains all useful information from the history
- aka **Markov State**
  - Given $S_t$, the future is independent of the history $H_t$
- Once the state is known, the history may be thrown away

**Observability**:

- Fully Obervation Environment:
  - $O_t=S_t^a=S_t^e$
  - **Markov Decision Process(MDP)**
- Partial observability:
  - **POMDP**
  - Agent must construct its own state representation
  - How?
    - Using complete history
    - Using **beliefs** of environment state
    - Using RNN





### RL agent

**major components**

- Policy: makes decision 
- Value function: evaluates each state and/or action
- Model: model of the environment

**Policy**

- Tells the agent how to behave
- maps from state to action
  - Deterministic: $a_t=π(s_t)$
  - Stochastic: $π(a|s) = P(A_t=a|S_t=s)$

**Value Function**

- Evaluate states/actions
- State value functions:

$$
v_{\pi}(s)=\mathbb{E}_{\pi}\left[R_{t+1}+\gamma R_{t+2}+\gamma^{2} R_{t+3}+\ldots \mid S_{t}=s\right]
$$

- State-action value functions:

$$
v(a|s)=\mathbb{E}\left[R_{t+1}+\gamma R_{t+2}+\gamma^{2} R_{t+3}+\ldots \mid A_t=a,S_{t}=s\right]
$$

**Model**:

- the model of environment interaction
- consists of two parts
  - $\mathcal{P}$: predicts the state, i.e. state transision probability matrix
  - $\mathcal{R}$: predict the reward, i.e. the reward distribution

**Example: Maze**

**Categorizing RL agents**

-  With/without Model: Model based/model free
-  Value based: value function and no policy
-  Policy based: policy and no value function
-  Actor-Critic: value function and policy



## Problems within RL

### Learning and planning

- Learning:
  - the model is unknown
  - interacts with the environment while improving its policy
- Planning
  - model is known
  - do not interact with the environment while improving its policy



### Exploration and Exploitation

- Reinforcement learning is like trial-and-error
- The agent should learn about the environment as well as gain more reward.
- Exploration ﬁnds more information about the environment
- Exploitation uses known information to maximise reward



### Prediction and Control

- Prediction: predict the state given a policy
- Control: find the best policy given the state

# Lec 2  MDP



**Why do we study Markov Decision Process in Reinforcement Learning?**

- Markov Property is a **good** property for RL
  - Simplify the history with the current state

- When the environment is fully observable, it can be described as a Markov Decision Process
- When it is partially observable, it can be converted into MDPs(POMDPs).

## Markov Process

### Markov Property

- The future is independent of the past given the present

A state $S_t$ is Markov if and only if
$$
\mathbb{P}\left[S_{t+1} \mid S_{t}\right]=\mathbb{P}\left[S_{t+1} \mid S_{1}, \ldots, S_{t}\right]
$$

### Markov Process

Markov Process (or Markov Chain) is a tuple $<\mathcal{S},\mathcal{P}>$

- $\mathcal{S}$: state space
- $\mathcal{P}$: state transition probability matrix

### Markov Reward Process

Markov Reward Process is a tuple $<\mathcal{S},\mathcal{P},\mathcal{R},\gamma>$

- $\mathcal{S}$: state space
- $\mathcal{P}$: state transition probability matrix
- $\mathcal{R}$: reward function. $\mathcal{R}_s = \mathbb{E}[R_{t+1}|S_t=s]$
- $γ$: discount factor

**Return** $G_t$: the total discounted reward from time-step $t$
$$
G_{t}=R_{t+1}+\gamma R_{t+2}+\ldots=\sum_{k=0}^{\infty} \gamma^{k} R_{t+k+1}
$$
Why Discount?

- Mathematically: for convergence and convenience
- Financially: interest rate
- Uncertainty about the future and preference for immediate reward

Also, for episodic MDPs, $γ$ can be 1.

**Value Function**
$$
v(s)=\mathbb{E}\left[G_{t} \mid S_{t}=s\right]
$$


## Markov Decision Process

### Definition

Markov Decision Process is a tuple $<\mathcal{S},\mathcal{A},\mathcal{P},\mathcal{R},γ>$

- $\mathcal{S}$: state space
- $\mathcal{A}$: action space
- $\mathcal{P}$: state transition probability matrix $\mathcal{P}_{s s^{\prime}}^{a}=\mathbb{P}\left[S_{t+1}=s^{\prime} \mid S_{t}=s, A_{t}=a\right]$
- $\mathcal{R}$: reward function. $\mathcal{R}_{s}^{a}=\mathbb{E}\left[R_{t+1} \mid S_{t}=s, A_{t}=a\right]$
- $γ$: discount factor

**Policy**

Policy $π$ is a distribution over actions given states.
$$
\pi(a|s) = P(A_t=a| S_t=s)
$$
With a fixed policy, MDP is reduced to Markov Reward Process  $<\mathcal{S},\mathcal{P}^π,\mathcal{R}^π,\gamma>$
$$
\begin{aligned} \mathcal{P}_{s, s^{\prime}}^{\pi} &=\sum_{a \in \mathcal{A}} \pi(a \mid s) \mathcal{P}_{s s^{\prime}}^{a} \\ \mathcal{R}_{s}^{\pi} &=\sum_{a \in \mathcal{A}} \pi(a \mid s) \mathcal{R}_{s}^{a} \end{aligned}
$$

### Value Function and Bellman Equation

state-value function
$$
v_{\pi}(s)=\mathbb{E}_{\pi}\left[G_{t} \mid S_{t}=s\right]
$$
action-state value function
$$
q_{\pi}(s, a)=\mathbb{E}_{\pi}\left[G_{t} \mid S_{t}=s, A_{t}=a\right]
$$
Relationship between $v_π(s)$ and $q_π(s,a)$
$$
v_{\pi}(s) = \sum_{a\in \mathcal{A}} \pi(a|s) q_{\pi}(s,a)
$$
**Bellman Expectation Equation**

Because
$$
q_{\pi}(s, a)=\mathcal{R}_{s}^{a}+\gamma \sum_{s^{\prime} \in \mathcal{S}} \mathcal{P}_{s s^{\prime}}^{a} v_{\pi}\left(s^{\prime}\right)
$$
Using (7) (8) we have
$$
v_{\pi}(s)=\sum_{a \in \mathcal{A}} \pi(a \mid s)\left(\mathcal{R}_{s}^{a}+\gamma \sum_{s^{\prime} \in \mathcal{S}} \mathcal{P}_{s s^{\prime}}^{a} v_{\pi}\left(s^{\prime}\right)\right) \\
q_{\pi}(s, a)=\mathcal{R}_{s}^{a}+\gamma \sum_{s^{\prime} \in \mathcal{S}} \mathcal{P}_{s s^{\prime}}^{a} \sum_{a^{\prime} \in \mathcal{A}} \pi\left(a^{\prime} \mid s^{\prime}\right) q_{\pi}\left(s^{\prime}, a^{\prime}\right)
$$
the matrix form 
$$
v_{\pi}=\mathcal{R}^{\pi}+\gamma \mathcal{P}^{\pi} v_{\pi}
$$


### Optimal Policy and Value Function

- Define **Optimal**

  - Deﬁne a partial ordering over policies $π>π'$ if $v_{π}(s) \geqslant v_{π'}(s),\forall s \in \mathcal{S}$ 

- Theorem: For any Markov Decision Process

  - There exists an optimal policy

  - All optimal policies achieve the optimal value function and action-value function

  - There is always a deterministic optimal policy, that is 

    - $$
      v_{*}(s)=\max _{a} q_{*}(s, a)
      $$

    - 

    - $$
      \pi_{*}(a \mid s)=\left\{\begin{array}{ll}1 & \text { if } a=\underset{a \in \mathcal{A}}{\operatorname{argmax}} q_{*}(s, a) \\ 0 & \text { otherwise }\end{array}\right.
      $$

      

Bellman Optimality Function
$$
v_{*}(s)=\max _{a} \mathcal{R}_{s}^{a}+\gamma \sum_{s^{\prime} \in \mathcal{S}} \mathcal{P}_{s s^{\prime}}^{a} v_{*}\left(s^{\prime}\right) \\
q_{*}(s, a)=\mathcal{R}_{s}^{a}+\gamma \sum_{s^{\prime} \in \mathcal{S}} \mathcal{P}_{s s^{\prime}}^{a} \max _{a^{\prime}} q_{*}\left(s^{\prime}, a^{\prime}\right)
$$


### Extensions to MDPs

- Inﬁnite MDPs
  - Countably inﬁnite state and/or action spaces
  - Continuous state and/or action spaces
    - Closed form for linear quadratic model (LQR)
  - Continuous time
    - Partial Differential Equations
    - Hamilton-Jacobi-Bellman Equation
    - Limiting case of Bellman Equation as $Δt\to 0$
- POMDP(Partially Oberservable Markov Decision Process)
  - POMDP is a tuple $<\mathcal{S},\mathcal{A},\mathcal{O},\mathcal{P},\mathcal{R},\mathcal{Z},γ>$
    - $\mathcal{S}$: state space
    - $\mathcal{A}$: action space
    - $\mathcal{O}$: observation space
    - $\mathcal{P}$: state transition probability matrix $\mathcal{P}_{s s^{\prime}}^{a}=\mathbb{P}\left[S_{t+1}=s^{\prime} \mid S_{t}=s, A_{t}=a\right]$
    - $\mathcal{R}$: reward function. $\mathcal{R}_{s}^{a}=\mathbb{E}\left[R_{t+1} \mid S_{t}=s, A_{t}=a\right]$
    - $\mathcal{Z}:$ obervation function $\mathcal{Z}_{s^{\prime} o}^{a}=\mathbb{P}\left[O_{t+1}=o \mid S_{t+1}=s^{\prime}, A_{t}=a\right]$
    - $γ$: discount factor
  - Belief States
    - History $H_t$: $H_{t}=A_{0}, O_{1}, R_{1}, \ldots, A_{t-1}, O_{t}, R_{t}$
    - A belief state $b(h)$ is a probability distribution over states conditioned on the history $h$, $b(h)=\left(\mathbb{P}\left[S_{t}=s^{1} \mid H_{t}=h\right], \ldots, \mathbb{P}\left[S_{t}=s^{n} \mid H_{t}=h\right]\right)$
    - The history $H_t$ satisfies the Markov Property
    - The belief state $b(H_t)$ satisﬁes the Markov property
  - So POMDP can be reduced to an history tree or a belief state tree
- Ergodic MDP
  - An ergodic Markov process has a stationary distribution $d^{\pi}(s)=\sum_{s^{\prime} \in \mathcal{S}} d^{\pi}\left(s^{\prime}\right) \mathcal{P}_{s^{\prime} s}$ 
  - An MDP is ergodic if the Markov chain induced by any policy is ergodic.
  - Average reward $\rho^{\pi}=\lim _{T \rightarrow \infty} \frac{1}{T} \mathbb{E}\left[\sum_{t=1}^{T} R_{t}\right]$


# Lecture 3 Dynamic Programming



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

  - Sync: 

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
>  For any metric space $\mathcal{V}$ that is complete (i.e. closed) under an operator $T ( v )$ , where $T$ is a $γ$-contraction,
>
>  1. $T$ converges to a unique ﬁxed point
>  2. At a linear convergence rate of $γ$





Bellman Expectation Operator

- Deﬁne the Bellman expectation backup operator $T^{\pi}$

  - $T^{\pi}(v)=\mathcal{R}^{\pi}+\gamma \mathcal{P}^{\pi} v$

- This operator is a $γ$-contraction

  - $$
    \begin{aligned}\left.\left|T^{\pi}(u)-T^{\pi}(v)\right|\right|_{\infty} &=\left\|\left(\mathcal{R}^{\pi}+\gamma \mathcal{P}^{\pi} u\right)-\left(\mathcal{R}^{\pi}+\gamma \mathcal{P}^{\pi} v\right)\right\|_{\infty} \\ &=\left\|\gamma \mathcal{P}^{\pi}(u-v)\right\|_{\infty} \\ & \leq\left\|\gamma \mathcal{P}^{\pi}\right\| u-v\left\|_{\infty}\right\|_{\infty} \\ & \leq \gamma\|u-v\|_{\infty} \end{aligned}
    $$

  








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




# Lec 4 MC-TD



Recap:

- Last Lecture: Dynamic Programming
  - Model-based Prediction and Control
- This Lecture: model-free prediction(evaluation)
  - Estimate the value function of an unknown MDP
  - Monte-Carlo Learning
  - Temporal-Difference Learning
- Next Lecture: model-free control
  - optimize the value function of an unknown MDP



## Monte-Carlo Learning

- model free: no knowledge of MDP
- learn directly from episodes of experience
- learns from complete episodes: no bootstrapping
- can only be applied to episodic MDPs



### First-Visit Policy Evaluation

To evaluate state $s$

- The ﬁrst time that $s$ is visited in an episode
  - $N(s) \leftarrow N(s)+1$
  - $S(s)\leftarrow S(s)+G_t$
- $V(s) = S(s)/N(s)$



### Every-Visit Policy Evaluation

To evaluate state $s$

- Every time that $s$ is visited in an episode
  - $N(s) \leftarrow N(s)+1$
  - $S(s)\leftarrow S(s)+G_t$
- $V(s) = S(s)/N(s)$



### Incremental Update of Mean

- $N(S_t) \leftarrow N(S_t)+1$
- $V(S_t)\leftarrow V(S_t)+\frac{1}{N(S_t)}(G_t-V(S_t))$ or in non-stationary problems $V(S_t)\leftarrow V(S_t)+\alpha(G_t-V(S_t))$





## Temporal-Difference Learning

- model free: no knowledge of MDP
- learn directly from episodes of experience
- learns from incomplete episodes: bootstrapping
- updates a guess based upon a guess



### TD(0)

- In MC, we use empirical return to estimate $G_t$

- In TD(0), we use $G_t=R_{t+1}+γV(s_{t+1})$ to estimate $G_t$, where

  - $R_{t+1}$ uses the empirical reward
  - $V(s_{t+1})$ uses the estimated value function

- Update value function based on estimation
  $$
  V\left(S_{t}\right) \leftarrow V\left(S_{t}\right)+\alpha\left(R_{t+1}+\gamma V\left(S_{t+1}\right)-V\left(S_{t}\right)\right)
  $$

  - $R_{t+1}+γV(s_{t+1}):$ TD target
  - $R_{t+1}+γV(s_{t+1})-V(S_t):$ TD error



### MC vs. TD



- Bias/Variance Trade-Off
  - Empirical Return $G_t$ is unbiased estimator of $v_{π}(S_t)$
  - $R_{t+1}+v_{π}(S_{t+1})$ is also unbiased
  - $R_{t+1}+\hat v(S_{t+1})$ is biased, but asymptotic unbiased.
  - But $R_{t+1}+\hat v(S_{t+1})$ reduces the variance
    - Because it depends only on $a_t,S_{t+1},R_{t+1}$

- MC:
  - unbiased, high variance
  - Good Convergence(even with function approximation)
  - Not sensitive to initial value
  - only learn from complete sequences
    - episodic MDPs
    - can not learn only
- TD
  - biased. low variance
  - TD(0) converges(but not with function approximation)
  - Sensitive to initial value
  - learn from incomplete sequences
    - non-terminating MDPs
    - can learn online





AB Example:

> Two state A and B, 8 episodes, $γ=1$
>
> 1. A,0,B,0
> 2. B,1
> 3. B,1
> 4. B,1
> 5. B,1
> 6. B,1
> 7. B,1
> 8. B,0
>
> Estimate $v(A)$ and $v(B)$



- Using MC, $v(A)=0,v(B)=6/8=0.75$
- Using TD, the model is like 

<img src="https://raw.githubusercontent.com/hanmochen/Pictures/master/20210304194516.png" alt="image-20210304194516156" style="zoom:25%;" />

so $v(A) = v(B)=0.75$



Discussion:

- MC doesn't account for the Markov property (state transition)

  - converges to the model with minimum mean-squared error

  - Fit the model to minimize

  - $$
    \sum_{k=1}^{K} \sum_{t=1}^{T_{k}}\left(G_{t}^{k}-V\left(s_{t}^{k}\right)\right)^{2}
    $$

- TD(0) consider the Markov property

  - converges to the model with max likelihood

  - Fit the MDP model

  - $$
    \begin{aligned} \hat{\mathcal{P}}_{s, s^{\prime}}^{a} &=\frac{1}{N(s, a)} \sum_{k=1}^{K} \sum_{t=1}^{T_{k}} \mathbf{1}\left(s_{t}^{k}, a_{t}^{k}, s_{t+1}^{k}=s, a, s^{\prime}\right) \\ \hat{\mathcal{R}}_{s}^{a} &=\frac{1}{N(s, a)} \sum_{k=1}^{K} \sum_{t=1}^{T_{k}} 1\left(s_{t}^{k}, a_{t}^{k}=s, a\right) r_{t}^{k} \end{aligned}
    $$

    

- If the environment is Markov and the dataset is big enough, MC and TD(0) converges to the same result.



Monte-Carlo Backup
$$
V\left(S_{t}\right) \leftarrow V\left(S_{t}\right)+\alpha\left(G_{t}-V\left(S_{t}\right)\right)
$$


<img src="https://raw.githubusercontent.com/hanmochen/Pictures/master/20210304200214.png" alt="image-20210304200214650" style="zoom:50%;" />



Temporal-Difference Backup
$$
V\left(S_{t}\right) \leftarrow V\left(S_{t}\right)+\alpha\left(R_{t+1}+\gamma V\left(S_{t+1}\right)-V\left(S_{t}\right)\right)
$$
<img src="https://raw.githubusercontent.com/hanmochen/Pictures/master/20210304200340.png" alt="image-20210304200340603" style="zoom:50%;" />

Dynamic Programming Backup
$$
V\left(S_{t}\right) \leftarrow \mathbb{E}_{\pi}\left[R_{t+1}+\gamma V\left(S_{t+1}\right)\right]
$$
<img src="https://raw.githubusercontent.com/hanmochen/Pictures/master/20210304200521.png" alt="image-20210304200521672" style="zoom:50%;" />

- Boostrapping: Estimate based on estimation (Depth)
- Sampling: Using one episode instead of all to update (Width)



<img src="../Library/Mobile Documents/Application Support/typora-user-images/image-20210304202528854.png" alt="image-20210304202528854" style="zoom:50%;" />



### TD(λ)

**n-step TD target**
$$
G_{t}^{(n)}=R_{t+1}+\gamma R_{t+2}+\ldots+\gamma^{n-1} R_{t+n}+\gamma^{n} V\left(S_{t+n}\right)
$$

$$
\begin{array}{l}G_{t}^{(1)}=R_{t+1}+\gamma V\left(S_{t+1}\right) \\ G_{t}^{(2)}=R_{t+1}+\gamma R_{t+2}+\gamma^{2} V\left(S_{t+2}\right) \\ \vdots \\ G_{t}^{(\infty)}=R_{t+1}+\gamma R_{t+2}+\ldots+\gamma^{T-t-1} R_{T}\end{array}
$$

when $n\to \infty$ TC turns into MC

 **Averaging n-Step returns: λ-return**

$$
G_{t}^{\lambda}=(1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_{t}^{(n)}
$$
**Forward-view TD(λ)**
$$
V\left(S_{t}\right) \leftarrow V\left(S_{t}\right)+\alpha\left(G_{t}^{\lambda}-V\left(S_{t}\right)\right)
$$

- Update value function use λ-return
- Use the future to compute $G_t^{\lambda}$
- can only be computed from complete episodes like MC

<img src="https://raw.githubusercontent.com/hanmochen/Pictures/master/20210305185819.png" alt="image-20210305185734887" style="zoom:50%;" />

**Backward-view TD(λ)**

- Forward-view provides theory while backward provides mechanism
- Update online from incomplete sequences

<img src="https://raw.githubusercontent.com/hanmochen/Pictures/master/20210305190105.png" alt="image-20210305190105246" style="zoom:50%;" />



**Eligibility Traces**

- Credit Assignment Problem
  - Frequency Heuristic
  - Recency Heuristic
- Combine Frequecy and Recency: Eligibility Trace
  - $E_0(s)=0,E_t(s) = γλE_t(s) + \mathbf{1}(S_t=s)   $
  - $α=γλ=1$: frequency; 0, recency



**Apply Eligibility Trace to Backward-view TD(λ)**

- Maintian an eligibility trace $E_t(s)$ for every state $s$ 
- Update value $V(s)$ online for every state $s$ after each step
- The Increment comes from
  - The TD-error $δ_t=R_{t+1}+\gamma V\left(S_{t+1}\right)-V\left(S_{t}\right)$
  - The eligibility trace $E_t(s)$
- $V(s) \leftarrow V(s)+\alpha \delta_{t} E_{t}(s)$



**Forward vs. Backward**



- $λ=0$：backward is equivalent to forward TD(0)
- $λ=1$：TD(1) is roughly equivalent to every-visit MC
  - If value function is only updated offline(at the end of episode), TD(1) is the same as MC
- General λ:
  - TD errors telescopes to λ-error $G_t^{λ}-V(S_t)$ 





**Offline vs. Online**

Online updates:

- Updates are applied online at each step
- Forward and backward are slightly different

Offline updates:

- Updates are cumulated within episode
- but applied only at the end of episode



**Summary**



<img src="https://raw.githubusercontent.com/hanmochen/Pictures/master/20210305215147.png" alt="image-20210305215147463" style="zoom:50%;" />



# Lec 5 Model-free Control



**On-Policy vs. Off-Policy**

- Policy we try to learn(optimize) π 
- Policy we use to learn from expererience μ
- On-Policy: $π=μ$; Off-policy: $π\neq μ $   



## On-Policy Monte-Carlo Control

### Generalized Policy Iteration



<img src="https://raw.githubusercontent.com/hanmochen/Pictures/master/20210306151431.png" alt="image-20210306151431254" style="zoom: 33%;" />

**Policy Evaluation**: Estimate $v_π$

**Policy Improvement**: Find $π'\geqslant π$



In Monte-Carlo control, 

- For policy evaluation, use Monte-Carlo evaluation
- For policy improvement, use greedy policy improvement



Greedy policy improvement over $V(s)$
$$
\pi^{\prime}(s)=\underset{a \in \mathcal{A}}{\operatorname{argmax}} \mathcal{R}_{s}^{a}+\mathcal{P}_{s s^{\prime}}^{a} V\left(s^{\prime}\right)
$$
But it requires MDP model, and Greedy policy improvement over action-value function $Q(s,a)$ is model-free
$$
\pi^{\prime}(s)=\underset{a \in \mathcal{A}}{\operatorname{argmax}} Q(s, a)
$$


$ε-$greedy Exploration

- With prob $1-ε$ choose greedy action
- With prob ε choose a random action

$$
\pi(a \mid s)=\left\{\begin{array}{ll}\epsilon / m+1-\epsilon & \text { if } a^{*}=\underset{a \in \mathcal{A}}{\operatorname{argmax}} Q(s, a) \\ \epsilon / m & \text { otherwise }\end{array}\right.
$$



Improvement Theorem

For any ε-greedy policy $π$, the ε-greedy policy $π'$ using $q_π$ satisfies $v_π'(s) \geqslant v_π(s)$
$$
\begin{aligned} q_{\pi}\left(s, \pi^{\prime}(s)\right) &=\sum_{a \in \mathcal{A}} \pi^{\prime}(a \mid s) q_{\pi}(s, a) \\ &=\epsilon / m \sum_{a \in \mathcal{A}} q_{\pi}(s, a)+(1-\epsilon) \max _{a \in \mathcal{A}} q_{\pi}(s, a) \\ & \geq \epsilon / m \sum_{a \in \mathcal{A}} q_{\pi}(s, a)+(1-\epsilon) \sum_{a \in \mathcal{A}} \frac{\pi(a \mid s)-\epsilon / m}{1-\epsilon} q_{\pi}(s, a) \\ &=\sum_{a \in \mathcal{A}} \pi(a \mid s) q_{\pi}(s, a)=v_{\pi}(s) \end{aligned}
$$


### Monte Carlo Control

**Monte Carlo Policy Iteration**



<img src="https://raw.githubusercontent.com/hanmochen/Pictures/master/20210306162357.png" alt="image-20210306162357730" style="zoom:33%;" />





**Monte-Carlo Control**



<img src="https://raw.githubusercontent.com/hanmochen/Pictures/master/20210306162527.png" alt="image-20210306162527113" style="zoom: 33%;" />



### Convergence

How to make sure it converges to $π^*,q^*$?

We have GLIE (Greedy in the Limit with Infinite Exploration)

- All state-action pairs are explored infinitely many times $\lim_{k\to\infty}N_k(s,a) = \infty$
- The policy converges on a greedy policy $\lim _{k \rightarrow \infty} \pi_{k}(a \mid s)=\mathbf{1}\left(a=\underset{a^{\prime} \in \mathcal{A}}{\operatorname{argmax}} Q_{k}\left(s, a^{\prime}\right)\right)$

For example, if we let ε reduces to 0 like $ε_k=\frac 1 k$, the ε-greedy is GILE



**GLIE Monte-Carlo control**

At iter $k$

- Sample episode using $π$ $\left\{S_{1}, A_{1}, R_{2}, \ldots, S_{T}\right\} \sim \pi$
- For each state $S_t$ and action $A_t$:
  - $N\left(S_{t}, A_{t}\right) \leftarrow N\left(S_{t}, A_{t}\right)+1$
  - $Q\left(S_{t}, A_{t}\right) \leftarrow Q\left(S_{t}, A_{t}\right)+\frac{1}{N\left(S_{t}, A_{t}\right)}\left(G_{t}-Q\left(S_{t}, A_{t}\right)\right)$
- Improve policy based on $Q(S,A)$
  - $\varepsilon \leftarrow 1/k$
  - $π \leftarrow ε-$greedy(Q)

Theorem:

GLIE Monte-Carlo control converges to the optimal action-value function $q_*(s,a)$



## On-Policy Temporal-Difference Control

TD has several advantages over MC

- lower variance
- online
- incomplete sequences

Apply TD instead of MC for control, we have Sarsa(λ)



### Sarsa

**Updating Q(s,a)**

<img src="https://raw.githubusercontent.com/hanmochen/Pictures/master/20210306170531.png" style="zoom: 33%;" />
$$
Q(S, A) \leftarrow Q(S, A)+\alpha\left(R+\gamma Q\left(S^{\prime}, A^{\prime}\right)-Q(S, A)\right)
$$


**On-Policy Control with Sarsa**



<img src="https://raw.githubusercontent.com/hanmochen/Pictures/master/20210306170809.png" alt="image-20210306170808924" style="zoom:33%;" />

Every **time-step**(In MC,we do that every episode)

- Policy evaluation: Sarsa update
- Policy Improvement: ε-greedy



<img src="https://raw.githubusercontent.com/hanmochen/Pictures/master/20210306171354.png" alt="image-20210306171354776" style="zoom:50%;" />



### Convergence

Sarsa converges to the optimal action value function under the following conditions:

- GLIE sequence of policies $π_t(a|s)$
- Robbins-Monro sequence of step-size $α_t$
  - $\sum_{t=1}^{\infty} \alpha_{t}=\infty$
  - $\sum_{t=1}^{\infty} \alpha_{t}^{2}<\infty$





### Sarsa(λ)

n-step return 
$$
q_{t}^{(n)}=R_{t+1}+\gamma R_{t+2}+\ldots+\gamma^{n-1} R_{t+n}+\gamma^{n} Q\left(S_{t+n}\right)
$$
λ-return
$$
q_{t}^{\lambda}=(1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} q_{t}^{(n)}
$$


**Forward View Sarsa(λ)**
$$
Q\left(S_{t}, A_{t}\right) \leftarrow Q\left(S_{t}, A_{t}\right)+\alpha\left(q_{t}^{\lambda}-Q\left(S_{t}, A_{t}\right)\right)
$$
**Backward View Sarsa(λ)**

Keep an eligibility trace for each state-action pair

- $E_0(s,a)=0$
- $E_t(s,a) = \gamma \lambda E_{t-1}(s, a)+\mathbf{1}\left(S_{t}=s, A_{t}=a\right)$

Update $Q(s,a)$ from TD error and eligibility trace

- $\delta_{t}=R_{t+1}+\gamma Q\left(S_{t+1}, A_{t+1}\right)-Q\left(S_{t}, A_{t}\right)$
- $Q(s, a) \leftarrow Q(s, a)+\alpha \delta_{t} E_{t}(s, a)$



**Algorithm**

<img src="https://raw.githubusercontent.com/hanmochen/Pictures/master/20210306173931.png" alt="image-20210306173931846" style="zoom:50%;" />



## Off-policy Learning

- Episode: $\left\{S_{1}, A_{1}, R_{2}, \ldots, S_{T}\right\} \sim \mu$
- Evaluate target policy $π(a|s)$ to compute $v_π(s)$ or $q_π(s,a)$



Why do we need off-policy learning?

- Learn from other agents
- Re-use experience
- Learn about optimal policy while following exploratory policy
- Learn about multiple policies while following one policy



### Importance Sampling

Estimate the expectation of a different distribution
$$
\begin{aligned} \mathbb{E}_{X \sim P}[f(X)] &=\sum P(X) f(X) \\ &=\sum Q(X) \frac{P(X)}{Q(X)} f(X) \\ &=\mathbb{E}_{X \sim Q}\left[\frac{P(X)}{Q(X)} f(X)\right] \end{aligned}
$$


**Importance Sampling for off-policy Monte-Carlo**

- Use returns generated from μ to evaluate π

- Importance Sampling

- $$
  G_{t}^{\pi / \mu}=\frac{\pi\left(A_{t} \mid S_{t}\right)}{\mu\left(A_{t} \mid S_{t}\right)} \frac{\pi\left(A_{t+1} \mid S_{t+1}\right)}{\mu\left(A_{t+1} \mid S_{t+1}\right)} \cdots \frac{\pi\left(A_{T} \mid S_{T}\right)}{\mu\left(A_{T} \mid S_{T}\right)} G_{t}
  $$

- Update value $V\left(S_{t}\right) \leftarrow V\left(S_{t}\right)+\alpha\left(G_{t}^{\pi / \mu}-V\left(S_{t}\right)\right)$

- Problem

  - $μ$ may be zero
  - increase variance

  

**Importance Sampling for off-policy TD**

- Use TD targets generated from μ to evaluate π 

- Importance Sampling

- $$
  \begin{aligned} V\left(S_{t}\right) & \leftarrow V\left(S_{t}\right)+\\ & \alpha\left(\frac{\pi\left(A_{t} \mid S_{t}\right)}{\mu\left(A_{t} \mid S_{t}\right)}\left(R_{t+1}+\gamma V\left(S_{t+1}\right)\right)-V\left(S_{t}\right)\right) \end{aligned}
  $$



- lower variance



### Q-Learning

- We use importance sampling to update value function $V(S)$
- Now consider off-policy learning with $Q(s,a)$
- We don't need importance sampling 

$$
Q\left(S_{t}, A_{t}\right) \leftarrow Q\left(S_{t}, A_{t}\right)+\alpha\left(R_{t+1}+\gamma Q\left(S_{t+1}, A^{\prime}\right)-Q\left(S_{t}, A_{t}\right)\right)
$$

- $A'$ is decided by μ 

**Off-Policy Control With Q-Learning**

- Behaviour Policy μ  
- Target Policy π 

- If we allow both μ and π to improve
  - improve π: greedy $\pi\left(S_{t+1}\right)=\underset{a^{\prime}}{\operatorname{argmax}} Q\left(S_{t+1}, a^{\prime}\right)$
  - improve μ: ε-greedy

**Q-Learning Control Algorithm**
$$
Q(S, A) \leftarrow Q(S, A)+\alpha\left(R+\gamma \max _{a^{\prime}} Q\left(S^{\prime}, a^{\prime}\right)-Q(S, A)\right)
$$
Theorem:

Q-Learning control converges to the optimal action-value function



<img src="https://raw.githubusercontent.com/hanmochen/Pictures/master/20210307175923.png" alt="image-20210307175922967" style="zoom:50%;" />





### DP vs. TD



<img src="https://raw.githubusercontent.com/hanmochen/Pictures/master/20210307180156.png" alt="image-20210307180156330" style="zoom:50%;" />



<img src="https://raw.githubusercontent.com/hanmochen/Pictures/master/20210307180221.png" alt="image-20210307180221775" style="zoom:50%;" />



# Lec 6 Value Function Approximation



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





# Lec 7 Policy Gradient



- Last Lecture

  - we approximated the value or action-value function using parameters θ,

  - $$
    \begin{aligned} V_{\theta}(s) & \approx V^{\pi}(s) \\ Q_{\theta}(s, a) & \approx Q^{\pi}(s, a) \end{aligned}
    $$

  - And policy is generated directly from the value function (e.g. using ε-greedy)

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





# Lec 8: Integrating Learning and Planning



## Model-Based Reinforcement Learning

Model-based vs model-free RL

- Model-Free RL
  - no model
  - Learn value function or policy from experience
- Model-based RL
  - Learn a model from experience
  - Plan value function and/or policy from model



<img src="https://raw.githubusercontent.com/hanmochen/Pictures/master/20210323172452.png" alt="image-20210323172452354" style="zoom: 33%;" />



Advantages:

- Efficiency: supervised learning methods
- Uncertainty can be discussed

Disadvantages:

- Two error sources
  - learning a model
  - find its value function



### Learning a model



A model $\mathcal{M}$ is a representation of an MDP $<\mathcal{S,A,P,R}>$, parametrized by $\eta$

- Assume $\mathcal{S,A}$ are known
- To learn a model is to learn $\mathcal{P_{\eta},R_{\eta}}$

 Goal: estimate model $\mathcal{M_{\eta}}$ from experience $\{S_1,A_1,R_2,\cdots,S_T\}, which is a supervised learning problem

- $s,a\to r$ is a regression problem
- $s,a\to s'$ is a density estimation problem



**Types of Model**

- Table Lookup Model
- Linear Expectation Model
- Linear Gaussian Model
- Gaussian Process Model
- Deep Belief Network Model





**Table Lookup Model**
$$
\begin{array}{l}\hat{\mathcal{P}}_{s, s^{\prime}}^{a}=\frac{1}{N(s, a)} \sum_{t=1}^{T} \mathbf{1}\left(S_{t}, A_{t}, S_{t+1}=s, a, s^{\prime}\right) \\ \hat{\mathcal{R}}_{s}^{a}=\frac{1}{N(s, a)} \sum_{t=1}^{T} 1\left(S_{t}, A_{t}=s, a\right) R_{t}\end{array}
$$




### Planning with a Model



**Sample-based Planning**

- Simple but Powerful 
- Use model to generate samples
- Sample Experience from model 

$$
\begin{aligned} S_{t+1} & \sim \mathcal{P}_{\eta}\left(S_{t+1} \mid S_{t}, A_{t}\right) \\ R_{t+1} &=\mathcal{R}_{\eta}\left(R_{t+1} \mid S_{t}, A_{t}\right) \end{aligned}
$$

- Apply model-free RL to samples

  

**Planning with an inaccurate model**

- When the model is inaccurate, the planning will lead to a suboptimal policy.
- Solution:
  - use model-free
  - reason about model uncertainty



## Integrated Architectures



### Dyna

**Real and Simulated Experience**

- Real Experience: from real environment
- Simulated Experience: from model



**Dyna**

- Learn a model from real experience
- Learn and plan value function and policy from real and simulated experience



<img src="https://raw.githubusercontent.com/hanmochen/Pictures/master/20210323211609.png" alt="image-20210323211609488" style="zoom:33%;" />

**Dyna-Q** 



<img src="https://raw.githubusercontent.com/hanmochen/Pictures/master/20210323211649.png" alt="image-20210323211649404" style="zoom:50%;" />



## Simulation-Based Search



**Forward Search**

- Select the best action by lookahead
- build a search tree with the current state $s_t$ at the root
- Use a model to look ahead
  - Not the complete MDP model
  - Just sub-MDP starting from now

<img src="https://raw.githubusercontent.com/hanmochen/Pictures/master/20210323212519.png" alt="image-20210323212519869" style="zoom:50%;" />



**Forward Search with simulated experiences**

- Simulate episodes of experience from $s_t$ with the model
- Apply model-free RL to simulated episodes
  - Monte-Carlo control: Monte-Carlo Search
  - Sarsa: TD search



### Monte-Carlo Search



**Simple Monte-Carlo Search**

- Given a model $\mathcal{M_V}$ and a simulation policy $π$

- For each state $a\in \mathcal{A}$

  - Simulate $K$ episodes from current state $s_t$

  $$
  \left\{s_{t}, a, R_{t+1}^{k}, S_{t+1}^{k}, A_{t+1}^{k}, \ldots, S_{T}^{k}\right\}_{k=1}^{K} \sim \mathcal{M}_{\nu}, \pi
  $$

  - Evaluate actions by mean return

  $$
  Q\left(s_{t}, a\right)=\frac{1}{K} \sum_{k=1}^{K} G_{t} \stackrel{P}{\rightarrow} q_{\pi}\left(s_{t}, a\right)
  $$

- Select current action with maximum value

$$
a_{t}=\underset{a \in \mathcal{A}}{\operatorname{argmax}} Q\left(s_{t}, a\right)
$$



**Monte-Carlo Tree Search**

- Given a model $\mathcal{M_V}$ 
- Simulate $K$ episodes from current state $s_t$ using current policy $π$

$$
\left\{s_{t}, a, R_{t+1}^{k}, S_{t+1}^{k}, A_{t+1}^{k}, \ldots, S_{T}^{k}\right\}_{k=1}^{K} \sim \mathcal{M}_{\nu}, \pi
$$

- Build a search tree containing visited states and actions

- Evaluate states $Q(s,a)$ by mean return of episodes from $(s,a)$
  $$
  Q(s, a)=\frac{1}{N(s, a)} \sum_{k=1}^{K} \sum_{u=t}^{T} \mathbf{1}\left(S_{u}, A_{u}=s, a\right) G_{u} \stackrel{P}{\rightarrow} q_{\pi}(s, a)
  $$

- In MCTS, the simulation policy $π$ improves

- Each simulation consists of two phases

  - Tree policy: pick actions to maximize $Q(s,a)$
  - Default policy: pick actions randomly

- Repeat:

  - Evaluare states $Q(s,a)$, by Monte-Carlo evaluation
  - Improve tree policy by ε-greedy

- Monte-Carlo control applied to simulated experience





Advantages of MCTS

- Highly selective best-first search
- Evaluate states dynamically
- Use sampling to break curse of dimensionality
- Works for black-box models
- Computationally efficient and parallelizable



### Temporal Difference Search



- Simulation Based Search
- Using TD instead of MC (uses bootstrapping)
  - MC tree search applies MC control to sub-MDP from now
  - TD search applies Sarsa to sub-MDP from now



**TD Search**

- Simulate episodes from current state $s_t$
- Estimate action-value function $Q(s,a)$
  - For each step of simulation, update action-values by Sarsa
  - $\Delta Q(S, A)=\alpha\left(R+\gamma Q\left(S^{\prime}, A^{\prime}\right)-Q(S, A)\right)$
- Select actions using $Q(s,a)$



**Dyna-2**

- In Dyna-2, the agent stores two sets of feature weights

  - Long-term memory: updated from real experience; general domain knowledge
  - Short-term memory: updated from simulated experience; specific local knowledge

  

# Lec 9 Exploration and Exploitation



## Introduction



**Exploration vs. Exploitation Dilemma**

- Online Decision-making involves a fundamental choice
  - Exploitation: make the best decision given current information
  - Exploration: gather more information

- Long-term strategy may sacrifice short-term interests



**Principles**

- **Naive Explorartion**

  - Add noise to greedy policy

- **Optimistic Initialization**

  - Assume the best until proven otherwise

- **Optimism in the Face of Uncertainty**

  - Prefer actions with uncertain values

- **Probability Matching**

  - Select actions according to probability they are best

- **Information State Search**

  



## Multi-Armed Bandit



A multi-armed bandit is a tuple $<\mathcal{A,R}>$

- $\mathcal{A}$ is a known set of $m$ actions (arms)
- $\mathcal{R}^a(r) = \mathbb{P}(r|a)$ is an unknown probability distribution over rewards
- At each time-step $t$
  - the agent selects an action $a_t\in A$
  - the environment generates a reward $r_t$
- Goal is to maximize cumulative reward $\sum\limits_{t=1}^T r_t$



### Regret



- Action-value function $Q(a)=\mathbb{E}[r \mid a]$
- optimal value $V^{*}=Q\left(a^{*}\right)=\max _{a \in \mathcal{A}} Q(a)$
- Regret: opportunity loss for one step $I_{t}=\mathbb{E}\left[V^{*}-Q\left(a_{t}\right)\right]$
- Total Regret:
  - $L_{t}=\mathbb{E}\left[\sum_{\tau=1}^{t} V^{*}-Q\left(a_{\tau}\right)\right]$
- Maximize cumulative reward = minimize total regret





**Counting Regret**



- Count: $N_t(a)$ expected number of selections for action $a$
- Gap: $Δ_a= V^* - Q(a)$

$$
\begin{aligned} L_{t} &=\mathbb{E}\left[\sum_{\tau=1}^{t} V^{*}-Q\left(a_{\tau}\right)\right] \\ &=\sum_{a \in \mathcal{A}} \mathbb{E}\left[N_{t}(a)\right]\left(V^{*}-Q(a)\right) \\ &=\sum_{a \in \mathcal{A}} \mathbb{E}\left[N_{t}(a)\right] \Delta_{a} \end{aligned}
$$



**Exploration-Exploitation trade-off** in the perspective of total regret

- If an algorithm forever explores, it will have linear total regret
- But if it never explores, still linear total regret
- Problem: find a balance between **Exploration-Exploitation** to achieve sublinear total regret



### Algorithms and total regret

- Greedy: linear total regret
- ε-greedy: linear total regret
- Optimistic Initialization
  - Initialize $Q(a)$ to high value
  - Encourage Systematic Exploration
  - Still lock onto suboptimal action
  - greedy/ε-greedy + optimistic initialization: linear total regret



- Decaying $ε_t$-greedy Algorithm

$$
\begin{array}{l}c>0 \\ d=\min _{a \mid \Delta_{a}>0} \Delta_{i} \\ \epsilon_{t}=\min \left\{1, \frac{c|\mathcal{A}|}{d^{2} t}\right\}\end{array}
$$





**Lower Bound**



- The performance is determined by similarity between optimal arm and others



**Theorem**

Asymptotic total regret is at least logarithmic in number of steps
$$
\lim _{t \rightarrow \infty} L_{t} \geq \log t \sum_{a \mid \Delta_{a}>0} \frac{\Delta_{a}}{K L\left(\mathcal{R}^{a}|| \mathcal{R}^{a^{*}}\right)}
$$


### Upper Confidence Bounds

**Optimism in the Face of Uncertainty**

- Estimate an upper confidence $\hat U_t(a)$ for each action-value
  - Small $N_t(a)$ leads to large  $\hat U_t(a)$
- Select action to max Upper Confidence Bound(UCB)

$$
a_{t}=\underset{a \in \mathcal{A}}{\operatorname{argmax}} \hat{Q}_{t}(a)+\hat{U}_{t}(a)
$$



**Determine the Upper Bound**

> Hoeffding's Inequality:
>
> Let $X_1,\cdots,X_n$ be i.i.d. random variables in $[0,1]$, and $\hat X_n = \frac 1 n \sum_{i=1}^n X_i$ is the sample mean.
>
> Then 
> $$
> \mathbb{P}[\mathbb{E}[X] \geqslant \hat X_n + u] \leqslant e^{-2nu^2}
> $$

Using that, we have 
$$
\mathbb{P}\left[Q(a)>\hat{Q}_{t}(a)+U_{t}(a)\right] \leq e^{-2 N_{t}(a) U_{t}(a)^{2}} = p
$$
So $U_{t}(a)=\sqrt{\frac{-\log p}{2 N_{t}(a)}}$ 

We can reduce $p$ as time step increases like $p=t^{-4}$

So $U_t(a) = \sqrt{\frac{2\log t }{N_t(a)}}$



**UCB1**
$$
a_{t}=\underset{a \in \mathcal{A}}{\operatorname{argmax}} Q(a)+\sqrt{\frac{2 \log t}{N_{t}(a)}}
$$


Theorem: The UCB1 algorithm achieves asymptotic logarithmic total regret. 
$$
\lim _{t \rightarrow \infty} L_{t} \leq 8 \log t \sum_{a \mid \Delta_{a}>0} \Delta_{a}
$$


### Bayesian Bandits



- exploits prior knowledge of rewards 
- compute posterior distribution of rewards 
- Use posterior to guide exploration
  - UCB
  - Probability matching
- Better performance with accurate prior 

**Bayesian UCB**

- Assume reward distribution is Gaussian $\mathcal{R}_a(r) = \mathcal{N}(r;μ_a,σ_a^2)$

$$
a_{t}=\underset{a \in \mathcal{A}}{\operatorname{argmax}} \mu_a+\frac{c\sigma_a }{\sqrt{{N_{t}(a)}}}
$$



**Probability Matching **
$$
\pi\left(a \mid h_{t}\right)=\mathbb{P}\left[Q(a)>Q\left(a^{\prime}\right), \forall a^{\prime} \neq a \mid h_{t}\right]
$$

- Select action according to probability that $a$ is optimal
  - can be optimistic to uncertainty
- But may be difficult to compute



**Thompson Sampling**

- Implements Probability Matching 

$$
\begin{aligned} \pi\left(a \mid h_{t}\right) &=\mathbb{P}\left[Q(a)>Q\left(a^{\prime}\right), \forall a^{\prime} \neq a \mid h_{t}\right.\\ &=\mathbb{E}_{\mathcal{R} \mid h_{t}}[\mathbf{1}(a=\underset{a \in \mathcal{A}}{\operatorname{argmax}} Q(a))] \end{aligned}
$$

- Use Bayes law to compute posterior distribution $p[\mathcal{R}|h]$
- Sample a reward distribution $\mathcal{R}$ from posterior and compute action-value function $Q(a) = \mathbb{E}[\mathcal{R}_a]$

- Select action maximizing value on sample



### Information State Search

**Value of Information**

- Quantify the value of information
- Help Trade-off between exploration and exploitation

**Information State Space**

- View bandits as sequential decision-making problems
- Define a information state $\tilde{s} = f(h_t)$
  - and induce a MDP
- Example: Bernoulli Bandits

**Solving information state space bandits**

- Model-free RL
  - e.g. Q-learning
- Bayesian-adaptive RL



**Bayes-Adaptive Bernoulli Bandits**

- Bernoulli Bandits:
  - $\mathcal{R}^a=\mathcal{B}(μ_a)$
  - Find which arm has the highest $μ_a$
- Information state $\hat s = <α,β>$
  - $(α_a,β_a)$ the number of success/fails in arm $a$
- Prior: beta distribution
- Compute posterior
  - transition matrix
- Can be solved by dynamic programming
  - Gittins index
- Or use simulation based search 



## Contextual Bandits



a tuple $<\mathcal{A,S,R}>$

- $\mathcal{A}$ is a known set of $m$ actions (arms)
- $\mathcal{S} = \mathbb{P}[s]$ is an unknown distribution over states
- $\mathcal{R}_s^a(r) = \mathbb{P}(r|s,a)$ is an unknown probability distribution over rewards
- At each time-step $t$
  - the environment generates state $s_t$
  - the agent selects an action $a_t\in A$
  - the environment generates a reward $r_t$
- Goal is to maximize cumulative reward $\sum\limits_{t=1}^T r_t$



Use Linear Regression to estimate $Q(s,a)$

- $Q_{\theta}(s, a)=\phi(s, a)^{\top} \theta \approx Q(s, a)$
- Using Least Squares Regression to estimate $Q(s,a)$ and its variance $σ^2(s,a)$ 

$$
\begin{aligned} A_{t} &=\sum_{\tau=1}^{t} \phi\left(s_{\tau}, a_{\tau}\right) \phi\left(s_{\tau}, a_{\tau}\right)^{\top} \\ b_{t} &=\sum_{\tau=1}^{t} \phi\left(s_{\tau}, a_{\tau}\right) r_{\tau} \\ \theta_{t} &=A_{t}^{-1} b_{t} 
\\ \sigma_{\theta}^{2}(s, a)&=\phi(s, a)^{\top} A^{-1} \phi(s, a)
\end{aligned}
$$



Using UCB
$$
a_{t}=\underset{a \in \mathcal{A}}{\operatorname{argmax}} Q_{\theta}\left(s_{t}, a\right)+c \sqrt{\phi\left(s_{t}, a\right)^{\top} A_{t}^{-1} \phi\left(s_{t}, a\right)}
$$


## MDPs



Same principles can be also applied to MDPs.



### Optimistic Initialization



**model-free RL**

- initialize action-value function $Q(s,a)$ to $\frac{r_{max}}{1-\gamma}$

- Run model-free RL algorithm
- Encourages systematic exploration of states and actions



**model-based RL**

- Construct an optimistic model of MDP
- Initialize transition to go to heaven
  - i.e.  transition to terminal state with $r_{max}$ reward
- Solve MDP by planning algorithm
- e.g. RMax algorithm





### Optimism in the Face of Uncertainty



**Model-free RL: UCB**
$$
a_{t}=\underset{a \in \mathcal{A}}{\operatorname{argmax}} Q\left(s_{t}, a\right)+U_{1}\left(s_{t}, a\right)+U_{2}\left(s_{t}, a\right)
$$

- $U_1(s_t,a)$: uncertainty in policy evaluation (easy to estimate)
- $U_2(s_t,a)$: uncertainty in policy improvement (hard)



**Model-based RL: bayesian**

- maintain posterior distribution 
- Use posterior to guide exploration
  - Bayesian UCB
  - Probability matching (Thompson Sampling)



**Thompson Sampling**
$$
\begin{aligned} \pi\left(s, a \mid h_{t}\right) &=\mathbb{P}\left[Q^{*}(s, a)>Q^{*}\left(s, a^{\prime}\right), \forall a^{\prime} \neq a \mid h_{t}\right] \\ &=\mathbb{E}_{\mathcal{P}, \mathcal{R} \mid h_{t}}\left[\mathbf{1}\left(a=\underset{a \in \mathcal{A}}{\operatorname{argmax}} Q^{*}(s, a)\right)\right] \end{aligned}
$$

- Use Bayes' Law to compute posterior
- Sample an MDP from posterior
- Solve sampled MDP using planning algorithm



### Information State Search 



- Augmented Information State Space
- $\tilde{\mathcal{M}}=\langle\tilde{\mathcal{S}}, \mathcal{A}, \tilde{\mathcal{P}}, \mathcal{R}, \gamma\rangle$
  - Bayesian-adaptive MDP
- Simulation-based search