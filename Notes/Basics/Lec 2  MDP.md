# Lec 2  MDP

[toc]

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

