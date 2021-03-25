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

	- asymptotic log total regret
	- require knowledge of gaps



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
> 

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