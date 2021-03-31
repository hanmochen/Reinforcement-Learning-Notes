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

  

