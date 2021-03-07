# Lec 5 Model-free Control

[toc]

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

  

**Importance Sampling for off-policy Monte-Carlo**

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

