# Lec 4 MC-TD

[TOC]

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



<img src="../../../../../Application Support/typora-user-images/image-20210304202528854.png" alt="image-20210304202528854" style="zoom:50%;" />



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

