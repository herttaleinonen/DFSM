# The Dynamic Foveated Search Model 

<a href="url"><img src="https://github.com/herttaleinonen/DFSM/blob/main/replay_model/images/dfsm.png" height="548" width="548" ></a>

## The model scheme:

Each search trial was approximated as a sequence of discrete time steps (Δ𝑡 = 50 ms). For each object
𝑖 at time step 𝑡, retinal eccentricity was computed as: 

$$
e_{i,t} = \lVert x_{i,t} - g_t \rVert,
$$

where $x_{i,t}$ and $g_t$ are the object and gaze positions in 2D coordinates. Orientation sensitivity was based on the visibility task data:

$$
S(e,v)=\Phi^{-1}(H(e,v))-\Phi^{-1}(F(e,v)),
$$

where 𝐻(𝑒, 𝑣) and 𝐹(𝑒, 𝑣) are hit and false-alarm rates, 𝑒 standing for retinal eccentricity and
𝑣 for object velocity, and $\Phi^{-1}$ is the inverse cumulative normal distribution. The resulting
values were interpolated across eccentricity to obtain a continuous sensitivity function
𝑆(𝑒, 𝑣) for each velocity condition. At each time step 𝑡, each object 𝑖 produces a noisy internal sensory sample $z_{i,t}$ which was
modelled as a Gaussian random variable:

$$
z_{i,t}\sim
\begin{cases}
\mathcal{N}(0,\eta), & \text{if the object is a distractor},\\
\mathcal{N}(S_{\Delta t}(e_{i,t},v),\eta), & \text{if the object is the target},
\end{cases}
$$

where 𝜂 is the sensory noise variance. Thus, distractors generate noise-only activity, whereas
targets generate signal-plus-noise activity. Per-step sensitivity was derived from the calibrated
𝑆(𝑒, 𝑣), measured over a 400 ms observation window (duration of the visibility task), by
assuming that discriminability grows with the square root of integration time:

$$
S_{\Delta t}(e_{i,t},v)=
 S(e_{i,t},v)\sqrt{\frac{\Delta t}{0.4}},
$$

Momentary sensory samples were converted into object-wise log-likelihood ratio (LLR) increments:

$$
\Delta\mathrm{LLR}_{i,t}=
\log
\frac{p(z_{i,t}\mid\text{target})}
{p(z_{i,t}\mid\text{distractor})},
$$

where 𝑝(𝑧 ∣ ⋅) represents the probability density of the sensory observation under the specified
hypothesis. Under Gaussian sensory noise with variance 𝜂, this simplifies to

$$
\Delta\mathrm{LLR}_{i,t}={S_{\Delta t}(e_{i,t},v)}{z_{i,t}}-\frac{1}{2}S_{\Delta t}(e_{i,t},v)^2.
$$

Object-specific evidence was accumulated over time:

$$
\mathrm{LLR}_{i,t}=\mathrm{LLR}_{i,t-1}+\Delta\mathrm{LLR}_{i,t}=\sum_{\tau=1}^{t}\Delta\mathrm{LLR}_{i,\tau}.
$$

To infer whether a target was present anywhere in the display, the model used the maximum
object-wise accumulated evidence as the decision variable:

$$
D_t=\max_i \mathrm{LLR}_{i,t}.
$$

At each time step 𝑡, the decision variable $D_t$ was compared against two fixed decision bounds.
A positive upper bound $\Theta_{p}$ governed target-present responses: if $D_t \ge \Theta_{p}$ the model
responded, “target present”, and reaction time was recorded as (𝑡 + 1)Δ𝑡. Target-absent
responses were governed by a fixed negative lower bound; if $D_t \le \Theta_{A}$, the model responded, 
“target absent”, and reaction time was recorded as (𝑡 + 1)Δ𝑡. If no bound was
crossed by stimulus offset (3.5 s), the model responded “target absent”.

For each participant, the parameters 𝜂 (sensory noise) and Θ (decision criterion) were
estimated via grid search by minimizing a composite loss function:

$$
L=\sum_v\left[(d'_h-d'_m)^2+(\log RT_h^{TP}-\log RT_m^{TP})^2+(\log RT_h^{TA}-\log RT_m^{TA})^2\right],
$$

where the subscripts *h* and *m* denote human and model respectively, and the sum runs over
velocity conditions 𝑣. Target-present *TP* , and target-absent *TA* reaction times are compared
on a log scale to treat proportional differences equivalently across conditions. Trial-wise
stimulus and gaze time series were precomputed once. Parameter estimation proceeded in
two phases: both *η* and *Θ* were searched over 12 points,
with a refined search of ±0.15 and ±1.0 respectively around the coarse optimum. 


# Requirements
Python 3.10. <br/>

# Installation
Clone the `replay_model` -folder to your machine.

# Running 

To run the model, run 
```
python3 main.py
```
By default, this command runs parameter fitting, model simulations and saccade predictions.

To recreate model vs. human figures and analyses, run
```
python3 figures.py
```

