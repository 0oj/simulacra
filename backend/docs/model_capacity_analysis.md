# Model Capacity Analysis for Rocket Landing Policy

**Question**: What is the theoretical minimum neural network capacity required to represent
a near-optimal policy for the Simulacra rocket landing task?

**System summary**: 13-dim observation (position, velocity, quaternion, angular velocity) mapped
to 3-dim action (throttle, tilt_x, tilt_z) through a feedforward MLP, with an inner PD attitude
controller handling gimbal tracking. Current 64x64 MLP (~5k params) works at 5-60m but degrades
at 200m.

---

## 1. Structure of the Optimal Policy

### 1.1 The analytical guidance law

The landing problem decomposes into two nearly independent sub-problems:

**Vertical (throttle)**: When to start braking and how hard.
The fuel-optimal vertical policy is a *suicide burn* (aka hoverslam): coast under gravity
until the last possible moment, then fire at maximum thrust to reach v=0 at y=0. The
trigger condition is:

```
h_brake = v_y^2 / (2 * (T_max/m - g))
```

where `v_y` is downward speed, `T_max/m` is max specific thrust, and `g` is gravity.
The policy is essentially:

```
throttle = 1.0   if altitude <= h_brake + margin
throttle = 0.0   otherwise
```

In practice, a continuous throttle profile tracking a velocity reference is more robust.
The LandingController uses `v_desired = -k * sqrt(h)`, which gives:

```
throttle ~ m * (g + Kp * (v_desired - v_y)) / T_max
         = m * (g + Kp * (-k*sqrt(h) - v_y)) / T_max
```

This is a function of (h, v_y, m) involving: one `sqrt`, two multiplications, additions,
and a clamp to [0, 1]. Approximately **5-7 free parameters** (k, Kp, g, m, T_max, and the
terminal blending gains).

**Horizontal (tilt)**: Where to point the rocket to reach the pad.
The guidance law is proportional navigation: steer to null the displacement to the pad
at a rate proportional to the displacement:

```
v_x_desired = k_h * (target_x - x) * altitude_factor
tilt_x = atan2(a_x_desired, a_y) ~ a_x_desired / a_y   (small angle)
```

where `a_x_desired = Kp_h * (v_x_desired - v_x)` and `a_y ~ g + vertical_correction`.
Symmetric for z-axis.

This involves: subtraction, multiplication, division, and a clamp. Another
**5-7 free parameters** per axis.

**Combined**: The full analytical policy has approximately **15-20 tunable parameters** and
requires computing these elementary functions: `sqrt`, multiplication, division, `clamp`.
No trigonometric functions are needed (the inner PD loop handles attitude).

### 1.2 The ZEM/ZEV optimal guidance perspective

The energy-optimal (minimum integral of acceleration squared) powered descent guidance
has a closed-form solution called ZEM/ZEV (Zero-Effort-Miss / Zero-Effort-Velocity):

```
a_cmd(t) = C1 * ZEM / t_go^2 + C2 * ZEV / t_go
```

where:
- `ZEM = r_target - r - v*t_go - 0.5*g*t_go^2` (where you'd end up with no thrust)
- `ZEV = v_target - v - g*t_go` (what velocity you'd have with no thrust)
- `t_go` = time to go until landing
- `C1, C2` are constants (C1=6, C2=2 for minimum-energy with fixed t_go)

This is a **linear function of state** (position, velocity) with **nonlinear dependence on
t_go** (via 1/t_go and 1/t_go^2). The time-to-go itself is the solution to a quartic or
solved iteratively. Once t_go is known, the acceleration command is just a weighted sum of
state variables.

Key insight: the optimal policy is nearly **piecewise-linear in state**, with the
nonlinearity concentrated in the t_go computation and the throttle on/off decision.

---

## 2. Universal Approximation Theory

### 2.1 Barron (1993): Single hidden layer bounds

Barron showed that a single-hidden-layer sigmoidal network with n nodes achieves
integrated squared error O(C_f^2 / n), where C_f is the first moment of the Fourier
magnitude distribution. For smooth, bounded functions in d dimensions with n nodes,
the approximation error is:

```
E[||f - f_hat||^2] = O(C_f^2 / n) + O(nd / N * log N)
```

For our 13-dim input with smooth target functions, this suggests ~100-500 neurons in a
single hidden layer could achieve <1% relative error, **but this ignores the
compositional structure** of our target function.

### 2.2 Yarotsky (2017): Deep ReLU network error bounds

Yarotsky's results are more directly applicable. Key constructions:

| Operation | Depth | Width | Total neurons |
|-----------|-------|-------|---------------|
| x^2 (to error eps) | O(log(1/eps)) | O(1) | O(log(1/eps)) |
| x*y (to error eps) | O(log(1/eps)) | O(1) | O(log(1/eps)) |
| sqrt(x) (to error eps) | O(log(1/eps)^2) | O(1) | O(log(1/eps)^2) |
| 1/x on [a,1] (to error eps) | O(log(1/eps)^2) | O(1) | O(log(1/eps)^2) |
| f in W^{n,inf}([0,1]^d) | O(log(1/eps)) | O(eps^{-d/n}) | O(eps^{-d/n} * log(1/eps)) |

For our policy (composition of ~5 elementary operations each to 1% precision):

- Each `sqrt`: ~O(log(100)^2) ~ 50 neurons
- Each multiplication: ~O(log(100)) ~ 7 neurons
- Each division: ~O(log(100)^2) ~ 50 neurons
- Additions/subtractions: free (linear operations)
- Clamps: 2 ReLUs each

**Total for the analytical policy via composition: ~150-300 neurons across ~O(log(1/eps))
= 5-10 layers.**

### 2.3 Telgarsky (2016): Depth separation

Telgarsky proved that depth-k^3 networks with O(1) width can compute functions that
require exponential (2^k) width to approximate with depth O(k). The key mechanism:
deep networks can implement functions with exponentially many linear pieces.

For our policy, the number of "linear regions" is small (the policy is piecewise-smooth
with ~3-5 distinct regimes: coast, brake, terminal, lateral correction, landed). This
means we don't need extreme depth. **Two to three hidden layers should suffice** since the
number of regimes is bounded and each regime is smooth.

### 2.4 Lu et al. (2017): Width-bounded universal approximation

Lu et al. showed that width-(d+4) = 17 ReLU networks are universal approximators for
13-dim input. This means a network of width 17 with sufficient depth can represent any
continuous function on our observation space. However, the required depth could be very
large for complex functions.

---

## 3. Why 64x64 Fails at 200m but Works at 60m

### 3.1 Phase complexity analysis

The landing trajectory from different altitudes requires qualitatively different behavior:

**From 60m (simple):**
The entire trajectory is one smooth arc. The rocket barely has time to coast. The policy
is essentially: immediately begin braking while steering toward the pad. One smooth
function covers the entire state space encountered.

**From 200m (complex):**
The trajectory passes through distinct phases:

1. **Free-fall coast** (200m -> ~80m): throttle = 0, maintain attitude, let gravity
   accelerate. The policy must learn to *not* fire the engine despite being far from
   the ground. Duration: ~5 seconds.

2. **Lateral correction** (~150m -> 80m): with +-20m horizontal offset and significant
   horizontal velocity, the rocket must compute the right tilt to arrive above the pad.
   Requires sustained tilt commands that anticipate future positions.

3. **Brake initiation** (~80m): the critical suicide-burn trigger. Must compute
   h_brake = v^2 / (2*(T_max/m - g)) accurately. At 200m drop, v ~ 50-60 m/s, so
   h_brake ~ 80-100m. Getting this wrong by 10m means crash or hover waste.

4. **Active braking** (80m -> 5m): full or near-full throttle while simultaneously
   correcting lateral error. The throttle and tilt computations are coupled through mass
   (which is changing rapidly due to burn).

5. **Terminal descent** (5m -> 0m): very slow, gentle touchdown. Entirely different
   gain structure from the braking phase.

### 3.2 Capacity accounting

Each phase requires a different smooth function. A ReLU MLP implements a
piecewise-linear function. The number of linear regions of a network with L layers of
width w is at most:

```
R(L, w) <= prod_{i=1}^{L} floor(w_i choose d) ~ (w/d)^(d*L)  [simplified]
```

For a 64x64 network (2 hidden layers, 13 inputs):
- Layer 1: 64 ReLU units create at most 64 linear regions in 13-dim space
- Layer 2: 64 more ReLU units, each splitting existing regions
- Total linear regions: upper bound of ~64^2 = 4096

But the number of *useful* linear regions is much smaller because many neurons
are inactive over the training distribution. In practice, a 64x64 network implements
something like **200-500 effective linear pieces** over the training distribution.

For the 60m task, the policy lives in a small region of state space (h in [0,60], v in
[-15, 0], small x offsets) and needs ~50-100 linear pieces. **64x64 is sufficient.**

For the 200m task with +-20m horizontal offset, the state space volume is roughly:

```
Volume ratio ~ (200/60) * (60/15) * (40/10) ~ 50x larger
```

With 5 distinct phases instead of ~2, and each phase needing ~50-100 pieces, the total
requirement is **250-500 linear pieces across a 50x larger volume**. This pushes against
the practical capacity of a 64x64 network, especially since:

- Many linear regions are "wasted" on parts of state space rarely visited
- The transitions between phases (especially brake initiation) require sharp boundaries
  that consume linear regions
- The division by mass (which changes throughout the trajectory) requires additional
  nonlinear capacity

### 3.3 The brake-initiation boundary is the bottleneck

The most demanding part of the 200m policy is the **throttle onset decision**: a
hyperbolic boundary in (h, v_y, m) space defined by:

```
h = v_y^2 / (2 * (T_max/m - g))    [throttle goes from 0 to 1]
```

This is a **quadratic-rational surface** in state space. Approximating it with piecewise
linear regions to 1m accuracy over the range h in [50, 120], v in [-60, -20], m in
[80, 140] requires:

```
N_pieces ~ (70/1) * (40/1) * (60/5) ~ 33,600  (crude uniform grid)
```

Of course an efficient piecewise linear approximation needs far fewer pieces, but even
with adaptive meshing, ~200-500 pieces for the boundary alone is realistic. This single
feature may consume most of a 64x64 network's capacity.

---

## 4. Practical Capacity Estimation

### 4.1 Parameter counting for the target function

The analytical policy (Section 1) has this computational graph:

```
Inputs: [x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]

# Throttle path (the hard part):
h_brake = vy^2 / (2 * (T_max/m_est - g))        # 1 square, 1 division
v_desired = -k * sqrt(y)                          # 1 sqrt
throttle = clamp(m_est * (g + Kp*(v_desired - vy)) / T_max, 0, 1)
                                                   # 1 mult, 1 div, adds

# Tilt_x path:
dx = -x                                           # negation
vx_desired = k_h * dx * min(1, y/50)              # 1 mult, 1 min
ax_desired = Kp_h * (vx_desired - vx)             # linear
tilt_x = clamp(m_est * ax_desired / T_eff, -1, 1) # 1 mult, 1 div

# Tilt_z path: symmetric to tilt_x
```

This requires approximately:
- 2 divisions (hardest for ReLU: ~50 neurons each at 1% accuracy)
- 1 square root (~50 neurons at 1% accuracy)
- 1 squaring (~7 neurons)
- 4 multiplications (~7 neurons each)
- Several min/max/clamp operations (~2 neurons each)

**Conservative total: ~200 neurons, organized in 4-6 layers.**

In terms of parameters: a 200-neuron network organized as 13-64-48-32-32-3 has:
- 13*64 + 64 = 896
- 64*48 + 48 = 3120
- 48*32 + 32 = 1568
- 32*32 + 32 = 1056
- 32*3 + 3 = 99
- **Total: ~6,700 parameters**

### 4.2 Adding margin for learnability

The above is the *representation* minimum. RL training requires additional capacity for:

1. **Approximation margin**: the network must be learnable via gradient descent, not just
   representable. Over-parameterization helps optimization (the lottery ticket hypothesis
   suggests 2-10x overparameterization).

2. **Value function**: PPO trains a value function alongside the policy. The value function
   is arguably *harder* to represent than the policy because it must predict cumulative
   future reward, which depends on the entire trajectory. A shared backbone helps but the
   value head needs its own capacity.

3. **Robustness to distribution shift**: during training, the policy visits states it
   hasn't seen before. Some excess capacity helps generalize across the state space.

4. **Stochastic policy**: PPO outputs a Gaussian distribution (mean + log_std). The
   log_std needs its own representational capacity, though typically less than the mean.

Practical recommendation: **3-5x the theoretical minimum**.

### 4.3 Recommended architectures

| Architecture | Parameters | Expected capability |
|--------------|-----------|-------------------|
| 64 x 64 (current) | ~5,300 | 5-60m altitude, small offsets |
| 128 x 128 | ~19,700 | 5-120m altitude, +-10m offset |
| 128 x 128 x 64 | ~27,500 | 5-200m altitude, +-20m offset |
| 256 x 128 x 64 | ~40,500 | Full envelope with margin |
| 256 x 256 (RAJS paper) | ~70,700 | Full envelope, robust |

The jump from 64x64 to 128x128x64 is not just about width but about **adding a third
layer**. Three layers can implement compositions of nonlinear functions (like
sqrt(h) * m / T_max) much more efficiently than two layers.

### 4.4 Why depth matters more than width for this problem

The policy is a **composition of elementary functions**: `clamp(m * (g + Kp * (-k * sqrt(h) - vy)) / T_max)`.

A two-layer network must approximate this entire composition as a single piecewise-linear
map from input to output. A three-layer network can use the first layer to compute
intermediate features (like `sqrt(h)`, `v_y^2`, `m * a_x`), the second layer to combine
them (like `h_brake`, `v_desired`), and the third to produce outputs.

This matches Telgarsky's depth separation result: the compositional structure of the
target function means depth provides exponential efficiency gains.

---

## 5. Literature Context

### 5.1 Powered descent guidance

The state of the art in analytical PDG is the G-FOLD algorithm (Blackmore et al., 2010),
used operationally on SpaceX Falcon 9 landings since ~2015. G-FOLD solves a convex
optimization (SOCP) online in ~100ms. The key insight is that the fuel-optimal thrust
profile is bang-bang (max thrust or zero), with at most two switches. The optimal
guidance is a linear function of state given the switching times.

For our simpler problem (no aerodynamics, no glideslope constraint, fixed landing site),
the optimal policy reduces to computing brake initiation time, which is essentially a
threshold in (h, v, m) space.

### 5.2 RL for rocket landing

The RAJS paper (LandSpace, 2024) used 256x256 networks with PPO and achieved 97% success
rate on a high-fidelity rocket model. They found that bootstrapping from a classical
controller was essential for training stability.

Other RL approaches for rocket landing typically use 64-256 neurons per layer with 2-3
hidden layers, consistent with our analysis that the policy requires O(200) neurons of
representational capacity.

### 5.3 Neural network approximation of control policies

Yarotsky (2017) provides the tightest known bounds for deep ReLU approximation of smooth
functions composed of elementary operations. For functions in the Sobolev space
W^{n,inf}([0,1]^d), a deep ReLU network needs O(eps^{-d/n} * log(1/eps)) neurons for
eps-approximation. With d=13 and n=2 (our policy is at least C^2 smooth within each
phase), this gives O(eps^{-6.5} * log(1/eps)) for global approximation -- clearly
dominated by the curse of dimensionality.

However, our policy has **low intrinsic dimensionality**: the throttle depends on ~3
effective variables (h, v_y, m), and each tilt depends on ~4 variables (x_i, v_i, h, m).
The effective dimension is ~4-5, not 13. This gives:

```
N_neurons ~ O(eps^{-5/2} * log(1/eps)) ~ O(eps^{-2.5} * log(1/eps))
```

At eps = 0.01 (1% accuracy): ~100,000^1.25 * 7 -- which is clearly too pessimistic.
These are worst-case bounds. The actual policy is far more structured than a generic
Sobolev function, being a composition of ~5 elementary operations on low-dimensional
projections of the input.

Using the compositional bounds: each elementary operation needs O(log(1/eps)^2) ~ 50
neurons, with ~10 operations composed sequentially. Total: **~500 neurons in ~5-8 layers**
for 1% accuracy.

### 5.4 The effective dimensionality argument

Our 13-dim observation has significant redundancy:
- (qw, qx, qy, qz) is 4 numbers encoding 3 DOF (unit quaternion constraint), and for
  the RL policy output (tilt_x, tilt_z), only the deviation from vertical matters --
  effectively 2 DOF.
- The policy barely uses (wx, wy, wz) since the inner PD loop handles angular rate
  damping. These are "nice to have" for anticipating actuator lag but not critical.

Effective input dimension: **~7-8** (x, y, z, vx, vy, vz, tilt_from_vertical, mass_proxy).

This further reduces the required capacity since approximation bounds scale exponentially
with input dimension.

---

## 6. Summary and Recommendations

### 6.1 Theoretical minimum

The analytical landing policy requires computing ~10 elementary operations (sqrt, multiply,
divide, clamp) on ~7 effective input dimensions. Based on Yarotsky's constructive bounds:

- **Representational minimum**: ~200-300 neurons in 4-6 layers (~3,000-7,000 parameters)
- **Learnable minimum** (with RL optimization margin): ~600-1,000 neurons in 3-4 layers
  (~10,000-25,000 parameters)

### 6.2 Why 64x64 breaks at 200m

Three compounding factors:

1. **Insufficient linear regions**: ~4,000 theoretical max, ~300 effective. The brake
   initiation boundary alone needs ~200-500 pieces over the 200m operating envelope.

2. **Missing depth**: two layers cannot efficiently compose sqrt * divide * multiply. The
   network must approximate the *entire composition* as a flat piecewise-linear map,
   wasting capacity.

3. **State space volume**: 50x larger state space than the 60m task means each linear
   region must cover more volume, reducing approximation quality at phase boundaries.

### 6.3 Concrete recommendation

For the full operating envelope (5-200m altitude, +-20m horizontal offset):

```python
policy_kwargs = dict(
    net_arch=dict(
        pi=[128, 128, 64],   # 3 layers for compositional structure
        vf=[256, 128, 64],   # Value function needs MORE capacity (predicts cumulative reward)
    ),
    activation_fn=nn.Tanh,   # Consider Tanh over ReLU: smoother policy, better for
                              # approximating sqrt/division natively
)
```

**Estimated parameters**: ~25,000 (policy) + ~40,000 (value) = ~65,000 total.

This is ~5x the current 64x64 network, which aligns with the ~50x increase in state
space volume and ~2.5x increase in phase complexity, modulated by the efficiency gain
from adding a third layer.

### 6.4 Alternative: use problem structure directly

Rather than increasing raw capacity, consider encoding known structure:

1. **Feature engineering**: add `sqrt(y)`, `vy^2`, and `vy^2 / y` as input features.
   This hands the network the intermediate computations it struggles to learn, potentially
   letting 64x64 work at 200m.

2. **Residual on analytical policy**: train the RL to output corrections to the
   LandingController. The network only needs to learn the *difference* from the analytical
   solution, which is a much simpler function (mostly near-zero with small corrections).
   A 32x32 network might suffice for residuals.

3. **Phase-conditioned policy**: explicitly detect the flight phase (coast/brake/terminal)
   and use a separate small network for each. Each sub-network operates in a smaller state
   space with a simpler target function.

---

## Sources

### Approximation theory
- [Barron 1993 - Universal Approximation Bounds](https://pages.cs.wisc.edu/~brecht/cs838docs/93.Barron.Universal.pdf)
- [Yarotsky 2017 - Error Bounds for Deep ReLU Networks](https://arxiv.org/abs/1610.01145)
- [Telgarsky 2016 - Benefits of Depth in Neural Networks](http://proceedings.mlr.press/v49/telgarsky16.pdf)
- [Lu et al. 2017 - Expressive Power of Neural Networks: A View from the Width](https://arxiv.org/abs/1709.02540)
- [Deep Network Approximation for Smooth Functions (SIAM)](https://epubs.siam.org/doi/10.1137/20M134695X)
- [Optimal Approximation of Piecewise Smooth Functions Using Deep ReLU Neural Networks](https://www.sciencedirect.com/science/article/abs/pii/S0893608018302454)

### Powered descent guidance
- [G-FOLD: Real-Time Fuel Optimal Guidance (Blackmore et al.)](https://www.researchgate.net/publication/258676350_G-FOLD_A_Real-Time_Implementable_Fuel_Optimal_Large_Divert_Guidance_Algorithm_for_Planetary_Pinpoint_Landing)
- [Lossless Convexification (Blackmore)](http://larsblackmore.com/losslessconvexification.htm)
- [Rethinking Propellant-Optimal PDG (AIAA)](https://arc.aiaa.org/doi/10.2514/1.G008343)
- [Terminal Soft Landing via Analytic Gravity Turn](https://arxiv.org/html/2409.01465)
- [ZEM/ZEV Feedback Guidance (Adaptive, RL-based)](https://www.sciencedirect.com/science/article/pii/S0094576520301247)

### RL for rocket landing
- [RAJS: Rocket Landing Control with Jump Start RL (LandSpace, 2024)](https://arxiv.org/abs/2407.15083)
- [RL Approach for Adaptive Tracking Control of Reusable Rocket](https://www.sciencedirect.com/science/article/abs/pii/S0925231224001486)
- [Propulsive Landing of Launchers with Deep RL](https://www.sciencedirect.com/science/article/abs/pii/S0094576524006751)

### Neural network complexity
- [VC Dimension of Neural Networks (Sontag)](http://www.sontaglab.org/FTPDIR/vc-expo.pdf)
- [Nearly-tight VC-dimension Bounds for Piecewise Linear Networks (JMLR)](https://jmlr.org/papers/volume20/17-612/17-612.pdf)
- [COLT 2021: How Many Neurons to Approximate Smooth Functions](http://blog.claytonsanford.com/2021/08/15/hssv21.html)
