# Literature Review: Feature Engineering and Observation Design in RL for Control

## Context

This review supports a final-year project on RL-based rocket landing control.
The project trains an RL policy (13-dim raw obs + 4 physics-informed features),
then distills it to symbolic form using SINDy-RL for verification and deployment trust.

The central research questions are:
1. What observation/feature engineering techniques improve RL for control tasks?
2. How does physics-informed observation design affect sample efficiency and generalizability?
3. Can we show that a larger network internally learns the same features we hand-craft?
4. How do we distill a trained policy into an interpretable, verifiable form?

---

## 1. Feature Engineering and Observation Design in RL for Control

### 1.1 The Hamilton-Dunlap-Hobbs Study on Observation Space Design

**Hamilton, N., Dunlap, K., & Hobbs, K.L. (2025).** Investigating the Impact of
Observation Space Design Choices on Training Reinforcement Learning Solutions for
Spacecraft Problems. *arXiv:2501.06016*.

- **Key findings**: Studies split into two groups: (a) impact of engineered "sensors"
  (derived features designed to help agents learn), and (b) impact of reference frames.
  Sensors are not strictly necessary, but most help agents learn more optimal behavior.
  Reference frame choice has limited impact but should be kept consistent.
- **Relevance**: Directly validates our approach of adding physics-informed features
  (sqrt(y), vy^2/2y, etc.) alongside raw state. Shows that engineered features improve
  optimality even when the raw state is sufficient for learning.
- **Access**: arXiv (open access). https://arxiv.org/abs/2501.06016

### 1.2 Survey on Physics-Informed Reinforcement Learning

**Banerjee, C., Nguyen, K., Fookes, C., & Raissi, M. (2023).** A Survey on Physics
Informed Reinforcement Learning: Review and Open Problems. *arXiv:2309.01909*.
Published in *Expert Systems with Applications* (2025).

- **Key findings**: Introduces a taxonomy of physics incorporation methods along the
  RL pipeline: observational bias (encoding physics in state representation),
  inductive bias (architecture design), and learning bias (loss function / reward
  shaping). Identifies that observational bias -- using multi-modal data reflecting
  physical principles -- is one of three primary ways to inject physics into RL.
- **Relevance**: Our sqrt(y) and vy^2/(2y) features are classic "observational bias":
  encoding known stopping-distance physics into the observation. The taxonomy helps
  frame our contribution.
- **Access**: arXiv (open access). https://arxiv.org/abs/2309.01909

### 1.3 State Representation Learning for Deep RL

**Lesort, T., Diaz-Rodriguez, N., Goudou, J.F., & Filliat, D. (2018).** State
Representation Learning: A Review. *Journal of Artificial Intelligence Research*,
63, 541-582.

- **Key findings**: Decoupling representation learning from policy learning treats
  them as two distinct problems. Mapping functionally equivalent states to similar
  points in latent space via metric-based methods enhances sample efficiency.
  Incorporating domain knowledge into state design often makes the difference between
  success and failure.
- **Relevance**: Motivates our feature engineering as a form of manual representation
  learning -- selecting the right basis functions to make the control-relevant
  structure explicit.
- **Access**: Open access via JAIR.

### 1.4 Structured Control Nets

**Srouji, M., Zhang, J., & Salakhutdinov, R. (2018).** Structured Control Nets for
Deep Reinforcement Learning. *Proceedings of ICML 2018*, PMLR 80:4742-4751.

- **Key findings**: Splits the MLP policy into a linear control module (for local
  stabilization) and a nonlinear control module (for global/forward-looking control).
  The two streams combine additively. Inspired by classical nonlinear control theory.
  Achieves better training efficiency, rewards, and generalization with smaller networks.
- **Relevance**: Directly relevant to our hypothesis that physics-informed features
  make the linear component of control explicit, allowing a smaller network to succeed.
  Our hand-crafted features (stopping distance, tilt) can be seen as making the linear
  control structure explicit.
- **Access**: arXiv (open access). https://arxiv.org/abs/1802.08311

### 1.5 Feature Selection for RL

**Li, L., Lozano, A.C., Ravi, S.N., & Luo, Y. (2015).** Feature Selection and
Feature Learning for High-Dimensional Batch Reinforcement Learning: A Survey.
*Machine Intelligence Research*, 12, 893-910.

- **Key findings**: For real-world RL with high-dimensional state spaces, it is
  difficult to use handcrafted features for value function approximation. However,
  data-driven feature selection can be incorporated into batch RL. Polynomial basis,
  Fourier basis, and RBF features are standard representations for continuous control.
- **Relevance**: Provides theoretical grounding for our choice of basis functions
  (polynomial-like features sqrt(y), quadratic vy^2/2y).
- **Access**: Springer (may need institutional access).
  DOI: 10.1007/s11633-015-0893-y

### 1.6 Potential-Based Reward Shaping

**Ng, A.Y., Harada, D., & Russell, S.J. (1999).** Policy Invariance Under Reward
Transformations: Theory and Application to Reward Shaping. *Proceedings of ICML
1999*.

- **Key findings**: If the shaping reward takes the form of a potential difference
  Phi(s') - Phi(s), the optimal policy is preserved (policy invariance). This allows
  injecting domain knowledge via reward without distorting the optimal solution.
- **Relevance**: While we inject physics through observations rather than rewards,
  this foundational work establishes the principle that domain knowledge can
  dramatically accelerate learning. Our energy-based features (vy^2/2y) could
  alternatively be used as potential functions for reward shaping.
- **Access**: Open access via various sources.

---

## 2. Rocket and Aerospace RL

### 2.1 6-DOF Planetary Powered Descent (Gaudet et al.)

**Gaudet, B., Linares, R., & Furfaro, R. (2020).** Deep Reinforcement Learning for
Six Degree-of-Freedom Planetary Landing. *Advances in Space Research*, 65(7),
1723-1741.

- **Key findings**: Learns a policy mapping estimated lander state directly to
  commanded thrust for each engine. Uses PPO. Achieves fuel-efficient trajectories
  over realistic deployment ellipses. The observation includes full 6-DOF state
  (position, velocity, orientation quaternion, angular velocity).
- **Relevance**: Establishes the 13-dim observation pattern (pos + vel + quat +
  omega) that we also use. Does NOT use engineered features -- raw state only.
  Our work extends this by showing engineered features can reduce required network
  capacity.
- **Access**: arXiv preprint available. https://arxiv.org/abs/1810.08719
  Journal: DOI: 10.1016/j.asr.2019.12.030

### 2.2 Adaptive ZEM/ZEV Guidance via Meta-RL

**Furfaro, R., Scorsoglio, A., Linares, R., & Massari, M. (2020).** Adaptive
Generalized ZEM-ZEV Feedback Guidance for Planetary Landing via a Deep
Reinforcement Learning Approach. *Acta Astronautica*, 171, 156-171.

- **Key findings**: Uses RL to learn the gains of a classical ZEM/ZEV guidance law
  as functions of spacecraft state. The observation is notably minimal: velocity
  error, estimated altitude, and time-to-go. Position coordinates are deliberately
  excluded (except altitude), yielding a policy that generalizes well to unseen
  regions of state space.
- **Relevance**: Demonstrates that physics-informed observation design (using
  domain-specific quantities like time-to-go and velocity error rather than raw
  state) yields superior generalization. This is the strongest precedent for our
  approach of replacing raw y with sqrt(y) and vy^2/(2y).
- **Access**: arXiv (open access). https://arxiv.org/abs/2003.02182
  Journal: DOI: 10.1016/j.actaastro.2020.02.051

### 2.3 Adaptive Guidance with Meta-RL and Integrated Navigation

**Gaudet, B., Furfaro, R., & Linares, R. (2020).** Adaptive Guidance and Integrated
Navigation with Reinforcement Meta-Learning. *Acta Astronautica*, 169, 180-190.

- **Key findings**: The observation given to the agent consists of velocity error,
  altitude, and time-to-go -- NOT raw position/velocity. This design choice is
  intentional: excluding translational coordinates produces a policy with good
  generalization. The agent can even learn guidance from raw Doppler radar altimeter
  readings, integrating guidance and navigation.
- **Relevance**: Strong evidence that physics-informed feature selection (time-to-go,
  velocity error) outperforms raw state for generalization in aerospace RL.
- **Access**: ScienceDirect. DOI: 10.1016/j.actaastro.2019.12.028

### 2.4 RAJS: Rocket Landing with Jump Start RL

**Jiang, W., et al. (2024).** Rocket Landing Control with Random Annealing Jump
Start Reinforcement Learning. *arXiv:2407.15083*.

- **Key findings**: Uses a prior feedback controller as a guide policy to facilitate
  exploration. Elevates success rate from 8% (baseline controller) to 97% on a
  high-fidelity LandSpace Simulink model. Addresses the goal-oriented sparse reward
  problem that makes rocket landing hard for standard RL.
- **Relevance**: The high-fidelity plant model and jump-start approach are relevant
  to our training methodology. Their observation design uses standard state vectors.
- **Access**: arXiv (open access). https://arxiv.org/abs/2407.15083

### 2.5 Rocket Engine Control with DRL (Dresia)

**Dresia, K. (2025).** Rocket Engine Control with Deep Reinforcement Learning.
*Doctoral dissertation*, RWTH Aachen University. DLR Research Report DLR-FB-2025-16.

- **Key findings**: Demonstrates DRL for rocket engine control on a real LOX/LNG
  engine (LUMEN) at DLR test bench P8.3. Achieves 1.3% control accuracy. Uses
  domain randomization for robustness. Discusses safety/reliability requirements
  for flight engine control.
- **Relevance**: One of the few works demonstrating RL-based rocket control on
  real hardware. Raises the safety/verification question that motivates our SINDy-RL
  distillation approach.
- **Access**: Open access via DLR elib. https://elib.dlr.de/219040/

### 2.6 Propulsive Landing with DRL (Atmospheric 6-DOF)

**Sena, C., et al. (2024).** Propulsive Landing of Launchers' First Stages with
Deep Reinforcement Learning. *Acta Astronautica*, 2024.

- **Key findings**: Examines atmospheric planetary landing of rocket first stages
  using RL (an understudied area vs. vacuum landing). Uses PPO with a 6-DOF
  simulation including aerodynamics. Provides an open-source simulation environment.
- **Relevance**: Closest to our problem setup (atmospheric, 6-DOF, first-stage
  landing). Uses standard state observation without engineered features.
- **Access**: ScienceDirect. DOI: 10.1016/j.actaastro.2024.12.013

### 2.7 Meta-RL with Transformers for Rocket Landing

**Carradori, J., Mooij, E., & Sagliano, M. (2025).** Transformer-Based Robust
Feedback Guidance for Atmospheric Powered Landing. *Proceedings of AIAA SciTech
2025*, AIAA 2025-2771.

- **Key findings**: Integrates Meta-RL with Gated Transformer XL networks for
  6-DOF atmospheric landing. Demonstrates robustness to aerodynamic uncertainty,
  navigation errors, and dispersed initial conditions via Monte Carlo simulations.
  Meta-RL optimizes a meta-policy for the entire task distribution, not a single task.
- **Relevance**: Represents the state-of-the-art in RL for atmospheric rocket
  landing. Their transformer architecture contrasts with our MLP approach.
- **Access**: AIAA (institutional access). DOI: 10.2514/6.2025-2771
  Also available as TU Delft thesis: https://repository.tudelft.nl/record/uuid:bf2a598c-9694-40cd-8dec-03b73d539b54

### 2.8 Curriculum RL for Rocket Landing

**Ferrante, R., et al. (2022).** Leveraging Curriculum Reinforcement Learning for
Rocket Powered Landing Guidance and Control. *IEEE Aerospace Conference 2022*.

- **Key findings**: Proposes curriculum RL that starts with easier tasks and
  increases difficulty by adding goals (attitude stability, terminal velocity,
  position, fuel). Each task provides a good initial policy for the next.
  Addresses poor sample efficiency from sparse, multi-objective rewards.
- **Relevance**: Curriculum learning is complementary to our feature engineering
  approach -- both aim to make the learning problem easier.
- **Access**: IEEE Xplore (institutional access). DOI: 10.1109/AERO53065.2022.9728176

### 2.9 Classical Convex Optimization Baseline (G-FOLD)

**Acikmese, B. & Ploen, S.R. (2007).** Convex Programming Approach to Powered
Descent Guidance for Mars Landing. *Journal of Guidance, Control, and Dynamics*,
30(5), 1353-1366.

- **Key findings**: Formulates minimum-fuel powered descent as a second-order cone
  program (SOCP) via lossless convexification. Solves in 100ms on 1.4GHz processor.
  Flight-tested with NASA JPL in 2013.
- **Relevance**: The classical baseline against which RL approaches must be compared.
  G-FOLD is optimal for 3-DOF but does not easily handle 6-DOF with complex
  constraints. RL's advantage is handling nonlinear dynamics and constraints that
  break convexity.
- **Access**: AIAA (institutional access). DOI: 10.2514/1.27553

---

## 3. Interpretability and Distillation of RL Policies

### 3.1 SINDy-RL

**Zolman, N., Brunton, S.L., & Kutz, J.N. (2025).** SINDy-RL: Interpretable and
Efficient Model-Based Reinforcement Learning. *Nature Communications*, 16, 10714.

- **Key findings**: Unifying framework combining SINDy with DRL. Creates
  interpretable representations of dynamics, reward, and policy. Uses E-SINDy
  (ensemble SINDy) dynamics models and behavior cloning with sparse dictionary
  learning for policy distillation. Achieves comparable performance to DRL with
  far fewer environment interactions. Resulting policy is orders of magnitude
  smaller than a neural network.
- **Relevance**: This is the primary method we plan to use for distillation.
  Our physics-informed features (sqrt(y), vy^2/(2y)) can serve as candidate
  library functions in the SINDy dictionary, making distillation more likely
  to succeed.
- **Access**: Open access (Nature Communications).
  https://www.nature.com/articles/s41467-025-65738-4
  Code: https://github.com/nzolman/sindy-rl

### 3.2 Original SINDy Method

**Brunton, S.L., Proctor, J.L., & Kutz, J.N. (2016).** Discovering Governing
Equations from Data by Sparse Identification of Nonlinear Dynamical Systems.
*Proceedings of the National Academy of Sciences*, 113(15), 3932-3937.

- **Key findings**: Uses sparse regression on a library of candidate nonlinear
  functions to identify the fewest terms governing dynamics from data. Only assumes
  sparsity in the function space. Recovers known physical laws (Lorenz, fluid
  dynamics) from data.
- **Relevance**: Foundation for SINDy-RL. The choice of candidate library functions
  is critical -- our physics-informed features are exactly the kind of terms that
  should appear in the library.
- **Access**: Open access (PNAS). DOI: 10.1073/pnas.1517384113

### 3.3 Symbolic Policy Distillation (SPID)

**Anonymous (under review).** Symbolic Policy Distillation for Interpretable
Reinforcement Learning. *OpenReview*.

- **Key findings**: Distills DNN policies into symbolic analytical equations via
  symbolic regression. Achieves superior preservation of both performance and
  fidelity. Provides mechanistic insights into policy behavior and training dynamics.
- **Relevance**: An alternative to SINDy-RL for our distillation step. The key
  difference is SPID distills the policy directly, while SINDy-RL distills both
  the model and the policy.
- **Access**: OpenReview (open access). https://openreview.net/forum?id=xLycLWUMas

### 3.4 Distillation via GBMs and Symbolic Regression

**Dhatt-Gauthier, M., et al. (2024).** Distilling Reinforcement Learning Policies
for Interpretable Robot Locomotion: Gradient Boosting Machines and Symbolic
Regression. *arXiv:2403.14328*.

- **Key findings**: Distills neural RL policies into GBMs, EBMs, and symbolic
  expressions for robot locomotion. Uses DAgger with a curriculum of episode-dependent
  alternation between expert and distilled policies to address distribution shift.
- **Relevance**: The DAgger-based distillation approach could complement our SINDy-RL
  approach. Demonstrates that RL policies for continuous control CAN be distilled
  to interpretable forms.
- **Access**: arXiv (open access). https://arxiv.org/abs/2403.14328

### 3.5 Discovering Symbolic Policies (Deep Symbolic Optimization)

**Landajuela, M., Petersen, B.K., Kim, S., et al. (2021).** Discovering Symbolic
Policies with Deep Reinforcement Learning. *Proceedings of ICML 2021*, PMLR
139:5979-5989.

- **Key findings**: Uses an autoregressive RNN to generate symbolic control policies
  (mathematical expressions). Risk-seeking policy gradient for optimization.
  "Anchoring" algorithm distills pre-trained neural policies into fully symbolic
  policies one action dimension at a time. Outperforms 7 DRL algorithms across
  8 benchmark environments.
- **Relevance**: An end-to-end alternative where the policy IS symbolic from the
  start, rather than train-then-distill. Could serve as a comparison method.
- **Access**: Open access via PMLR.
  https://proceedings.mlr.press/v139/landajuela21a.html

### 3.6 Discovering Symbolic Models from Deep Learning

**Cranmer, M., Sanchez-Gonzalez, A., Battaglia, P., et al. (2020).** Discovering
Symbolic Models from Deep Learning with Inductive Biases. *Advances in NeurIPS 33*.

- **Key findings**: Trains a GNN with inductive biases (sparsity in latent space),
  then applies symbolic regression to extract explicit physical relations. Recovers
  known force laws and Hamiltonians. The symbolic expressions generalize BETTER
  to out-of-distribution data than the neural network itself.
- **Relevance**: Strong evidence that distilling neural networks to symbolic form
  is not just about interpretability -- it can actually improve generalization.
  Directly supports our thesis that the bigger network learns features that can
  be extracted and expressed symbolically.
- **Access**: arXiv (open access). https://arxiv.org/abs/2006.11287

### 3.7 Verifiable RL via Policy Extraction (VIPER)

**Bastani, O., Pu, Y., & Solar-Lezama, A. (2018).** Verifiable Reinforcement
Learning via Policy Extraction. *Advances in NeurIPS 31*.

- **Key findings**: VIPER learns high-fidelity decision tree policies from DNN
  policies via imitation learning (DAgger). The resulting decision tree can be
  formally verified. Demonstrates on Atari games and a collision avoidance system.
- **Relevance**: Establishes the verify-after-distillation pipeline that our
  project follows (train NN -> distill to interpretable form -> verify).
- **Access**: Open access. https://people.csail.mit.edu/asolar/papers/BastaniPS18.pdf

### 3.8 Programmatically Interpretable RL (PIRL)

**Verma, A., Murali, V., Singh, R., Kohli, P., & Chaudhuri, S. (2018).**
Programmatically Interpretable Reinforcement Learning. *Proceedings of ICML 2018*,
PMLR 80:5045-5054.

- **Key findings**: Transforms neural network policies into high-level programming
  language expressions. Uses the neural policy to direct local search over
  programmatic policies, minimizing divergence.
- **Relevance**: Another distillation approach. The "programmatic" output is more
  structured than pure symbolic regression but less compact.
- **Access**: Open access via PMLR.
  https://proceedings.mlr.press/v80/verma18a/verma18a.pdf

---

## 4. Interpretability of Internal Representations

### 4.1 Analyzing Hidden Activations of Deep Policy Networks

**McInroe, T., Spurrier, M., Sieber, J., & Conneely, S. (2021).** Analyzing the
Hidden Activations of Deep Policy Networks: Why Representation Matters.
*arXiv:2103.06398*.

- **Key findings**: Shows that a well-organized internal representation is a
  prerequisite for learning good action-selection. A poor initial representation
  can cause unrecoverable collapse. A good representation allows organization
  even before training begins.
- **Relevance**: Directly supports our hypothesis that physics-informed features
  provide a better initial representation, preventing collapse and enabling smaller
  networks. If we probe our 128x128x64 network's hidden layers, we should find
  units that correlate with our hand-crafted features.
- **Access**: arXiv (open access). https://arxiv.org/abs/2103.06398

### 4.2 Properties of Neural Network Representations in RL

**Wang, H., Miahi, E., White, M., Machado, M.C., et al. (2024).** Investigating
the Properties of Neural Network Representations in Reinforcement Learning.
*Artificial Intelligence*, 330, 104100.

- **Key findings**: Introduces and measures six representational properties over
  25,000+ agent-task settings. Demonstrates that better representations (as measured
  by linear probing) lead to better performance after retraining. Understanding
  representation evolution during training reveals possible failure modes.
- **Relevance**: Provides the methodology (linear probing, representational metrics)
  we could use to test whether the 128x128x64 network's hidden layers learn our
  physics-informed features.
- **Access**: ScienceDirect. DOI: 10.1016/j.artint.2024.104100
  arXiv preprint: https://arxiv.org/abs/2203.15955

### 4.3 Formal Verification of Neural Networks for Safety-Critical RL

**Corsi, D., Marchesini, E., & Farinelli, A. (2021).** Formal Verification of
Neural Networks for Safety-Critical Tasks in Deep Reinforcement Learning.
*Proceedings of UAI 2021*.

- **Key findings**: Provides a formulation for safety properties ensuring rational
  decisions. Introduces ProVe, a formal verification tool. Shows that models
  respecting soft constraints are safer than models evaluated only on long-term
  reward.
- **Relevance**: Motivates our SINDy-RL distillation: a symbolic policy is far
  easier to formally verify than a neural network. The safety-critical nature of
  rocket landing demands verification.
- **Access**: Open access via PMLR.
  https://proceedings.mlr.press/v161/corsi21a/corsi21a.pdf

### 4.4 Mechanistic Interpretability Review

**Bereska, L.F. & Gavves, E. (2024).** Mechanistic Interpretability for AI Safety:
A Review. *arXiv:2404.14082*.

- **Key findings**: Comprehensive review of mechanistic interpretability techniques:
  linear probes, sparse autoencoders, activation patching, circuit analysis.
  Features are the building blocks of network understanding. Probing trains simple
  classifiers on internal activations to test for specific information.
- **Relevance**: Provides the technical toolkit for our interpretability analysis.
  We can use linear probes on hidden layers to test if neurons encode sqrt(y),
  vy^2/(2y), etc. The concept of "superposition" (Elhage et al. 2022) may explain
  why larger networks can represent more physics-informed features than they have
  dedicated neurons for.
- **Access**: arXiv (open access). https://arxiv.org/abs/2404.14082

---

## 5. Network Architecture and Capacity for Control

### 5.1 Training Larger Networks for Deep RL

**Ota, K., Oiki, T., Jha, D., Mariyama, T., & Nikovski, D. (2021).** Training
Larger Networks for Deep Reinforcement Learning. *arXiv:2102.07920*. Published in
*Machine Learning* (Springer), 2024.

- **Key findings**: Simply building deeper MLPs does not improve DRL performance.
  A wider MLP generally improves performance, but deeper MLPs do not. This is
  unlike supervised learning. The loss surface of deeper networks is more complex
  and sensitive to hyperparameters, explaining why width is preferred over depth
  for RL.
- **Relevance**: Directly relevant to our architecture comparison (64x64 with
  features vs. 128x128x64 without). The finding that width matters more than depth
  suggests our 128x128x64 network may be learning wider representations of the
  physics, while the 64x64 + features network gets the same information more
  efficiently.
- **Access**: arXiv (open access). https://arxiv.org/abs/2102.07920

### 5.2 Understanding Plasticity in Neural Networks

**Lyle, C., Zheng, Z., Nikishin, E., et al. (2023).** Understanding Plasticity in
Neural Networks. *Proceedings of ICML 2023*, PMLR 202.

- **Key findings**: Deep RL networks lose plasticity (ability to change predictions
  in response to new information) during training. Connected to changes in loss
  landscape curvature. Identifies parameterization and optimization choices that
  preserve plasticity.
- **Relevance**: Plasticity loss could differentially affect our two architectures.
  The 128x128x64 network may be more susceptible to plasticity loss, while the
  64x64 + features network, by having a simpler representation to learn, may
  maintain plasticity better.
- **Access**: Open access via PMLR.
  https://proceedings.mlr.press/v202/lyle23b/lyle23b.pdf

### 5.3 Universal Approximation with Quadratic Networks

**Fan, F., Xiong, J., & Wang, G. (2020).** Universal Approximation with Quadratic
Deep Networks. *Neural Networks*, 124, 383-392.

- **Key findings**: A ReLU quadratic network can represent any univariate polynomial
  globally and uniquely. Quadratic networks are more efficient approximators for
  polynomial function classes than standard networks. With the same structure,
  quadratic networks can approximate functions that conventional networks cannot.
- **Relevance**: Our physics-informed features include quadratic terms (vy^2/(2y)).
  This result suggests that without these features, the network must use more
  capacity to implicitly approximate the same quadratic relationships.
- **Access**: ScienceDirect. DOI: 10.1016/j.neunet.2020.01.007

### 5.4 Physics-Informed Neural Nets for Control (PINC)

**Antonelo, E.A., Camponogara, E., Seman, L.O., et al. (2024).** Physics-Informed
Neural Nets for Control of Dynamical Systems. *Neurocomputing*, 579, 127419.

- **Key findings**: Proposes a PINN-based architecture augmented with extra inputs
  (initial state, control input) for control applications. Drawing from multiple
  shooting and collocation methods. Shows physics-informed networks learn dynamics
  more accurately and respect conservation laws better than standard DNNs.
- **Relevance**: Demonstrates that physics-informed architectures (not just
  physics-informed observations) improve control. Our feature engineering approach
  is the observation-level analog of their architecture-level approach.
- **Access**: arXiv preprint available. https://arxiv.org/abs/2104.02556
  Journal: DOI: 10.1016/j.neucom.2024.127419

---

## 6. Summary of Key Themes

### 6.1 Physics-Informed Observations Improve Sample Efficiency

Multiple papers demonstrate that encoding domain knowledge into observations
(rather than relying on the network to discover it) improves both sample efficiency
and generalization:
- Hamilton et al. (2025): engineered sensors help spacecraft RL
- Furfaro et al. (2020): time-to-go and velocity error outperform raw state
- Gaudet et al. (2020): minimal physics-informed obs generalize to unseen regions
- Banerjee et al. (2023): taxonomy identifies "observational bias" as a primary
  method for physics-informed RL

### 6.2 Distillation to Symbolic Form is Feasible and Beneficial

The SINDy-RL pipeline (Zolman et al. 2025) and related symbolic distillation
methods (Landajuela et al. 2021, Cranmer et al. 2020, SPID) demonstrate that:
- Neural policies CAN be distilled to symbolic form
- Symbolic policies are orders of magnitude smaller
- Symbolic policies sometimes generalize BETTER than neural ones
- The choice of candidate library functions is critical for success

### 6.3 Network Architecture Matters for What is Learned

- Width matters more than depth for RL (Ota et al. 2021)
- Well-organized internal representations are prerequisites for learning (McInroe et al. 2021)
- Plasticity loss is a real concern for larger networks (Lyle et al. 2023)
- Quadratic networks are more efficient for polynomial function classes (Fan et al. 2020)

### 6.4 Verification Requires Interpretability

- Neural network verification for safety-critical control is possible but hard (Corsi et al. 2021)
- Distilled symbolic/decision-tree policies are far easier to verify (Bastani et al. 2018)
- Rocket engine control has been demonstrated on real hardware with DRL (Dresia 2025),
  raising the practical urgency of verification

---

## 7. Implications for Our Project

1. **Feature design is well-motivated**: Our sqrt(y), vy^2/(2y), horiz_dist, and
   horiz_dist/y features align with the "observational bias" approach documented
   across the PIRL literature. The Furfaro/Gaudet work on ZEM/ZEV provides the
   strongest precedent.

2. **The interpretability experiment is novel**: While many papers train RL with
   or without engineered features, few explicitly probe whether a larger network
   internally learns the same features. This is a gap we can fill using the
   linear probing methodology from Wang et al. (2024) and McInroe et al. (2021).

3. **SINDy-RL is the right distillation tool**: It is published in Nature
   Communications (2025), includes policy distillation via behavior cloning with
   sparse dictionary learning, and our physics-informed features are natural
   candidate library functions.

4. **The architecture comparison needs careful framing**: The 64x64 + features
   vs. 128x128x64 comparison is not just about parameter count -- it is about
   whether explicit physics encoding can substitute for implicit feature learning.
   The Structured Control Nets paper (Srouji et al. 2018) provides the theoretical
   framing.
