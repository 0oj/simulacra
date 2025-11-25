# Why is this project structured the way it is?

Our goal is to write good control software for our rocket. The simulated world exists because we want to be able to test different algorithms and see their results. Since the search space is rather large, we want to iterate quickly and cheaply. The issue now is this: we have to simulate the world. In real life, physics is taken care of by magic ie. the laws of the universe. In our simulation, we have to use the found mathematical models to replicate the real world. Some mathematical models are simple, like SUVAT equations. Others are complex, and time-consuming to simulate, but more accurate, like CFDs. But that isn't even our core problem. The reason for this structure: between noumena, phenomena, and onboard is the following... In order to simulate the next timestep, we need to work with "True" information ie. the actual position of the rocket, the actual velocity of the rocket, the actual orientation of the rocket, the actual thrust vector coming out of the rocket, and the torques it will actually apply. These are the noumena. This is information that is true in our happy mathematical world run by god who exactly applies his laws. However, from within the rocket, it's a troubled world, and we are both blind and seeing. We can now only measure. This is phenomena. The inbetween layer. True information from noumena is taken by phenomena and passed to onboard. Phenomena exposes interfaces to observe and interact with the world. These interfaces are sensors, and actuators. The onboard computer only gets to have information about the true world through its sensors. These sensors though, are measuring instruments, which are noisy, and biased, and erroneous - all things that we model. The rocket then uses this information and makes its best efforts to control the rocket. This is onboard, this is the Controller. Then, using the actuators provided by phenomena, it tries to effect change, and steer itself. Phenomena then passes these actions such as: changing the gimbal angles, to noumena, who then converts it back into happy math land (obviously, even these actions aren't perfect, and phenomena has taken care of adding noise and inaccuracy). But when it's been passed to noumena, we assume it is true, and we are in happy math land of clean vectors and tensors and what not. We integrate the rocket computer's effects, take a timestep forward with our simulation, and the cycle repeats.

# Conventions
Here’s a block you can paste under `# Conventions`:

---

In this project we fix a single 3D geometric language so that physics, control, and visualisation all talk about the same world. We use a **right‑handed coordinate system** with **RGB = XYZ**: the **X axis is red**, **Y axis is green**, and **Z axis is blue**, such that \(x \times y = z\). Both the **world frame** and the **rocket’s body frame** use these same colored axes; the world frame is fixed to “the universe”, while the body frame is rigidly attached to the rocket. The rocket’s **orientation** is represented by a **unit quaternion** \(q = (w, x, y, z)\) which maps **body vectors into world vectors**. Internally, quaternions live in the basis \(\{1, i, j, k\}\) where \(i\), \(j\), and \(k\) correspond to rotations about the **X (red)**, **Y (green)**, and **Z (blue)** axes respectively, with the standard multiplication rules \(i^2 = j^2 = k^2 = ijk = -1\) and \(ij = k,\ jk = i,\ ki = j\). The rocket’s **angular velocity** is stored as a 3‑vector \(\boldsymbol{\omega}^B = [\omega_x,\ \omega_y,\ \omega_z]\) expressed in the **body frame** (roll, pitch, yaw rates about the rocket’s own RGB axes) in radians per second. Its **moment of inertia** is a \(3 \times 3\) **inertia tensor** \(I^B\) in the body frame about the center of mass, typically diagonal for a symmetric rocket but allowed to be a full symmetric matrix. From these, **angular momentum** in the body frame is \(h^B = I^B \boldsymbol{\omega}^B\), and world‑frame quantities are obtained by applying the rotation \(R(q)\) derived from the quaternion. All dynamics, sensor models, and controllers are written against these conventions so that “noumena” (true state), “phenomena” (sensors/actuators), and “onboard” (controller/estimator) share a consistent mental model of 3D space.

---

# Conventions

## Axes & colors (right‑handed)
  - **X (red)**, **Y (green)**, **Z (blue)** with \(x \times y = z\).
  - Right‑hand rule: **thumb = X**, **index = Y**, **middle/ring = Z**.

## Basis planes
  - **XY plane**: spanned by X (red) and Y (green), normal +Z (blue).
  - **YZ plane**: spanned by Y (green) and Z (blue), normal +X (red).
  - **ZX plane**: spanned by Z (blue) and X (red), normal +Y (green).
  - **Quaternion basis-plane mapping**:  
    - \(i\) = rotation about X (red), lies in the **YZ plane**  
    - \(j\) = rotation about Y (green), lies in the **ZX plane**  
    - \(k\) = rotation about Z (blue), lies in the **XY plane**

## Frames
  - **World frame**: fixed reference; gravity, wind, and positions live here.
  - **Body frame**: attached to rocket; thrust vector, torques, and IMU readings are naturally expressed here.

## Quaternions
  - Stored as `(w, x, y, z)` corresponding to \(w \cdot 1 + x \cdot i + y \cdot j + z \cdot k\).
  - **Axis mapping**:  
    - \(i\) ↔ X (red), \(j\) ↔ Y (green), \(k\) ↔ Z (blue).
  - **Basis plane mapping**:  
    - \(i\) (X/red): rotation in the YZ plane  
    - \(j\) (Y/green): rotation in the ZX plane  
    - \(k\) (Z/blue): rotation in the XY plane
  - **Algebra**:  
    - \(i^2 = j^2 = k^2 = ijk = -1\).  
    - \(ij = k,\ jk = i,\ ki = j\); reversed order gains a minus sign (e.g. \(ji = -k\)).

## Orientation
  - `orientation` is a **unit quaternion** mapping **body → world**.
  - Rotation matrix \(R(q)\) derived from `orientation` converts body vectors to world vectors.

## Angular velocity
  - Stored as **body‑frame** vector \(\boldsymbol{\omega}^B = [\omega_x,\omega_y,\omega_z]\) in rad/s.
  - Components are roll (about X/red), pitch (about Y/green), yaw (about Z/blue).
  - Quaternion dynamics: \(\dot{q} = \tfrac{1}{2}\,\Omega(\boldsymbol{\omega}^B)\,q\).

## Inertia & angular momentum
  - **Inertia tensor**: \(I^B \in \mathbb{R}^{3 \times 3}\), body frame, about COM (often diagonal \(\text{diag}(I_x, I_y, I_z)\)).
  - **Angular momentum**: \(h^B = I^B \boldsymbol{\omega}^B\) (body frame), \(h^W = R(q) h^B\) (world frame).
  - Free‑space rigid‑body dynamics: \(I^B \dot{\boldsymbol{\omega}}^B + \boldsymbol{\omega}^B \times (I^B \boldsymbol{\omega}^B) = \tau^B\).
