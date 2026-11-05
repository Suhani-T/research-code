"""
Independent fixed-time synchronization design for Lorenz master-slave systems.
Design basis: implicit-Lyapunov-function (ILF) fixed-time theorem (Polyakov, Efimov,
Perruquetti, Automatica 2015, Theorem 5 / Theorem 9), specialised to Q(V,e)=e'e/V^2-1.
This is NOT the controller of Shafiq & Ahmad (2026); that law is not in the supplied PDF.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import matplotlib.pyplot as plt

# ---------------- parameters ----------------
SIG, RHO, B = 10.0, 28.0, 8.0 / 3.0          # Lorenz parameters (identical master/slave)
KAPPA = 2.0                                   # feedback gain, must satisfy KAPPA > D_BAR
NU    = 0.5                                   # outer-phase exponent, nu > 0
D_BAR = 0.6                                   # assumed bound: ||d(t)||_2 <= D_BAR
V_MIN = 1e-3                                  # regularisation radius (simulation only)
A     = KAPPA - D_BAR                         # guaranteed decay constant
T_BOUND = 1.0 / A + 1.0 / (A * NU)            # fixed-time bound, independent of e(0)

def lorenz(s):
    x, y, z = s
    return np.array([SIG * (y - x), x * (RHO - z) - y, x * y - B * z])

def dist(t):                                  # matched, bounded, time-varying disturbance
    return np.array([0.3 * np.sin(2 * t), 0.3 * np.cos(3 * t), 0.2 * np.sin(t) + 0.1])

# check the disturbance really satisfies ||d|| <= D_BAR
_tt = np.linspace(0, 100, 200001)
assert max(np.linalg.norm(dist(t)) for t in _tt[::50]) <= D_BAR

def v_law(e):
    """Fixed-time feedback v(e). Ideal law: -KAPPA*e/||e|| (||e||<1), -KAPPA*||e||^NU*e (||e||>=1).
    Inside ||e||<V_MIN it is replaced by the linear law -KAPPA*e/V_MIN (continuous; cf. Remark 11
    and Algorithm 10 of the 2015 paper) to avoid numerical chattering."""
    n = np.linalg.norm(e)
    if n >= 1.0:
        return -KAPPA * n**NU * e
    if n >= V_MIN:
        return -KAPPA * e / n
    return -KAPPA * e / V_MIN

def control(xm, e):
    df = lorenz(xm + e) - lorenz(xm)          # f(y)-f(x): exact model cancellation
    v = v_law(e)
    return -df + v, v, df

def rhs(t, s):
    xm, e = s[:3], s[3:]
    fm = lorenz(xm)
    u, v, df = control(xm, e)
    return np.concatenate([fm, df + u + dist(t)])   # master; error dynamics e' = df+u+d

def Q(V, e):                                   # implicit Lyapunov equation Q(V,e)=0
    return e @ e / V**2 - 1.0

# ---------------- main simulation ----------------
xm0 = np.array([-5.0, 6.0, 20.0])
xs0 = np.array([8.0, -7.0, 35.0])
e0 = xs0 - xm0
T_END = 6.0
t_eval = np.linspace(0, T_END, 6001)
sol = solve_ivp(rhs, (0, T_END), np.concatenate([xm0, e0]), method="LSODA",
                t_eval=t_eval, rtol=1e-9, atol=1e-12)
t = sol.t
XM, E = sol.y[:3].T, sol.y[3:].T
XS = XM + E
U = np.zeros_like(E); Vc = np.zeros_like(E)
for i in range(len(t)):
    u, v, _ = control(XM[i], E[i]); U[i], Vc[i] = u + 0.0, v
V = np.linalg.norm(E, axis=1)                  # Lyapunov function V(e)=||e||

# verify implicit definition: root of Q(V,e)=0 equals ||e||
for i in (0, 500, 2000):
    if V[i] > 1e-9:
        Vroot = brentq(lambda w: Q(w, E[i]), 1e-12, 1e6)
        assert abs(Vroot - V[i]) < 1e-6 * max(1, V[i])

# comparison-system envelope from  V' <= -A V^(1+NU) (V>=1),  V' <= -A (V<1)
V0 = V[0]
t1 = (1 - V0**(-NU)) / (A * NU) if V0 > 1 else 0.0
def envelope(tt):
    tt = np.asarray(tt)
    out = np.empty_like(tt)
    m = tt < t1
    out[m] = (V0**(-NU) + A * NU * tt[m])**(-1 / NU)
    out[~m] = np.maximum(1.0 - A * (tt[~m] - t1), 0.0)
    return out
env = envelope(t)

reach = t[np.argmax(V <= V_MIN)]
print(f"||e(0)|| = {V0:.3f}")
print(f"Fixed-time bound T_max = 1/A + 1/(A*nu) = {T_BOUND:.4f} s  (A={A}, nu={NU})")
print(f"Time to reach ||e|| <= {V_MIN}: {reach:.4f} s")
print(f"max(V - envelope) = {np.max(V - env):.2e}  (should be <= ~0)")
print(f"final ||e|| = {V[-1]:.2e}; residual ball radius bound V_MIN*D_BAR/KAPPA = {V_MIN*D_BAR/KAPPA:.2e}")

# ---------------- fixed-time check: sweep of initial error magnitude ----------------
print("\nSweep over ||e(0)|| (direction (1,-1,1)/sqrt3):  reaching time of ||e||<=V_MIN")
ev = lambda tt, s: np.linalg.norm(s[3:]) - V_MIN
ev.terminal, ev.direction = True, -1
for scale in [0.1, 1, 10, 1e2, 1e3, 1e4]:
    e_init = scale * np.array([1, -1, 1]) / np.sqrt(3)
    r = solve_ivp(rhs, (0, 20), np.concatenate([xm0, e_init]), method="LSODA",
                  events=ev, rtol=1e-9, atol=1e-12)
    tr = r.t_events[0][0] if len(r.t_events[0]) else np.nan
    print(f"  ||e0|| = {scale:9.1f}  ->  t = {tr:.4f} s   (bound {T_BOUND:.4f} s)  {'OK' if tr <= T_BOUND else 'VIOLATION'}")

# ---------------- plots ----------------
fig, ax = plt.subplots(3, 3, figsize=(16, 12))
names = ["x1", "x2", "x3"]
for i in range(3):
    ax[0, i].plot(t, XM[:, i], label=f"master {names[i]}")
    ax[0, i].plot(t, XS[:, i], "--", label=f"slave {names[i]}")
    ax[0, i].set_title(f"State {names[i]}"); ax[0, i].set_xlabel("t [s]"); ax[0, i].legend(); ax[0, i].grid(alpha=.3)
ax[1, 0].plot(t, E); ax[1, 0].set_title("Synchronization errors e_i"); ax[1, 0].legend(["e1", "e2", "e3"]); ax[1, 0].grid(alpha=.3)
ax[1, 1].semilogy(t, V + 1e-16, label="V=||e||"); ax[1, 1].semilogy(t, env + 1e-16, "r--", label="comparison envelope")
ax[1, 1].axvline(T_BOUND, color="k", ls=":", label=f"T_max={T_BOUND:.2f}s"); ax[1, 1].set_ylim(1e-5, 1e2)
ax[1, 1].set_title("Lyapunov function V(t) (log)"); ax[1, 1].legend(); ax[1, 1].grid(alpha=.3, which="both")
ax[1, 2].plot(t, V, label="V=||e||"); ax[1, 2].plot(t, env, "r--", label="envelope"); ax[1, 2].set_xlim(0, 3)
ax[1, 2].set_title("V(t) (linear)"); ax[1, 2].legend(); ax[1, 2].grid(alpha=.3)
ax[2, 0].plot(t, U); ax[2, 0].set_title("Total control u (incl. cancellation)"); ax[2, 0].legend(["u1", "u2", "u3"]); ax[2, 0].grid(alpha=.3)
ax[2, 1].plot(t, Vc); ax[2, 1].set_title("Stabilizing part v"); ax[2, 1].legend(["v1", "v2", "v3"]); ax[2, 1].grid(alpha=.3)
ax[2, 1].set_xlim(0, 3)
ax[2, 2].remove(); ax3 = fig.add_subplot(3, 3, 9, projection="3d")
ax3.plot(*XM.T, lw=.6, label="master"); ax3.plot(*XS.T, lw=.6, ls="--", label="slave")
ax3.set_title("Phase portrait"); ax3.legend()
for a in ax[1:, :2].ravel(): a.set_xlabel("t [s]")
plt.tight_layout(); 