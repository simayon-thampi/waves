# BLE Channel Sounding Ranging Pipeline — Deep Dive

## 0. Signal Model Foundation

BLE Channel Sounding transmits a known tone on each **channel index** $k$, which maps to a carrier frequency:

$$f_k = f_0 + k \cdot \Delta f \quad \text{(Hz)}$$

where $\Delta f = 1\,\text{MHz}$ for CS. The received complex channel response at frequency $f_k$ for a single reflector at distance $d$ is:

$$H(f_k) = A_k \cdot e^{j\phi_k}, \quad \phi_k = -\frac{2\pi f_k \cdot 2d}{c} + \phi_0$$

The **two-way phase** accumulated at channel $k$:

$$\phi_k = -\frac{4\pi f_k d}{c} + \phi_0$$

This is linear in $f_k$. The **slope** is:

$$\frac{d\phi}{df} = -\frac{4\pi d}{c} \quad \text{(rad/Hz)}$$

So:

$$\boxed{d = -\frac{c}{4\pi} \cdot \frac{d\phi}{df}}$$

Everything downstream is trying to robustly estimate this slope.

---

## 1. Cleaned ChannelResponse (~50–65 Tones)

Before any estimation, raw tones are filtered:

| Rejection Criterion | Why |
|---|---|
| Amplitude below threshold | Low SNR tones corrupt phase |
| Phase discontinuities > threshold | Likely cycle slips |
| Advertising/reserved channels | Protocol artifacts |
| Outlier IQ points | Hardware glitches |

**Result:** A sparse set of $(f_k, \phi_k, A_k)$ tuples, typically 50–65 out of 79 channels.

---

## 2. Phase Unwrapping

Raw phase from the radio is wrapped to $[-\pi, +\pi)$:

$$\phi_k^{\text{raw}} = \text{angle}(H(f_k)) \in (-\pi, \pi]$$

Unwrapping removes $2\pi$ jumps:

$$\phi_k^{\text{unwrapped}} = \phi_k^{\text{raw}} + 2\pi \cdot n_k$$

where $n_k \in \mathbb{Z}$ is chosen so adjacent phase differences are $< \pi$:

$$n_k = n_{k-1} - \text{round}\!\left(\frac{\phi_k^{\text{raw}} - \phi_{k-1}^{\text{raw}}}{2\pi}\right)$$

The unwrapped phase is what the estimators actually fit.

---

## 3. Scene Classifier

Runs **before** RANSAC to label the propagation environment. Uses three features:

### 3.1 Null Depth

In multipath, constructive/destructive interference creates amplitude fades. The null depth captures this:

$$\text{NullDepth} = \frac{A_{\max} - A_{\min}}{A_{\max} + A_{\min}} \in [0, 1]$$

- **LOS:** shallow fades → NullDepth ≈ 0.1–0.3
- **Multipath:** deep fades → NullDepth > 0.5

### 3.2 Phase Residual RMS

Fit a line through all unwrapped phases, compute residuals:

$$\hat{\phi}_k = \hat{m} \cdot f_k + \hat{b}$$

$$\text{PhaseRMS} = \sqrt{\frac{1}{N}\sum_{k=1}^{N}(\phi_k - \hat{\phi}_k)^2} \quad \text{(rad)}$$

- **LOS:** small residuals (near-linear phase)
- **Multipath/NLOS:** large residuals due to phase distortion

### 3.3 Reject Fraction

$$r = \frac{\text{tones rejected in cleaning step}}{79}$$

High $r$ → heavily cluttered channel.

### Classification Logic (thresholded decision tree)

```
NullDepth < τ_N  AND  PhaseRMS < τ_P  →  LOS
NullDepth ≥ τ_N  OR   PhaseRMS ≥ τ_P  →  MULTIPATH
r > τ_r                                →  NLOS (hard block)
```

The label is passed to the estimators to select the best algorithm.

---

## 4. RANSAC — Shared Robust Fitting Core

All three estimators use this as their inlier selection stage.

**Inputs:** $N$ pairs $\{(f_k,\, \phi_k)\}_{k=1}^{N}$

**Goal:** Find slope $m$ and intercept $b$ such that the linear model $\hat\phi = m f + b$ fits the majority of data, ignoring outliers from multipath.

### Algorithm

```
best_score = 0
best_slope, best_intercept = None, None

for i in 1..100:
    # 1. Minimal sample
    S = random sample of 6 tones from {(f_k, φ_k)}

    # 2. Hypothesis: fit line to S
    m_i, b_i = polyfit(f[S], φ[S], deg=1)

    # 3. Consensus: count inliers across all N tones
    residuals = |φ_k - (m_i·f_k + b_i)|  for all k
    inlier_mask = (residuals < τ_RANSAC)
    score = sum(inlier_mask)

    # 4. Update best
    if score > best_score:
        best_score = score
        best_slope = m_i
        best_intercept = b_i
        best_inlier_mask = inlier_mask
```

> **Why 6-point samples?** A line needs only 2 points, but 6 gives a statistically stable initial fit while remaining fast. The minimum is overconstrained to reduce degenerate hypotheses.

**Inlier threshold $\tau_{\text{RANSAC}}$** is tuned empirically (e.g., 0.3–0.5 rad), representing the maximum phase error attributable to noise rather than a multipath component.

**Output:** `best_slope` (rad/Hz), `best_intercept` (rad), `inlier_mask` (boolean array, length $N$)

---

## 5. Weighted Least Squares Estimator ⭐ *Recommended for Multipath*

After RANSAC selects inliers, this refines the estimate using **amplitude as a reliability weight**.

**Motivation:** Tones with higher received amplitude $A_k$ have higher SNR and therefore more trustworthy phase measurements:

$$\sigma_{\phi_k}^2 \approx \frac{1}{2 \cdot \text{SNR}_k} \approx \frac{\sigma_n^2}{A_k^2}$$

So weight $w_k \propto A_k^2$ (or simply $w_k = A_k$ in practice).

### Weighted Polyfit

Let $\mathbf{f}_{\text{in}}$, $\boldsymbol{\phi}_{\text{in}}$, $\mathbf{w}_{\text{in}}$ be the inlier frequencies, phases, and amplitudes:

$$\mathbf{W} = \text{diag}(w_1, \ldots, w_M), \quad \mathbf{F} = \begin{bmatrix} f_1 & 1 \\ \vdots & \vdots \\ f_M & 1 \end{bmatrix}$$

Weighted Least Squares solution:

$$\begin{bmatrix} \hat{m} \\ \hat{b} \end{bmatrix} = (\mathbf{F}^T \mathbf{W} \mathbf{F})^{-1} \mathbf{F}^T \mathbf{W} \boldsymbol{\phi}$$

### Distance Conversion

$$\boxed{d_{\text{WLS}} = -\hat{m} \cdot \frac{c}{4\pi}}$$

where $c = 3 \times 10^8\,\text{m/s}$ and $\hat{m}$ is in rad/Hz.

**Why recommended for multipath?** Low-amplitude tones are precisely the ones experiencing destructive interference (nulls), and they carry the worst phase information. Down-weighting them is physically motivated.

---

## 6. Phase Slope Estimator (Unweighted)

Same as WLS but $w_k = 1$ for all inliers:

$$\hat{m} = \frac{\sum (f_k - \bar{f})(\phi_k - \bar{\phi})}{\sum (f_k - \bar{f})^2}$$

or equivalently `numpy.polyfit(f_in, phi_in, 1)`.

$$\boxed{d_{\text{PS}} = -\hat{m} \cdot \frac{c}{4\pi}}$$

**Limitation:** Equal weight to all inliers regardless of their SNR. A surviving inlier at a deep fade can still bias the slope.

---

## 7. IFFT Estimator

This estimator works in the **delay domain** rather than fitting a slope.

### 7.1 Build the Channel Transfer Function

Reconstruct $H(f)$ from the inlier subset:

$$H(f_k) = A_k \cdot e^{j\phi_k} \quad \text{for } k \in \text{inliers}$$

Non-inlier bins are **zeroed out** (sparse spectrum):

$$H[k] = \begin{cases} A_k e^{j\phi_k} & k \in \text{inliers} \\ 0 & \text{otherwise} \end{cases}$$

### 7.2 Apply Hann Window

To suppress sidelobes caused by the non-contiguous, sparse spectrum:

$$w_k = 0.5\left(1 - \cos\!\left(\frac{2\pi k}{K-1}\right)\right)$$

$$\tilde{H}[k] = H[k] \cdot w_k$$

### 7.3 IFFT → Channel Impulse Response

$$h[n] = \text{IFFT}(\tilde{H})[n]$$

The time axis is:

$$t_n = \frac{n}{B_{\text{total}}} \quad \text{where } B_{\text{total}} = N_{\text{bins}} \cdot \Delta f$$

For BLE CS with 79 channels at 1 MHz spacing: $B = 79\,\text{MHz}$, giving a delay resolution of:

$$\Delta t = \frac{1}{B} \approx 12.7\,\text{ns} \quad \Rightarrow \quad \Delta d = \frac{c}{2B} \approx 1.9\,\text{m}$$

### 7.4 Peak Search in Causal Window

The direct-path echo arrives within $[0, 8\,\text{ns}]$ (i.e., $d < 2.4\,\text{m}$ at $t = 8\,\text{ns}$). Search only here to reject late-arriving multipath:

$$n^* = \arg\max_{n:\, t_n \in [0,\, 8\,\text{ns}]} |h[n]|^2$$

$$\boxed{d_{\text{IFFT}} = t_{n^*} \cdot c}$$

> **Resolution note:** At 79 MHz bandwidth, $\Delta d \approx 1.9\,\text{m}$. Sub-meter precision requires zero-padding the IFFT (interpolation in the delay domain).

---

## 8. Summary Table

| Stage | What it computes | Key formula |
|---|---|---|
| **Phase model** | Phase vs frequency is linear | $\phi_k = -\frac{4\pi f_k d}{c} + \phi_0$ |
| **Unwrapping** | Remove $2\pi$ ambiguity | $\phi_k^u = \phi_k^r + 2\pi n_k$ |
| **Scene Classifier** | LOS / MP / NLOS label | NullDepth, PhaseRMS, $r$ |
| **RANSAC** | Robust inlier selection | 100 iter × 6-pt samples, $\tau$ threshold |
| **WLS** | Amplitude-weighted slope | $(\mathbf{F}^T\mathbf{W}\mathbf{F})^{-1}\mathbf{F}^T\mathbf{W}\boldsymbol{\phi}$ |
| **Phase Slope** | Unweighted slope | `polyfit(f, φ, 1)` |
| **IFFT** | Delay-domain peak | $d = t_{\text{peak}} \cdot c$, peak in $[0, 8\,\text{ns}]$ |
| **Distance** | All slope estimators | $d = -\hat{m} \cdot c / 4\pi$ |

---

## 9. Estimator Selection Rationale

```
LOS       → Phase Slope is fast and sufficient (low residuals, few outliers)
Multipath → WLS is preferred (amplitude weighting suppresses null-fade tones)
NLOS      → IFFT can sometimes recover direct path from CIR shape;
             WLS degrades badly if the direct path is fully blocked
```

The scene classifier exists precisely to route the measurement to the most appropriate estimator rather than using a one-size-fits-all approach.
