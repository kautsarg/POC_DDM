# Denoising Quality Metrics: Reference Guide

Three metrics used to rank denoising methods in `multi_wavelet_denoising_v2.ipynb`:
**Residual autocorr lag-1**, **SNR (dB)**, and **Fidelity (corr)**.

---

## 1. SNR (dB) — Signal-to-Noise Ratio

### What it is
SNR measures how much of the signal variance survived the filter relative to how much noise was removed. It answers: *did the filter remove noise without throwing away signal?*

### Formula

$$\text{SNR}(\hat{x}) = 10 \log_{10} \left( \frac{\text{Var}(\hat{x})}{\text{Var}(x - \hat{x})} \right)$$

where $x$ is the raw signal, $\hat{x}$ is the denoised signal, and $x - \hat{x}$ is the extracted noise.

**In code** (per sample, then `nanmean` across all samples):
```python
noise  = raw - denoised
vd, vn = np.var(denoised, axis=1), np.var(noise, axis=1)
snr_v  = np.where(vn > 0, 10 * np.log10(vd / vn), np.nan)
SNR    = float(np.nanmean(snr_v))
```

Samples where the extracted noise has zero variance (the filter changed nothing) produce `nan` and are excluded from the average.

### Interpretation
| Value | Meaning |
|---|---|
| Very high (e.g. > 20 dB) | Almost all variance in the output is genuine signal; very little noise remains |
| Moderate (5–15 dB) | A balance: the filter removed some noise while keeping most shape |
| Low or negative | More variance in the noise channel than in the denoised signal — likely over-smoothed or the "noise" is correlated/structured |

**Direction: ↑ higher is better.**

A subtlety: SNR is computed from variance, not energy. An aggressive smoother that flattens the curve to near-zero will have both low `Var(ŷ)` and low `Var(noise)`, potentially yielding a misleading high SNR. Use alongside Fidelity and TV ratio to catch this.

### Why it matters here
Kinetic fluorescence curves contain both genuine dynamics (exponential rise, sigmoidal shape) and measurement noise. SNR distinguishes filters that genuinely separate signal from noise from those that merely attenuate everything. It is the primary "how clean is the output?" metric.

### Key references
- Shannon, C. E. (1948). *A Mathematical Theory of Communication.* Bell System Technical Journal. — foundational formulation of SNR in terms of signal power.
- Johnson, D. H. (2006). *Signal-to-noise ratio.* Scholarpedia. — modern treatment of empirical SNR estimation for discrete digital signals, discusses variance-based estimators.
- Kay, S. M. (1993). *Fundamentals of Statistical Signal Processing, Volume I: Estimation Theory.* Prentice Hall. — standard reference for SNR in the context of Wiener filtering and optimal estimation.

---

## 2. Fidelity (corr) — Pearson Correlation

### What it is
Fidelity measures how well the denoised curve preserves the *shape* of the original. It answers: *did the filter maintain the overall trajectory, or did it distort the signal's dynamics?*

### Formula

$$\text{Fidelity}(\hat{x}) = \frac{1}{N} \sum_{i=1}^{N} \rho(x_i,\, \hat{x}_i)$$

where $\rho$ is the Pearson correlation coefficient:

$$\rho(a, b) = \frac{\sum_t (a_t - \bar{a})(b_t - \bar{b})}{\sqrt{\sum_t (a_t - \bar{a})^2} \cdot \sqrt{\sum_t (b_t - \bar{b})^2}}$$

**In code** (per sample Pearson r, then `nanmean`):
```python
from scipy.stats import pearsonr

def _safe_corr(a, b):
    return np.nan if np.std(a) == 0 or np.std(b) == 0 else pearsonr(a, b)[0]

fidelity = float(np.nanmean([_safe_corr(raw[i], denoised[i]) for i in range(len(raw))]))
```

Constant signals (zero std) return `nan` and are excluded.

### Interpretation
| Value | Meaning |
|---|---|
| ≈ 1.0 | Near-perfect shape preservation; the denoised curve tracks the raw trajectory faithfully |
| 0.95–0.99 | Good; minor smoothing artefacts |
| < 0.9 | Shape distortion; peaks shifted, dynamics blunted, or over-aggressive smoothing |

**Direction: ↑ higher is better.**

Pearson correlation is scale- and mean-invariant, which makes it *insensitive to amplitude or offset changes* introduced by the filter. A filter that scales every value by 0.5 gets a Fidelity of 1.0. This is intentional for this use case — we care about trajectory shape, not absolute amplitude. For amplitude fidelity, check the SD_max ratio metric.

### Why it matters here
Kinetic curves encode biologically meaningful dynamics (time-to-peak, slope, plateau). Fidelity catches filters that shift, distort, or smear these features even if the output looks visually "clean". It is complementary to SNR: a filter can have good SNR (much noise removed) but poor Fidelity (shape distorted), or vice versa.

### Key references
- Pearson, K. (1895). *Notes on regression and inheritance in the case of two parents.* Proceedings of the Royal Society of London. — original derivation of the correlation coefficient.
- Benesty, J., Chen, J., Huang, Y., & Cohen, I. (2009). *Pearson Correlation Coefficient.* In: *Noise Reduction in Speech Processing.* Springer. — practical treatment of Pearson r as a signal fidelity measure in filtering contexts, directly analogous to this use case.
- Spiess, A. N., et al. (2015). *Impact of smoothing on parameter estimation in quantitative DNA amplification experiments.* Clinical Chemistry. — evaluates how SG and moving average filters distort kinetic curve shape (TTP, slope), using correlation-based fidelity assessment; highly relevant to this codebase.

---

## 3. Residual Autocorr Lag-1 — AC lag-1

### What it is
Residual autocorrelation at lag 1 tests whether the *noise removed by the filter* is truly random (white noise), or whether it still contains structured signal components that the filter failed to separate. It answers: *is what we removed actually noise, or did the filter accidentally extract real signal?*

### Background: white noise assumption
A well-designed denoising filter should leave behind residuals $e_t = x_t - \hat{x}_t$ that behave like white noise — i.e., identically distributed and *uncorrelated* across time. If residuals are autocorrelated at lag 1 (i.e., $e_t$ predicts $e_{t+1}$), it means the filter failed to fully separate signal from noise: the "noise" still has temporal structure, which is a hallmark of signal.

### Formula

$$\text{AC lag-1} = \frac{1}{N} \sum_{i=1}^{N} \text{corr}(e_{i,1:T-1},\; e_{i,2:T})$$

where $e_i = x_i - \hat{x}_i$ is the mean-centred residual for sample $i$, and `corr` is Pearson r between consecutive time steps.

**In code:**
```python
def ac1(raw, denoised):
    res = raw - denoised
    return np.nanmean([
        np.corrcoef(
            (e := res[i] - res[i].mean())[:-1],   # e_t
            e[1:]                                   # e_{t+1}
        )[0, 1]
        for i in range(len(res))
        if res[i].std() > 0  # skip zero-residual samples
    ])
```

This is the sample-mean Pearson autocorrelation at lag 1, computed on the mean-centred residuals of each curve independently.

### Interpretation
| Value | Meaning |
|---|---|
| ≈ 0 | Residuals are white noise — the filter cleanly separated signal from noise |
| > 0 (positive) | Residuals are positively autocorrelated — the filter is "following" noise bumps or failed to extract all signal (under-smoothing) |
| < 0 (negative) | Residuals are negatively autocorrelated — the filter is oscillating around the signal (ringing artefact, common in FFT/wavelet with too-aggressive thresholding) |

**Direction: ↓ lower (closer to 0) is better.**

A small positive value is normal and expected for real biological signals — the noise in fluorescence data is not purely i.i.d. (photon shot noise, thermal drift). The key diagnostic is: does the filter *reduce* AC lag-1 compared to the raw signal, and does the residual AC stay below the raw signal's own autocorrelation?

### Connection to classical residual diagnostics
This metric is closely related to two classical statistics:

**Durbin-Watson statistic (d):**
$$d = \frac{\sum_{t=2}^{T} (e_t - e_{t-1})^2}{\sum_{t=1}^{T} e_t^2} \approx 2(1 - r_1)$$

where $r_1$ is the lag-1 autocorrelation. So $d \approx 2$ (no serial correlation) ↔ AC lag-1 ≈ 0; $d < 2$ ↔ positive AC; $d > 2$ ↔ negative AC. The two formulations are essentially equivalent; the notebook uses the correlation form directly.

**Ljung-Box Q statistic:**
$$Q(m) = T(T+2) \sum_{k=1}^{m} \frac{r_k^2}{T-k}$$

Tests the joint null hypothesis that all lags 1 through $m$ are zero. AC lag-1 tests only the first lag, which is both the most sensitive and most interpretable for smooth kinetic curves.

### Why it matters here
For kinetic fluorescence curves:
- **Under-smooth** (e.g. moving average with a narrow window): the residuals still show the curve's rise phase → large positive AC lag-1
- **Good filter**: residuals look like measurement noise → AC lag-1 near 0
- **Over-smooth / ringing** (e.g. FFT with too-low cutoff): residuals contain negative-autocorrelated ringing → negative AC lag-1

This makes AC lag-1 a sensitive *qualitative* diagnostic: it distinguishes under-fitting, good fit, and over-fitting in a single number, whereas SNR and Fidelity can be ambiguous in those three regimes.

### Key references
- Durbin, J., & Watson, G. S. (1950). *Testing for Serial Correlation in Least Squares Regression: I.* Biometrika, 37(3/4), 409–428. — introduced the DW statistic; the foundational paper for lag-1 residual autocorrelation testing.
- Durbin, J., & Watson, G. S. (1951). *Testing for Serial Correlation in Least Squares Regression: II.* Biometrika, 38(1/2), 159–178. — extended the DW test with distributional results.
- Ljung, G. M., & Box, G. E. P. (1978). *On a measure of lack of fit in time series models.* Biometrika, 65(2), 297–303. — the Ljung-Box test; the modern standard for testing white-noise residuals across multiple lags simultaneously.
- Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control.* Wiley. — comprehensive textbook treatment of residual autocorrelation, ACF/PACF, and white-noise testing.
- Spiess, A. N., et al. (2015). *Impact of smoothing on parameter estimation in quantitative DNA amplification experiments.* Clinical Chemistry. — domain-specific reference; shows how residual structure (including serial correlation) reflects filter quality for sigmoid-shaped kinetic curves.

---

## Summary Table

| Metric | Formula | Direction | What fails | What passes |
|---|---|---|---|---|
| **SNR (dB)** | $10\log_{10}(\text{Var}(\hat{x}) / \text{Var}(x-\hat{x}))$ | ↑ higher | Over-smooth or noisy output | Much signal power, little noise power |
| **Fidelity (corr)** | $\rho(x_i, \hat{x}_i)$, mean across samples | ↑ higher (→ 1.0) | Peak/shape distortion, amplitude scaling issues | Denoised curve tracks raw trajectory faithfully |
| **Residual AC lag-1** | $\text{corr}(e_{t}, e_{t+1})$, mean across samples | ↓ lower (→ 0) | Under-smooth (positive), ringing (negative) | Residuals are white noise — clean separation |

### Complementary reading
The three metrics cover different failure modes and should be read together:

- **High SNR + low Fidelity** → filter removed genuine signal dynamics (distortion)
- **High Fidelity + high AC lag-1** → filter follows the noisy trajectory faithfully but hasn't removed the noise structure
- **High SNR + high Fidelity + AC ≈ 0** → good denoising: clean, shape-preserving, and the removed component is structureless noise

### Broader signal processing references
- Wiener, N. (1949). *Extrapolation, Interpolation, and Smoothing of Stationary Time Series.* MIT Press. — the theoretical foundation for optimal linear filtering; SNR and fidelity tradeoffs are derived here.
- Donoho, D. L., & Johnstone, I. M. (1994). *Ideal spatial adaptation by wavelet shrinkage.* Biometrika, 81(3), 425–455. — the Donoho-Johnstone universal threshold (used for the wavelet variants in this pipeline); derived under the white-noise residual assumption that AC lag-1 tests.
- Oppenheim, A. V., & Schafer, R. W. (2009). *Discrete-Time Signal Processing* (3rd ed.). Prentice Hall. — foundational DSP textbook covering SNR, filter frequency response, and noise characterisation.
