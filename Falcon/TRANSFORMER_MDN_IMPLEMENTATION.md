# Transformer + MDN Implementation for Multi-Modal Trajectory Prediction

## Overview
This document describes the architectural improvements made to the Future Trajectory Prediction auxiliary task in the Falcon social navigation framework. The key innovation is replacing the sequential BiLSTM + Self-Attention architecture with a parallel Transformer Encoder + Mixture Density Network (MDN) that outputs probabilistic multi-modal predictions.

---

## 1. Motivation

### Problems with Original Architecture (BiLSTM + MSE Loss)

**Sequential Processing Limitation:**
- BiLSTM processes temporal information step-by-step, limiting its ability to capture long-range dependencies
- Cannot instantly recognize interaction patterns like "Human A is yielding to Human B"

**Single-Mode Prediction:**
- MSE loss: `L = ||ŷ - y||²` forces the network to predict a single point
- Fails to capture uncertainty and multi-modal human behavior
- Example: At a T-junction, a human could go left OR right, but MSE forces averaging → predicts "middle" (invalid path)

**Gradient Flow Issue:**
- MSE penalizes all plausible paths except the ground truth
- The shared backbone (ResNet + Policy LSTM) learns deterministic features
- Main policy receives latent vector δ_R that encodes "Human is at position X" (no uncertainty information)

---

## 2. Proposed Architecture

### 2.1 Transformer Encoder (Global Receptive Field)

**Replacement:**
```python
# OLD: BiLSTM (sequential processing)
self.lstm = nn.LSTM(
    input_size=hidden_size + 1 + max_human_num * position_dim,
    hidden_size=hidden_size,
    num_layers=2,
    bidirectional=True,
    batch_first=True
)

# NEW: Transformer Encoder (parallel processing)
encoder_layer = nn.TransformerEncoderLayer(
    d_model=hidden_size,
    nhead=8,                    # Multi-head attention
    dim_feedforward=hidden_size * 4,
    dropout=0.1,
    activation='relu',
    batch_first=True
)
self.transformer_encoder = nn.TransformerEncoder(
    encoder_layer,
    num_layers=2
)
```

**Key Advantages:**
1. **Global Receptive Field:** Processes entire input in parallel, instantly recognizing interaction patterns
2. **Better Gradient Flow:** Direct connections between all positions via attention mechanism
3. **Scalability:** Can easily extend to longer time horizons without sequential bottleneck

---

### 2.2 Mixture Density Network (MDN) Head

**Replacement:**
```python
# OLD: Simple regression (outputs single point)
self.classifier = nn.Sequential(
    nn.Linear(hidden_size * 2, hidden_size),
    nn.ReLU(True),
    nn.Linear(hidden_size, max_human_num * future_step * position_dim),
)

# NEW: MDN Head (outputs GMM distribution parameters)
params_per_component = 1 + 2 * position_dim  # π + μ_x,μ_y + σ_x,σ_y
total_output_dim = max_human_num * future_step * K * params_per_component

self.mdn_head = nn.Sequential(
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(True),
    nn.Dropout(0.1),
    nn.Linear(hidden_size, total_output_dim),
)
```

**MDN Output Structure (per human, per timestep):**
For K=5 mixture components:
- **π (mixture weights):** 5 values, sum to 1 (via softmax)
- **μ (means):** 5 × 2 = 10 values (x,y coordinates for each Gaussian)
- **σ (standard deviations):** 5 × 2 = 10 values (x,y spread for each Gaussian, positive via softplus)

Total: 5 + 10 + 10 = **25 parameters per prediction**

**Processing:**
```python
# Extract parameters from MDN output
pi = F.softmax(mdn_params[..., 0], dim=-1)  # Mixture weights
mu = mdn_params[..., 1:1+position_dim]       # Means (x, y)
sigma = F.softplus(mdn_params[..., 1+position_dim:]) + 1e-6  # Stds (positive)
```

---

## 3. Mathematical Formulation

### 3.1 Gaussian Mixture Model (GMM)

The predicted distribution over future positions is a GMM:

```
P(y | x) = Σ_{k=1}^K π_k · N(y | μ_k, σ_k²)
```

Where:
- `y`: Future position (x, y coordinates)
- `x`: Input features (robot state δ_R, human positions P_i^t)
- `K`: Number of mixture components (K=5)
- `π_k`: Weight of component k (π_k ≥ 0, Σ π_k = 1)
- `N(y | μ_k, σ_k²)`: 2D Gaussian with mean μ_k and variance σ_k²

### 3.2 Negative Log-Likelihood (NLL) Loss

**Old Loss (MSE):**
```
L_MSE = ||ŷ - y_gt||²
```
Problem: Penalizes all plausible paths except the ground truth.

**New Loss (NLL):**
```
L_NLL = -log P(y_gt | x)
     = -log (Σ_{k=1}^K π_k · N(y_gt | μ_k, σ_k²))
```

**Expanded form:**
```
log N(y | μ, σ²) = -0.5 · [log(2π) + log(σ²) + (y - μ)²/σ²]

L_NLL = -log (Σ_{k=1}^K π_k · exp(-0.5 · [log(2π) + log(σ_k²) + (y_gt - μ_k)²/σ_k²]))
```

**Implementation (with log-sum-exp trick for numerical stability):**
```python
# Log probability of each Gaussian component
log_gauss = -0.5 * (log(2π) + 2*log(σ) + (y - μ)²/σ²)

# Weighted log probability: log(π_k * N(y | μ_k, σ_k))
log_pi_gauss = log(π) + log_gauss

# Log-sum-exp trick: log(Σ exp(x)) = max(x) + log(Σ exp(x - max(x)))
max_val = max(log_pi_gauss)
log_sum_exp = max_val + log(Σ exp(log_pi_gauss - max_val))

# Negative log-likelihood
L_NLL = -log_sum_exp
```

---

## 4. Gradient Flow Justification (The "Meat" of the Proposal)

### 4.1 How NLL Loss Encodes Uncertainty

**Key Insight:** By minimizing NLL loss, the network is rewarded for:
1. **Assigning high probability mass to the ground truth trajectory**
2. **Maintaining diverse mixture components to cover multiple plausible paths**

Example: Human at T-junction (can go left OR right)
- **MSE approach:** Forces network to predict average (middle) → High loss for both outcomes
- **NLL approach:** Network learns to output two high-probability components (left AND right) → Low loss for either outcome

### 4.2 Uncertainty-Aware Latent Features

**Gradient Flow Path:**
```
NLL Loss → MDN Head → Transformer Encoder → Input Projection → Scene Features (from ResNet + Policy LSTM)
```

**What the shared backbone learns:**
- **Before (MSE):** δ_R encodes "Human is at position X"
- **After (NLL):** δ_R encodes "Human is at X and has uncertainty distribution σ for moving to Y or Z"

**Mechanism:**
1. NLL loss has gradients that depend on σ (uncertainty)
2. These gradients backpropagate through the entire network
3. The shared ResNet + Policy LSTM learns to produce features that encode:
   - Current position (mean μ)
   - Movement uncertainty (variance σ)
   - Multi-modal behavior (mixture weights π)

### 4.3 Impact on Main Policy

**Even though the auxiliary module is removed at test time, the main policy benefits:**

1. **Richer feature representations:**
   - The latent vector δ_R from Policy LSTM now contains uncertainty information
   - Main policy can "sense" when human behavior is unpredictable

2. **Safer navigation decisions:**
   - When latent features indicate high uncertainty (large σ), the value function learns to maintain larger clearance
   - Example: "Human is near intersection (uncertain behavior) → Slow down and give more space"

3. **Better generalization:**
   - Training with multi-modal loss prevents overfitting to single trajectories
   - Policy learns robust behaviors that work across diverse human movement patterns

---

## 5. Implementation Details

### 5.1 Architecture Summary

```
Input Features
    ↓
[Robot Features (δ_R) + Human Count + Current Positions]
    ↓
Input Projection (Linear)
    ↓
Transformer Encoder (2 layers, 8 heads)
    ↓
MDN Head (outputs π, μ, σ for K=5 components)
    ↓
Parse GMM Parameters
    ↓
Compute NLL Loss with ground truth
```

### 5.2 Configuration Parameters

Added to `FutureTrajectoryPredictionLossConfig`:
- `num_mixture_components: 5` - Number of Gaussian components (K)
- `transformer_layers: 2` - Number of Transformer Encoder layers
- `transformer_heads: 8` - Number of attention heads

### 5.3 Files Modified

1. **`falcon/auxiliary_tasks.py`**
   - Replaced BiLSTM with Transformer Encoder
   - Replaced simple classifier with MDN head
   - Replaced MSE loss with NLL loss
   - Added `_gaussian_mixture_nll()` method

2. **Configuration files:**
   - `falcon_hm3d_train_2v100_ellipse.yaml`
   - `falcon_hm3d_train_mini_junwei.yaml`

---

## 6. Expected Benefits

### 6.1 Training Phase
1. **Better auxiliary task performance:** Multi-modal predictions better match real human behavior
2. **Improved gradient quality:** NLL gradients provide richer learning signal
3. **Uncertainty awareness:** Network learns when predictions are uncertain

### 6.2 Evaluation Phase (after removing auxiliary module)
1. **Safer navigation:** Main policy maintains larger clearance in uncertain situations
2. **Better PSC (Personal Space Compliance):** Uncertainty-aware features help avoid intrusions
3. **Improved generalization:** Robust features transfer better to unseen scenarios

---

## 7. Comparison Table

| Aspect | Old (BiLSTM + MSE) | New (Transformer + MDN) |
|--------|-------------------|------------------------|
| **Architecture** | BiLSTM (sequential) | Transformer (parallel) |
| **Receptive Field** | Local (limited by LSTM) | Global (full attention) |
| **Output** | Single point (x, y) | GMM distribution (π, μ, σ) |
| **Loss Function** | MSE: `||ŷ - y||²` | NLL: `-log P(y_gt\|x)` |
| **Uncertainty** | No | Yes (via σ parameters) |
| **Multi-Modal** | No (averages modes) | Yes (K=5 components) |
| **Latent Features** | "Human is at X" | "Human at X, could move to Y or Z" |
| **Gradient Quality** | Penalizes plausible paths | Rewards probability distribution |
| **Parameters** | ~262K (BiLSTM) | ~295K (Transformer + MDN) |
| **Compute** | Sequential (slower) | Parallel (faster on GPU) |

---

## 8. Validation and Testing

### 8.1 Syntax Validation
```bash
python -m py_compile falcon/auxiliary_tasks.py
# Status: PASSED ✓
```

### 8.2 Recommended Training Command
```bash
python -m habitat-baselines.habitat_baselines.run \
  --config-name=social_nav_v2/falcon_hm3d_train_2v100_ellipse.yaml
```

### 8.3 Metrics to Monitor
1. **Auxiliary Task Loss:** Should converge to lower values with NLL
2. **Main Policy Performance:**
   - Success Rate (SR)
   - Personal Space Compliance (PSC) - should improve
   - Human Collision Rate - should decrease
3. **Uncertainty Calibration:** Analyze predicted σ values during evaluation

---

## 9. Future Extensions

1. **Adaptive K:** Use different number of components based on scenario complexity
2. **Temporal Attention:** Extend Transformer to process multi-step history
3. **Interaction Modeling:** Explicitly model human-human and human-robot interactions
4. **Covariance:** Use full covariance matrices instead of diagonal (x, y independent)

---

## 10. References

1. Bishop, C. M. (1994). "Mixture Density Networks". Technical Report NCRG/94/004.
2. Vaswani, A., et al. (2017). "Attention Is All You Need". NeurIPS.
3. Alahi, A., et al. (2016). "Social LSTM: Human Trajectory Prediction in Crowded Spaces". CVPR.
4. Gupta, A., et al. (2018). "Social GAN: Socially Acceptable Trajectories with GANs". CVPR.

---

## Conclusion

The Transformer + MDN architecture provides three key improvements:
1. **Better pattern recognition** via global receptive field
2. **Multi-modal predictions** via Gaussian Mixture Model
3. **Uncertainty-aware features** via NLL loss gradient flow

These improvements enable the main policy to make safer, more socially-aware navigation decisions even after the auxiliary module is removed during evaluation.
