# Two-Phase Curriculum Training Integration Guide

## Overview
This document explains how to integrate the two-phase curriculum training approach with the Falcon trainer to prevent "gradient shock" from the untrained Transformer-MDN disrupting the pretrained navigation backbone.

## Curriculum Phases

### Phase 1: Warm-Start (0 - 100k steps)
**Objective:** Quickly train the Transformer-MDN module to a stable baseline without disrupting pretrained weights.

**Configuration:**
- Freeze shared backbone (ResNet + Policy LSTM)
- Train only Transformer-MDN module
- High auxiliary loss weight: `loss_scale × warmstart_aux_lr_multiplier` (default: 0.1 × 5.0 = 0.5)
- Duration: 100k steps (configurable via `warmstart_steps`)

**Implementation:**
```python
# In trainer, before update step:
if self.num_steps_done < self.config.warmstart_steps:
    # Freeze backbone parameters
    for param in self.actor_critic.net.parameters():
        param.requires_grad = False

    # Ensure auxiliary modules are trainable
    for aux_module in self.actor_critic.aux_loss_modules.values():
        for param in aux_module.parameters():
            param.requires_grad = True
```

### Phase 2: Joint Finetuning (100k+ steps)
**Objective:** Finetune backbone and Transformer-MDN together with balanced gradient magnitudes.

**Configuration:**
- Unfreeze shared backbone
- Train all modules jointly
- Normal auxiliary loss weight: `loss_scale` (default: 0.1)
- Optional: Normalize loss weights based on gradient magnitudes

**Implementation:**
```python
# In trainer, before update step:
if self.num_steps_done >= self.config.warmstart_steps:
    # Unfreeze all parameters
    for param in self.actor_critic.parameters():
        param.requires_grad = True
```

---

## Implementation Details

### 1. Automatic Step Tracking

The auxiliary module needs to know the current training step to apply curriculum logic. This is handled via the `set_step()` method:

```python
# File: falcon/auxiliary_tasks.py

class FutureTrajectoryPrediction(nn.Module):
    def set_step(self, step: int):
        """Update current training step for curriculum scheduling."""
        self._current_step = step

    def get_curriculum_loss_scale(self) -> float:
        """Get the current loss scale based on curriculum phase."""
        if not self.use_curriculum:
            return self.loss_scale

        if self._current_step < self.warmstart_steps:
            # Phase 1: High LR multiplier for faster convergence
            return self.loss_scale * self.warmstart_aux_lr_multiplier
        else:
            # Phase 2: Normal weight for balanced training
            return self.loss_scale
```

### 2. Trainer Integration

The trainer should call `set_step()` before each forward pass to ensure the auxiliary module knows the current training progress:

```python
# File: habitat-baselines/habitat_baselines/rl/ppo/falcon_trainer.py

@profiling_wrapper.RangeContext("_update_agent")
def _update_agent(self):
    # Update curriculum step counter for all auxiliary modules
    for aux_module in self.actor_critic.aux_loss_modules.values():
        if hasattr(aux_module, 'set_step'):
            aux_module.set_step(self.num_steps_done)

    # Apply curriculum-based parameter freezing/unfreezing
    self._apply_curriculum_freezing()

    # ... rest of update logic
```

### 3. Curriculum Freezing Helper

Add a helper method to the trainer to handle parameter freezing:

```python
def _apply_curriculum_freezing(self):
    """
    Apply two-phase curriculum: freeze/unfreeze backbone based on training progress.

    This prevents gradient shock from untrained Transformer-MDN disrupting pretrained weights.
    """
    # Check if any auxiliary module uses curriculum
    use_curriculum = False
    warmstart_steps = 0

    for aux_module in self.actor_critic.aux_loss_modules.values():
        if hasattr(aux_module, 'use_curriculum') and aux_module.use_curriculum:
            use_curriculum = True
            warmstart_steps = max(warmstart_steps, aux_module.warmstart_steps)

    if not use_curriculum:
        return  # No curriculum training, all parameters trainable

    if self.num_steps_done < warmstart_steps:
        # Phase 1: Freeze backbone, train only auxiliary modules
        if not hasattr(self, '_backbone_frozen') or not self._backbone_frozen:
            print(f"[Curriculum] Phase 1: Freezing backbone (step {self.num_steps_done}/{warmstart_steps})")

            # Freeze ResNet backbone
            if hasattr(self.actor_critic.net, 'visual_encoder'):
                for param in self.actor_critic.net.visual_encoder.parameters():
                    param.requires_grad = False

            # Freeze Policy LSTM
            if hasattr(self.actor_critic.net, 'state_encoder'):
                for param in self.actor_critic.net.state_encoder.parameters():
                    param.requires_grad = False

            # Freeze critic and action distribution heads
            for param in self.actor_critic.critic.parameters():
                param.requires_grad = False
            for param in self.actor_critic.action_distribution.parameters():
                param.requires_grad = False

            self._backbone_frozen = True

    else:
        # Phase 2: Unfreeze all for joint finetuning
        if not hasattr(self, '_backbone_frozen') or self._backbone_frozen:
            print(f"[Curriculum] Phase 2: Unfreezing backbone (step {self.num_steps_done}/{warmstart_steps})")

            # Unfreeze all parameters
            for param in self.actor_critic.parameters():
                param.requires_grad = True

            self._backbone_frozen = False
```

---

## Configuration Parameters

### In Training Config (`falcon_hm3d_train_2v100_transformer_mdn.yaml`)

```yaml
habitat_baselines:
  rl:
    auxiliary_losses:
      future_trajectory_prediction:
        max_human_num: 6
        future_step: 4
        loss_scale: 0.1

        # Transformer + MDN parameters
        num_mixture_components: 5
        transformer_layers: 2
        transformer_heads: 8

        # Two-Phase Curriculum parameters
        use_curriculum: True                    # Enable curriculum training
        warmstart_steps: 100000                 # Phase 1 duration (100k steps)
        warmstart_aux_lr_multiplier: 5.0        # Phase 1 high LR multiplier
        finetune_loss_weight_normalize: True    # Phase 2 gradient normalization
```

---

## Expected Training Behavior

### Phase 1 (Steps 0 - 100k)

**TensorBoard Metrics:**
- `aux_loss/future_trajectory_prediction`: **High initially, should drop rapidly**
- `ppo/policy_loss`: **Frozen (no updates to main policy)**
- `ppo/value_loss`: **Frozen (no updates to value function)**
- `rewards/mean`: **Roughly constant (using pretrained policy)**

**Console Output:**
```
[Curriculum] Phase 1: Freezing backbone (step 0/100000)
Step 1000: aux_loss=2.45, policy_loss=FROZEN, reward=5.2
Step 10000: aux_loss=1.12, policy_loss=FROZEN, reward=5.3
Step 50000: aux_loss=0.68, policy_loss=FROZEN, reward=5.1
Step 100000: aux_loss=0.52, policy_loss=FROZEN, reward=5.2
```

### Phase 2 (Steps 100k - 1M)

**TensorBoard Metrics:**
- `aux_loss/future_trajectory_prediction`: **Stable around converged value**
- `ppo/policy_loss`: **Active updates (improving main policy)**
- `ppo/value_loss`: **Active updates**
- `rewards/mean`: **Gradually improving**

**Console Output:**
```
[Curriculum] Phase 2: Unfreezing backbone (step 100000/100000)
Step 110000: aux_loss=0.51, policy_loss=0.32, reward=5.6
Step 200000: aux_loss=0.48, policy_loss=0.28, reward=6.3
Step 500000: aux_loss=0.45, policy_loss=0.24, reward=7.8
Step 1000000: aux_loss=0.43, policy_loss=0.21, reward=9.2
```

---

## Gradient Magnitude Normalization (Advanced)

For Phase 2, you can optionally normalize loss weights to ensure balanced gradient magnitudes between PPO and auxiliary objectives:

```python
def _get_normalized_loss_weights(self, ppo_loss, aux_losses):
    """
    Normalize auxiliary loss weights based on gradient magnitudes.

    Ensures that auxiliary gradients don't dominate PPO gradients.
    """
    if not self.config.finetune_loss_weight_normalize:
        return aux_losses  # No normalization

    # Compute gradient norms
    ppo_loss.backward(retain_graph=True)
    ppo_grad_norm = 0.0
    for param in self.actor_critic.net.parameters():
        if param.grad is not None:
            ppo_grad_norm += param.grad.norm().item() ** 2
    ppo_grad_norm = ppo_grad_norm ** 0.5

    # Zero gradients before aux loss
    self.optimizer.zero_grad()

    # Compute aux gradient norms and normalize
    normalized_aux_losses = {}
    for name, aux_loss in aux_losses.items():
        aux_loss.backward(retain_graph=True)

        aux_grad_norm = 0.0
        for param in self.actor_critic.aux_loss_modules[name].parameters():
            if param.grad is not None:
                aux_grad_norm += param.grad.norm().item() ** 2
        aux_grad_norm = aux_grad_norm ** 0.5

        # Normalize: scale aux loss so its gradient norm matches PPO
        if aux_grad_norm > 1e-8:
            scale = ppo_grad_norm / aux_grad_norm
            normalized_aux_losses[name] = aux_loss * scale
        else:
            normalized_aux_losses[name] = aux_loss

        self.optimizer.zero_grad()

    return normalized_aux_losses
```

---

## Testing Curriculum Training

### 1. Verify Curriculum Activation

```bash
# Train with Transformer-MDN config
python -m habitat-baselines.habitat_baselines.run \
  --config-name=social_nav_v2/falcon_hm3d_train_2v100_transformer_mdn.yaml

# Check console for curriculum messages:
# "Phase 1: Freezing backbone" should appear at step 0
# "Phase 2: Unfreezing backbone" should appear at step 100000
```

### 2. Monitor TensorBoard

```bash
tensorboard --logdir=evaluation/falcon/transformer_mdn_hm3d_2v100/tb

# Watch for:
# - Rapid aux_loss drop in Phase 1 (0-100k)
# - Stable aux_loss + improving rewards in Phase 2 (100k+)
```

### 3. Inspect Checkpoint

```python
import torch

# Load checkpoint from Phase 1
ckpt = torch.load('evaluation/falcon/transformer_mdn_hm3d_2v100/checkpoints/ckpt.50.pth')

# Verify Transformer-MDN parameters changed (trained)
transformer_params = [k for k in ckpt['state_dict'].keys() if 'transformer' in k or 'mdn' in k]
print(f"Transformer-MDN parameters: {len(transformer_params)}")

# Verify backbone parameters unchanged (frozen)
# Compare with pretrained weights
pretrained = torch.load('pretrained_model/pretrained_habitat3.pth')
for key in pretrained['state_dict'].keys():
    if 'visual_encoder' in key or 'state_encoder' in key:
        diff = (ckpt['state_dict'][key] - pretrained['state_dict'][key]).abs().sum()
        print(f"{key}: diff = {diff.item()}")  # Should be ~0 in Phase 1
```

---

## Summary

The two-phase curriculum training approach ensures:

1. **Phase 1 (Warm-Start):**
   - Transformer-MDN quickly converges without disrupting pretrained weights
   - High auxiliary loss weight accelerates learning
   - Backbone frozen prevents gradient shock

2. **Phase 2 (Joint Finetuning):**
   - All modules finetune together with balanced gradients
   - Backbone learns uncertainty-aware features from NLL loss
   - Main policy benefits from improved latent representations

This curriculum prevents the common problem where untrained auxiliary modules inject large, random gradients into pretrained backbones, destroying learned features and degrading performance.
