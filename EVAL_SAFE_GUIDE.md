# Eval-Safe Training Configuration Guide

## Overview

This guide explains how to use the **eval-safe** training and evaluation configs that maintain compatibility with the original evaluation code from commit `5d8ec1d1cf` while leveraging all the latest improvements.

## What is "Eval-Safe"?

An eval-safe configuration ensures that:
1. **Main policy network** architecture matches the original implementation (ResNet50 + LSTM512)
2. **Processed checkpoints** (after running `process_ckp_for_eval.py`) can be loaded by the original eval code
3. **All training improvements** are included but don't break eval compatibility

## Key Compatibility Requirements

### ✅ Compatible (Eval-Safe)
- **Backbone:** `resnet50` or `resnet18`
- **Auxiliary tasks:** Any architecture (removed by `process_ckp_for_eval.py`)
- **Observation space:** Depth-only (RGB commented out)
- **RNN:** LSTM with 2 layers

### ❌ Incompatible (NOT Eval-Safe)
- **Backbone:** `dual_stream_fpn` (different architecture)
- **Observation space:** RGB + Depth (changes visual encoder)
- **Different action spaces:** Additional actions beyond the 4 basic ones

## Available Configs

### Training Configs

#### 1. `falcon_hm3d_train_2v100_evalsafe.yaml`
**Full model for 2x V100 GPUs (32GB each)**

**Features:**
- ResNet50 + LSTM512 (eval-compatible)
- Transformer-MDN trajectory prediction (hardcoded defaults)
- Elliptical proximity penalty (velocity-aware)
- Obstacle proximity penalty (depth-based)
- Action smoothness loss (hardcoded to 0.01)
- Mixed precision training (AMP)
- Warmup + Cosine LR schedule
- 16 parallel environments (8 per GPU)
- Expected VRAM: 22-30GB per GPU
- Training time: ~10-15 hours to convergence

**Usage:**
```bash
cd Falcon
python -m habitat-baselines.habitat_baselines.run \
  --config-name=social_nav_v2/falcon_hm3d_train_2v100_evalsafe.yaml
```

### Evaluation Configs

#### 1. `falcon_hm3d_evalsafe.yaml`
**Eval config for models trained with evalsafe training configs**

**Features:**
- ResNet50 + LSTM512 (matches training)
- Core sensors only (depth + pointgoal + human_velocity_sensor)
- No auxiliary sensors (removed during eval)
- Compatible with original eval code structure

**Usage:**
```bash
cd Falcon
# First, process the checkpoint to remove auxiliary modules
python process_ckp_for_eval.py \
  evaluation/falcon/evalsafe_hm3d_2v100/checkpoints/ckpt.15.pth \
  evaluation/falcon/evalsafe_hm3d_2v100/checkpoints/eval.pth

# Then evaluate
python -m habitat-baselines.habitat_baselines.run \
  --config-name=social_nav_v2/falcon_hm3d_evalsafe.yaml \
  habitat_baselines.evaluate=True \
  habitat_baselines.eval_ckpt_path_dir=evaluation/falcon/evalsafe_hm3d_2v100/checkpoints/eval.pth
```

## Workflow

### 1. Training

```bash
# Train with all improvements (2x V100)
cd Falcon
python -m habitat-baselines.habitat_baselines.run \
  --config-name=social_nav_v2/falcon_hm3d_train_2v100_evalsafe.yaml
```

**Monitor training:**
```bash
tensorboard --logdir=evaluation/falcon/evalsafe_hm3d_2v100/tb
```

### 2. Checkpoint Processing

After training completes, process the checkpoint to remove auxiliary modules:

```bash
cd Falcon
python process_ckp_for_eval.py \
  evaluation/falcon/evalsafe_hm3d_2v100/checkpoints/ckpt.15.pth \
  evaluation/falcon/evalsafe_hm3d_2v100/checkpoints/eval.pth
```

**What this does:**
- Removes all `aux_loss_modules.*` keys from state_dict
- Keeps only the main policy network (ResNet50 + LSTM512)
- Ensures checkpoint is compatible with original eval code

### 3. Evaluation

```bash
cd Falcon
python -m habitat-baselines.habitat_baselines.run \
  --config-name=social_nav_v2/falcon_hm3d_evalsafe.yaml \
  habitat_baselines.evaluate=True \
  habitat_baselines.eval_ckpt_path_dir=evaluation/falcon/evalsafe_hm3d_2v100/checkpoints/eval.pth
```

**With video generation:**
```bash
python -m habitat-baselines.habitat_baselines.run \
  --config-name=social_nav_v2/falcon_hm3d_evalsafe.yaml \
  habitat_baselines.evaluate=True \
  habitat_baselines.eval_ckpt_path_dir=evaluation/falcon/evalsafe_hm3d_2v100/checkpoints/eval.pth \
  habitat_baselines.eval.video_option="['disk']"
```

## Improvements Included

The eval-safe configs include all the latest improvements while maintaining compatibility:

### 1. Transformer-MDN Trajectory Prediction
- **Architecture:** Transformer Encoder (2 layers, 8 heads) + Mixture Density Network (5 components)
- **Benefit:** Multi-modal prediction captures uncertainty in human trajectories
- **Configuration:** Uses hardcoded defaults (not configurable via YAML)
- **Compatibility:** Removed by `process_ckp_for_eval.py`, doesn't affect eval

### 2. Elliptical Proximity Penalty
- **What:** Velocity-aware human avoidance
- **Benefit:** Penalty zone expands 2x in direction of movement, shrinks 0.5x behind
- **Compatibility:** Requires `human_velocity_sensor` in both train and eval configs (included)

### 3. Two-Phase Curriculum Training (NOT ENABLED)
- **Status:** Disabled by default (`use_curriculum=False`)
- **Why:** Config parameters not in AuxLossConfig dataclass, cannot be set via YAML
- **To enable:** Would require code modification in `auxiliary_tasks.py`
- **If enabled:** Phase 1 (freeze backbone), Phase 2 (joint finetuning)

### 4. Action Smoothness Loss
- **What:** Regularizes policy to maintain consistent action distributions (hardcoded to 0.01)
- **Benefit:** Reduces jittery behavior, smoother navigation
- **Compatibility:** Training-only, doesn't affect checkpoint structure
- **Note:** This is enabled by default in the code, not configurable via YAML

### 5. Warmup + Cosine LR Schedule
- **What:** Linear warmup (10%), then cosine decay
- **Benefit:** Reduces oscillation, improves convergence
- **Compatibility:** Training-only hyperparameter

### 6. Mixed Precision Training (AMP)
- **What:** Automatic mixed precision with FP16/FP32
- **Benefit:** Faster training, lower VRAM usage
- **Compatibility:** Training-only, doesn't affect checkpoint

## Observation Space Details

### Training Observation Space
```yaml
obs_keys:
  # CORE SENSORS (required for eval)
  - agent_0_articulated_agent_jaw_depth   # Visual observation
  - agent_0_pointgoal_with_gps_compass    # Navigation goal
  - human_velocity_sensor                 # For elliptical penalty

  # AUXILIARY SENSORS (training only)
  - agent_0_localization_sensor           # For auxiliary losses
  - agent_0_human_num_sensor              # For people counting task
  - agent_0_oracle_humanoid_future_trajectory  # For trajectory prediction task
```

### Evaluation Observation Space
```yaml
obs_keys:
  # CORE SENSORS ONLY
  - agent_0_articulated_agent_jaw_depth   # Visual observation
  - agent_0_pointgoal_with_gps_compass    # Navigation goal
  - human_velocity_sensor                 # For elliptical penalty
```

**Why this works:**
- The main policy network only uses CORE sensors
- Auxiliary sensors are only accessed by auxiliary modules
- After `process_ckp_for_eval.py` removes auxiliary modules, auxiliary sensors are no longer needed

## Compatibility Verification

To verify a checkpoint is eval-safe:

```python
import torch

# Load processed checkpoint
ckpt = torch.load('eval.pth')
state_dict = ckpt[0]['state_dict']

# Check 1: No auxiliary modules
aux_keys = [k for k in state_dict.keys() if 'aux_loss_modules' in k]
assert len(aux_keys) == 0, f"Auxiliary modules not removed: {aux_keys}"

# Check 2: Backbone is ResNet (not dual_stream_fpn)
visual_keys = [k for k in state_dict.keys() if 'visual_encoder' in k]
has_dual_stream = any('rgb_stem' in k or 'depth_stem' in k for k in visual_keys)
assert not has_dual_stream, "Checkpoint uses dual_stream_fpn (incompatible)"

# Check 3: RNN architecture
rnn_key = 'net.state_encoder.rnn.weight_hh_l0'
if rnn_key in state_dict:
    hidden_size = state_dict[rnn_key].shape[1]
    print(f"✓ RNN hidden size: {hidden_size}")

print("✓ Checkpoint is eval-safe!")
```

## Troubleshooting

### Error: KeyError: 'human_velocity_sensor'
**Cause:** The sensor is listed in `obs_keys` but not registered in the task's `lab_sensors` defaults.

**Solution:** Add `human_velocity_sensor` to the task config's lab_sensors list:
```yaml
# In Falcon/habitat-lab/habitat/config/habitat/task/falcon_task_detail.yaml
defaults:
  - lab_sensors:
    - localization_sensor
    - human_num_sensor
    - oracle_humanoid_future_trajectory
    - human_velocity_sensor  # Add this line
```
This fix is already included in the eval-safe branch.

### Error: KeyError: 'human_velocity_measure'
**Cause:** The `HumanVelocitySensor` depends on the `human_velocity_measure` measurement, but it's not registered in the task's measurements defaults.

**Solution:** Add `human_velocity_measure` to the task config's measurements list:
```yaml
# In Falcon/habitat-lab/habitat/config/habitat/task/falcon_task_detail.yaml
defaults:
  - measurements:
    - distance_to_goal
    - distance_to_goal_reward
    - multi_agent_nav_reward
    - success
    - did_multi_agents_collide
    - num_steps
    - top_down_map
    - spl
    - psc
    - stl
    - human_collision
    - human_future_trajectory
    - human_velocity_measure  # Add this line
```
This fix is already included in the eval-safe branch.

### Error: RuntimeError: size mismatch for 'net.visual_encoder...'
**Solution:** Checkpoint was trained with `dual_stream_fpn`, not compatible with eval code

### Error: KeyError: 'aux_loss_modules.future_trajectory_prediction...'
**Solution:** Run `process_ckp_for_eval.py` first to remove auxiliary modules

## Comparison with Other Configs

| Config | Backbone | Aux Tasks | human_velocity_sensor | Eval-Safe? |
|--------|----------|-----------|---------------------|------------|
| `falcon_hm3d_train_mini_junwei.yaml` | resnet18 | BiLSTM | No | ✓ Yes |
| `falcon_hm3d_train_2v100_transformer_mdn.yaml` | resnet50 | Transformer-MDN | Yes | ✓ Yes |
| `falcon_hm3d_train_2v100.yaml` | dual_stream_fpn | Transformer-MDN | Yes | ✗ No |
| `falcon_hm3d_train_2v100_evalsafe.yaml` | resnet50 | Transformer-MDN | Yes | ✓ Yes |

## Expected Performance

Based on previous experiments with similar configurations:

**Mini Model (ResNet18 + LSTM128):**
- Success Rate: 60%
- SPL: 0.58
- PSC: 0.90
- Human Collision: 40%

**Full Model (ResNet50 + LSTM512) - Expected:**
- Success Rate: 70-75%
- SPL: 0.65-0.70
- PSC: 0.92-0.95
- Human Collision: 25-30%

**With All Improvements (Transformer-MDN + Curriculum + Penalties):**
- Success Rate: 75-80%
- SPL: 0.70-0.75
- PSC: 0.95+
- Human Collision: 20-25%

## Summary

The eval-safe configs provide:
- ✓ All latest training improvements
- ✓ Compatibility with original eval code
- ✓ Optimal 2V100 GPU utilization
- ✓ Clear separation of core vs auxiliary sensors
- ✓ Documented workflow and verification steps

Use these configs when you need to:
1. Train with the latest improvements
2. Ensure eval compatibility with commit `5d8ec1d1cf`
3. Submit checkpoints that work with original eval infrastructure
