import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.config.default_structured_configs import AuxLossConfig
from habitat_baselines.rl.ppo.policy import Net
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass

@dataclass
class PeopleCountingLossConfig(AuxLossConfig):
    """People Counting predictive coding loss"""

    max_human_num: int = 6
    loss_scale: float = 0.1

@dataclass
class GuessHumanPositionLossConfig(AuxLossConfig):
    """Guess Human Position predictive coding loss"""

    max_human_num: int = 6
    position_dim: int = 2
    loss_scale: float = 0.1

@dataclass
class FutureTrajectoryPredictionLossConfig(AuxLossConfig):
    """Future Trajectory predictive coding loss with Transformer + MDN"""

    max_human_num: int = 6
    future_step: int = 4
    loss_scale: float = 0.1
    num_mixture_components: int = 5  # K Gaussian components for multi-modal prediction
    transformer_layers: int = 2  # Number of Transformer Encoder layers
    transformer_heads: int = 8  # Number of attention heads

    # Two-Phase Curriculum Training parameters
    use_curriculum: bool = False  # Enable two-phase curriculum training
    warmstart_steps: int = 100000  # Phase 1 duration: freeze backbone, train Transformer-MDN
    warmstart_aux_lr_multiplier: float = 5.0  # Phase 1: High LR multiplier for auxiliary task
    finetune_loss_weight_normalize: bool = True  # Phase 2: Normalize gradient magnitudes
    
@baseline_registry.register_auxiliary_loss(name="people_counting")
class PeopleCounting(nn.Module):
    r"""
    People Counting task helps the agent estimate the number of people in the current scene.
    The output is a discrete value between 0 and max_human_num, representing the number of people detected.
    """

    def __init__(
        self,
        action_space: gym.spaces.Box,
        net: Net,
        max_human_num: int = 6,
        position_dim: int = 2,
        loss_scale: float = 0.1,
        future_step: int = 4,
    ):
        super().__init__()
        self.max_human_num = max_human_num
        self.loss_scale = loss_scale
        hidden_size = net.output_size
        
        # LSTM to process temporal information
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        
        # Attention mechanism to focus on important features
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4, batch_first=True)
        
        # Classifier to predict the number of people
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, max_human_num + 1),  # Output logits for classes 0 to max_human_num
        )
        
        # CrossEntropy loss for classification
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, aux_loss_state, batch):
        # Use perception embedding as input
        scene_features = aux_loss_state['rnn_output']  # (batch_size, hidden_size)
        
        # Pass through LSTM to capture temporal dependencies
        lstm_output, _ = self.lstm(scene_features)  # (batch_size, hidden_size)

        # Apply Attention mechanism
        attn_output, _ = self.attention(lstm_output, lstm_output, lstm_output)  # (batch_size, seq_len, hidden_size)
        
        # Average pooling over the sequence length dimension to aggregate features
        # attn_output_mean = attn_output.mean(dim=1)  # (batch_size, hidden_size)

        # Pass the result through the classifier
        logits = self.classifier(attn_output)  # (batch_size, max_human_num + 1)
        
        logits = torch.clamp(logits, min=-10, max=10)
        # Ground truth is the number of people in the scene
        target = batch["observations"]["human_num_sensor"].squeeze(-1).long()  # (batch_size,)
        
        # Calculate CrossEntropy loss
        ori_loss = self.loss_fn(logits, target)

        # FIX: Remove sigmoid that causes gradient vanishing
        # Use clamping instead to prevent extreme values
        loss = self.loss_scale * torch.clamp(ori_loss, max=5.0)

        return dict(loss=loss)

@baseline_registry.register_auxiliary_loss(name="guess_human_position")
class GuessHumanPosition(nn.Module):
    def __init__(
        self,
        action_space: gym.spaces.Box,
        net: Net,
        max_human_num: int = 6,
        position_dim: int = 2,
        loss_scale: float = 0.1,
        future_step: int = 4,
    ):
        super().__init__()
        self.loss_scale = loss_scale
        hidden_size = net.output_size
        self.position_dim = position_dim
        self.max_human_num = max_human_num
        
        self.lstm = nn.LSTM(input_size=hidden_size + 1, hidden_size=hidden_size, batch_first=True)
        
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4, batch_first=True)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, max_human_num * position_dim),
        )
        self.loss_fn = nn.MSELoss(reduction='none')

    def forward(self, aux_loss_state, batch):

        scene_features = aux_loss_state['rnn_output']  # (t, n, -1)
        human_num_features = batch["observations"]["human_num_sensor"].to(torch.float32)
        features = torch.cat((scene_features, human_num_features), dim=-1)

        lstm_output, _ = self.lstm(features)  # (num_step, hidden_size)
        attn_output, _ = self.attention(lstm_output, lstm_output, lstm_output)  # (num_step, hidden_size)

        positions_pred = self.classifier(attn_output)  # (max_human_num * position_dim)
        batch_size = scene_features.size(0)
        positions_pred = positions_pred.view(batch_size, self.max_human_num, self.position_dim)  # (n, max_human_num, position_dim)

        positions_gt = batch["observations"]["oracle_humanoid_future_trajectory"][:, :, 0, :]  # (n, num_people, position_dim)
        positions_gt_agent0 = batch["observations"]["localization_sensor"][:, [0, 2]]
        positions_gt_agent0_repeated = positions_gt_agent0.unsqueeze(1).repeat(1, 6, 1)
        positions_gt_relative = positions_gt - positions_gt_agent0_repeated

        mask = (positions_gt != -100.0).all(dim=-1).unsqueeze(-1)  # (n, num_people, 1)
        
        loss_per_position = self.loss_fn(positions_pred, positions_gt_relative)  # (n, max_human_num, position_dim)
        
        masked_loss = loss_per_position * mask  # (batch_size, max_human_num, future_step, position_dim)
        
        # if mask.sum() < 1:
        #     loss = torch.norm(loss_per_position) / 1e5
        # else:
        #     loss = masked_loss.sum() / mask.sum()
        #     max_val = masked_loss.max().detach()
        #     if max_val < 1e-5:
        #         loss = torch.norm(loss_per_position) / 1e5
        #     else:
        #         loss = loss / max_val 
        
        # return dict(loss=loss)

        if mask.sum() < 1:
            loss = torch.norm(loss_per_position) / 1e5
        else:
            loss_mean = masked_loss.mean()
            loss_std = masked_loss.std()

            if loss_std > 1e-5:
                normalized_loss = (masked_loss - loss_mean) / loss_std
            else:
                normalized_loss = masked_loss / loss_mean

            loss = normalized_loss.sum() / mask.sum()

        # FIX: Remove sigmoid, use direct loss with clamping
        final_loss = torch.clamp(loss, max=2.0) * self.loss_scale

        return dict(loss=final_loss)

@baseline_registry.register_auxiliary_loss(name="future_trajectory_prediction")
class FutureTrajectoryPrediction(nn.Module):
    """
    Transformer + MDN architecture for multi-modal future trajectory prediction.

    Key improvements:
    1. Transformer Encoder (global receptive field) replaces BiLSTM (sequential processing)
    2. Mixture Density Network outputs GMM distribution instead of single point
    3. NLL loss captures uncertainty and multi-modality

    Architecture:
    - Input: Robot features (δ_R) + Human positions (P_i^t) + Human count
    - Transformer Encoder: 2 layers, 8 heads (parallel processing)
    - MDN Head: Outputs (π, μ, σ) for K=5 Gaussian components per (human, timestep, x/y)
    - Loss: -log(Σ_k π_k * N(P_gt | μ_k, σ_k))

    Gradient Flow Justification:
    By training with NLL loss, the shared backbone (ResNet + Policy LSTM) learns to
    encode uncertainty in the latent features. The robot's representation δ_R will
    capture "Human is at X and could move to regions Y or Z" instead of just "Human is at X".
    This uncertainty-aware encoding helps the main policy make safer decisions (e.g.,
    maintaining larger clearance when human behavior is uncertain).
    """
    def __init__(
        self,
        action_space: gym.spaces.Box,
        net: Net,
        max_human_num: int = 6,
        position_dim: int = 2,
        loss_scale: float = 0.1,
        future_step: int = 4,
        num_mixture_components: int = 5,
        transformer_layers: int = 2,
        transformer_heads: int = 8,
        use_curriculum: bool = False,
        warmstart_steps: int = 100000,
        warmstart_aux_lr_multiplier: float = 5.0,
        finetune_loss_weight_normalize: bool = True,
    ):
        super().__init__()
        self.max_human_num = max_human_num
        self.position_dim = position_dim
        self.future_step = future_step
        self.loss_scale = loss_scale
        self.num_mixture_components = num_mixture_components
        hidden_size = net.output_size

        # Two-Phase Curriculum parameters
        self.use_curriculum = use_curriculum
        self.warmstart_steps = warmstart_steps
        self.warmstart_aux_lr_multiplier = warmstart_aux_lr_multiplier
        self.finetune_loss_weight_normalize = finetune_loss_weight_normalize
        self._current_step = 0  # Track training progress

        # Input projection: Combine scene features, human count, and current positions
        input_dim = hidden_size + 1 + max_human_num * position_dim

        # Project input to hidden_size for Transformer
        self.input_projection = nn.Linear(input_dim, hidden_size)

        # Transformer Encoder (replaces BiLSTM + Self-Attention)
        # Global receptive field allows instant pattern recognition
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=transformer_heads,
            dim_feedforward=hidden_size * 4,  # Standard Transformer ratio
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers
        )

        # MDN Head: Outputs GMM parameters (π, μ, σ) for multi-modal prediction
        # For each (human, timestep, coordinate), output K mixture components:
        #   - K weights (π): mixture coefficients (sum to 1)
        #   - K * position_dim means (μ): center of each Gaussian (x, y)
        #   - K * position_dim stds (σ): spread of each Gaussian (x, y)
        # Total per prediction: K + K*2 + K*2 = K*(1 + 2*position_dim)
        params_per_component = 1 + 2 * position_dim  # π + μ_x,μ_y + σ_x,σ_y
        total_output_dim = max_human_num * future_step * num_mixture_components * params_per_component

        self.mdn_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, total_output_dim),
        )

    def set_step(self, step: int):
        """
        Update current training step for curriculum scheduling.
        Should be called from trainer before each forward pass.
        """
        self._current_step = step

    def get_curriculum_loss_scale(self) -> float:
        """
        Get the current loss scale based on curriculum phase.

        Phase 1 (Warm-Start, 0 - warmstart_steps):
            - Apply high loss scale multiplier to quickly train Transformer-MDN
            - Backbone is frozen (controlled by trainer)
            - loss_scale = base_scale * warmstart_aux_lr_multiplier

        Phase 2 (Joint Finetuning, warmstart_steps+):
            - Use base loss scale
            - Backbone is unfrozen (controlled by trainer)
            - Optionally normalize loss magnitudes
        """
        if not self.use_curriculum:
            return self.loss_scale

        if self._current_step < self.warmstart_steps:
            # Phase 1: Warm-Start with high LR multiplier
            return self.loss_scale * self.warmstart_aux_lr_multiplier
        else:
            # Phase 2: Joint Finetuning with normalized weights
            # TODO: Implement loss weight normalization based on gradient magnitudes
            # For now, just use base loss scale
            return self.loss_scale

    def forward(self, aux_loss_state, batch):
        """
        Forward pass: Transformer Encoder → MDN Head → NLL Loss

        Flow:
        1. Concatenate robot features (δ_R) + human count + current positions
        2. Project to hidden_size and add positional encoding (optional)
        3. Process through Transformer Encoder (parallel, global receptive field)
        4. MDN head outputs GMM parameters (π, μ, σ)
        5. Compute NLL: -log(Σ_k π_k * N(P_gt | μ_k, σ_k))
        """
        scene_features = aux_loss_state["rnn_output"]  # (batch_size, hidden_size)
        batch_size = scene_features.size(0)
        human_num_features = batch["observations"]["human_num_sensor"].to(torch.float32)  # (batch_size, 1)
        position_features = batch["observations"]["oracle_humanoid_future_trajectory"][:, :, 0, :].reshape(batch_size, -1)  # (batch_size, max_human_num * position_dim)

        # Concatenate all input features
        features = torch.cat((scene_features, human_num_features, position_features), dim=-1)  # (batch_size, input_dim)

        # Project to hidden_size and add sequence dimension for Transformer
        projected_features = self.input_projection(features).unsqueeze(1)  # (batch_size, 1, hidden_size)

        # Pass through Transformer Encoder (processes entire sequence in parallel)
        # This captures global interaction patterns (e.g., "Human A yielding to Human B")
        transformer_output = self.transformer_encoder(projected_features)  # (batch_size, 1, hidden_size)
        transformer_output = transformer_output.squeeze(1)  # (batch_size, hidden_size)

        # MDN Head: Output GMM parameters
        mdn_output = self.mdn_head(transformer_output)  # (batch_size, total_output_dim)

        # Parse GMM parameters for each (human, timestep)
        # Reshape: (batch_size, max_human_num, future_step, K, params_per_component)
        K = self.num_mixture_components
        params_per_component = 1 + 2 * self.position_dim  # π + μ_x,μ_y + σ_x,σ_y
        mdn_params = mdn_output.view(batch_size, self.max_human_num, self.future_step, K, params_per_component)

        # Extract and process each parameter type
        # π (mixture weights): Apply softmax to ensure they sum to 1
        pi_logits = mdn_params[..., 0]  # (batch_size, max_human_num, future_step, K)
        pi = F.softmax(pi_logits, dim=-1)  # Normalize across K components

        # μ (means): x, y coordinates for each Gaussian component
        mu = mdn_params[..., 1:1+self.position_dim]  # (batch_size, max_human_num, future_step, K, 2)

        # σ (standard deviations): Apply softplus to ensure positivity
        sigma_raw = mdn_params[..., 1+self.position_dim:]  # (batch_size, max_human_num, future_step, K, 2)
        sigma = F.softplus(sigma_raw) + 1e-6  # Add epsilon for numerical stability

        # Get ground truth positions (relative to robot)
        positions_gt = batch["observations"]["oracle_humanoid_future_trajectory"][:, :, -self.future_step:, :]  # (batch_size, num_people, future_step, position_dim)
        positions_gt_agent0 = batch["observations"]["localization_sensor"][:, [0, 2]]  # (batch_size, 2)
        positions_gt_agent0_repeated = positions_gt_agent0.unsqueeze(1).unsqueeze(2).repeat(1, self.max_human_num, self.future_step, 1)  # (batch_size, max_human_num, future_step, 2)
        positions_gt_relative = positions_gt - positions_gt_agent0_repeated

        # Create mask for valid humans (positions != -100.0)
        mask = (positions_gt != -100.0).all(dim=-1)  # (batch_size, max_human_num, future_step)

        # Compute NLL loss for Gaussian Mixture Model
        # P(y|x) = Σ_k π_k * N(y | μ_k, σ_k)
        # L = -log(P(ground_truth | x))
        loss = self._gaussian_mixture_nll(
            positions_gt_relative,  # Ground truth positions
            pi,    # Mixture weights
            mu,    # Means
            sigma, # Standard deviations
            mask   # Valid human mask
        )

            loss = normalized_loss.sum() / mask.sum()

        # FIX: Remove sigmoid, use direct loss with clamping
        final_loss = torch.clamp(loss, max=2.0) * self.loss_scale

        return dict(loss=final_loss)

    def _gaussian_mixture_nll(self, targets, pi, mu, sigma, mask):
        """
        Compute Negative Log-Likelihood for Gaussian Mixture Model.

        Mathematical formulation:
            P(y|x) = Σ_k π_k * N(y | μ_k, σ_k²)
            L = -log(P(y_gt | x)) = -log(Σ_k π_k * N(y_gt | μ_k, σ_k²))

        Args:
            targets: (batch_size, max_human_num, future_step, position_dim) - ground truth
            pi: (batch_size, max_human_num, future_step, K) - mixture weights
            mu: (batch_size, max_human_num, future_step, K, position_dim) - means
            sigma: (batch_size, max_human_num, future_step, K, position_dim) - stds
            mask: (batch_size, max_human_num, future_step) - valid human mask

        Returns:
            Scalar NLL loss
        """
        # Expand targets to compare with all K components
        targets_expanded = targets.unsqueeze(-2)  # (batch_size, max_human_num, future_step, 1, position_dim)

        # Compute Gaussian log-probability for each component
        # log N(y | μ, σ²) = -0.5 * [log(2π) + log(σ²) + (y-μ)²/σ²]
        log_2pi = torch.log(torch.tensor(2.0 * 3.14159265359, device=targets.device))

        # Compute squared Mahalanobis distance: (y - μ)² / σ²
        diff = targets_expanded - mu  # (batch_size, max_human_num, future_step, K, position_dim)
        mahalanobis = (diff / sigma) ** 2  # (batch_size, max_human_num, future_step, K, position_dim)

        # Log probability of each Gaussian component
        # Sum over position_dim (x and y are independent)
        log_gauss = -0.5 * (log_2pi + 2 * torch.log(sigma) + mahalanobis)  # (batch_size, max_human_num, future_step, K, position_dim)
        log_gauss = log_gauss.sum(dim=-1)  # (batch_size, max_human_num, future_step, K)

        # Weighted log probability: log(π_k * N(y | μ_k, σ_k))
        log_pi_gauss = torch.log(pi + 1e-8) + log_gauss  # (batch_size, max_human_num, future_step, K)

        # Log-sum-exp trick for numerical stability: log(Σ exp(x)) = max(x) + log(Σ exp(x - max(x)))
        max_log_pi_gauss = log_pi_gauss.max(dim=-1, keepdim=True)[0]
        log_sum_exp = max_log_pi_gauss + torch.log(
            torch.sum(torch.exp(log_pi_gauss - max_log_pi_gauss), dim=-1, keepdim=True) + 1e-8
        )
        log_sum_exp = log_sum_exp.squeeze(-1)  # (batch_size, max_human_num, future_step)

        # Negative log-likelihood
        nll = -log_sum_exp  # (batch_size, max_human_num, future_step)

        # Apply mask and compute mean NLL over valid predictions
        masked_nll = nll * mask

        if mask.sum() < 1:
            # No valid humans, return small penalty
            return torch.tensor(0.01, device=targets.device, requires_grad=True)
        else:
            # Average NLL over valid predictions
            return masked_nll.sum() / mask.sum()

cs = ConfigStore.instance()

cs.store(
    package="habitat_baselines.rl.auxiliary_losses.people_counting",
    group="habitat_baselines/rl/auxiliary_losses",
    name="people_counting",
    node=PeopleCountingLossConfig,
)

cs.store(
    package="habitat_baselines.rl.auxiliary_losses.guess_human_position",
    group="habitat_baselines/rl/auxiliary_losses",
    name="guess_human_position",
    node=GuessHumanPositionLossConfig,
)

cs.store(
    package="habitat_baselines.rl.auxiliary_losses.future_trajectory_prediction",
    group="habitat_baselines/rl/auxiliary_losses",
    name="future_trajectory_prediction",
    node=FutureTrajectoryPredictionLossConfig,
)