"""
Configuration parameters for DAIR-FedMoE framework.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Transformer parameters
    num_layers: int = 12
    hidden_dim: int = 512
    num_heads: int = 8
    ff_expansion: int = 4
    dropout: float = 0.1
    
    # MoE parameters
    num_stable_experts: int = 4
    num_drift_experts: int = 4
    expert_hidden_dim: int = 512
    gate_hidden_dim: int = 128
    
    # Drift detection
    drift_window_size: int = 500
    drift_smoothing_factor: float = 0.95
    
    # Loss reweighting
    min_weight: float = 1.0
    max_weight: float = 5.0
    weight_epsilon: float = 1e-6

@dataclass
class TrainingConfig:
    """Training configuration."""
    # Federated learning
    num_clients: int = 20
    num_rounds: int = 200
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 1e-3
    
    # RL parameters
    ppo_clip: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    ppo_epochs: int = 4
    actor_lr: float = 5e-4
    critic_lr: float = 1e-3
    
    # Reward weights
    drift_reward_weight: float = 2.0
    expert_count_weight: float = 0.5

@dataclass
class PrivacyConfig:
    """Privacy configuration."""
    # Differential privacy
    clip_norm: float = 1.0
    noise_scale: float = 1.2
    delta: float = 1e-5
    
    # Secure aggregation
    use_secure_agg: bool = True
    min_clients: int = 10

@dataclass
class DatasetConfig:
    """Dataset configuration."""
    # ISCX-VPN
    iscx_vpn_classes: int = 14
    iscx_vpn_samples_per_client: Tuple[int, int] = (3000, 5000)
    
    # ISCX-Tor
    iscx_tor_classes: int = 20
    iscx_tor_samples_per_client: Tuple[int, int] = (3000, 4500)
    
    # CIC-IDS2017
    cic_ids_classes: int = 10
    cic_ids_samples_per_client: Tuple[int, int] = (2500, 3500)
    
    # UNSW-NB15
    unsw_nb15_classes: int = 10
    unsw_nb15_samples_per_client: Tuple[int, int] = (3000, 4000)

@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    privacy: PrivacyConfig = PrivacyConfig()
    dataset: DatasetConfig = DatasetConfig()
    
    def __post_init__(self):
        """Validate configuration parameters."""
        assert self.model.num_stable_experts > 0, "Number of stable experts must be positive"
        assert self.model.num_drift_experts > 0, "Number of drift experts must be positive"
        assert self.training.num_clients > 0, "Number of clients must be positive"
        assert self.training.num_rounds > 0, "Number of rounds must be positive"
        assert self.privacy.clip_norm > 0, "Clip norm must be positive"
        assert self.privacy.noise_scale > 0, "Noise scale must be positive" 