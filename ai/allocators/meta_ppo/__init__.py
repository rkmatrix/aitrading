"""Meta-PPO Adaptive Online Allocator package (Phase 15)."""
from .allocator_env import AllocatorEnv, AllocatorStep
from .policy import MetaPPOPolicy
from .trainer import OnlineAllocatorTrainer
from .meta_controller import MetaController
from .constraints import project_simplex, clamp_trust_region
from .reward import RewardEngine
from .buffer import OnPolicyBuffer
from .telemetry import Telemetry
