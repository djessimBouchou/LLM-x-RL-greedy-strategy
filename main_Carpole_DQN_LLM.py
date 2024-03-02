# ----------------------------------------------------------------
#                  LLM + DQN  on Carpole
# ----------------------------------------------------------------

import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer

from carpole_env import CartPoleEnv
from models.DQN_LLM import DQNAgent

env = CartPoleEnv()

name_model = "mistralai/Mistral-7B-Instruct-v0.2"
# name_model = "google/gemma-7b"

parser = argparse.ArgumentParser(description='Argument Parser for the parameters')

parser.add_argument('--network_type', type=str, default="Mlp", help='Type of network (default: Mlp)')
parser.add_argument('--memory_size', type=int, default=1000, help='Size of memory (default: 1000)')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 32)')
parser.add_argument('--target_update', type=int, default=100, help='Frequency of target network update (default: 100)')
parser.add_argument('--LLM_epsilon_decay', type=float, default=1/2000, help='Epsilon decay for LLM (default: 1/2000)')
parser.add_argument('--max_LLM_epsilon', type=float, default=1.0, help='Maximum epsilon for LLM (default: 1.0)')
parser.add_argument('--min_LLM_epsilon', type=float, default=0.9999, help='Minimum epsilon for LLM (default: 0.9999)')
parser.add_argument('--epsilon_decay', type=float, default=1/2000, help='Epsilon decay (default: 1/2000)')
parser.add_argument('--max_epsilon', type=float, default=1.0, help='Maximum epsilon (default: 1.0)')
parser.add_argument('--min_epsilon', type=float, default=0.1, help='Minimum epsilon (default: 0.1)')
parser.add_argument('--warm_up_llm_episodes', type=int, default=30, help='Warm up episodes for LLM (default: 30)')
parser.add_argument('--max_context_LLM', type=int, default=2020, help='Maximum context for LLM (default: 2020)')
parser.add_argument('--num_frames', type=int, default=10000, help='Number of frames (default: 10000)')
parser.add_argument('--wandb_monitor', action='store_true', help='Enable wandb monitoring')
parser.add_argument('--name', type=str, default='LLM_Only', help='Name of the training (default: LLM_Only)')

args = parser.parse_args()


dqn_carpole = DQNAgent(
    network_type=args.network_type,
    env=env,
    name_llm_model=name_model,
    memory_size=args.memory_size,
    batch_size=args.batch_size,
    target_update=args.target_update,
    LLM_epsilon_decay=args.LLM_epsilon_decay,
    max_LLM_epsilon=args.max_LLM_epsilon,
    min_LLM_epsilon=args.min_LLM_epsilon,
    epsilon_decay=args.epsilon_decay,
    max_epsilon=args.max_epsilon,
    min_epsilon=args.min_epsilon,
    warm_up_llm_episodes=args.warm_up_llm_episodes,
    max_context_LLM=args.max_context_LLM,
)

dqn_carpole.train(num_frames=args.num_frames, wandb_monitor=args.wandb_monitor, name=args.name)
