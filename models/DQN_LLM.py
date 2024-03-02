# Initial source : https://github.com/Curt-Park/rainbow-is-all-you-need/blob/master/01.dqn.ipynb 

import os
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output
from .LLM import LLM

import pandas as pd
from time import time

import wandb

class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, obs_dim, size: int, batch_size: int = 32):
        self.obs_buf = np.zeros([size, *obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, *obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray, 
        rew: float, 
        next_obs: np.ndarray, 
        done: bool,
    ):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(obs=self.obs_buf[idxs],
                    next_obs=self.next_obs_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def __len__(self) -> int:
        return self.size
    

class MlpNetwork(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        """Initialization."""
        super(MlpNetwork, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, 128), 
            nn.ReLU(),
            nn.Linear(128, 128), 
            nn.ReLU(), 
            nn.Linear(128, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        return self.layers(x)
    

class CnnNetwork(nn.Module):
    def __init__(self, in_channels: int, out_dim: int):
        """Initialization."""
        super(CnnNetwork, self).__init__()

        dim_flatter = 64 # To change and calculate automatically

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=(2, 2), stride = 1), 
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 16, kernel_size=(2, 2), stride = 1),  
            nn.ReLU(), 
            nn.Flatten(),
            nn.Linear(dim_flatter, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        if len(x.shape) == 3:
            x = x[None, ...]
        x = x.permute([0, 3, 1, 2])
        out = self.layers(x)
        return out
    


class DQNAgent:
    """DQN Agent interacting with environment.
    
    Attribute:
        env (gym.Env): openAI Gym environment
        memory (ReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        epsilon (float): parameter for epsilon greedy policy
        epsilon_decay (float): step size to decrease epsilon
        max_epsilon (float): max value of epsilon
        min_epsilon (float): min value of epsilon
        target_update (int): period for target model's hard update
        gamma (float): discount factor
        dqn (Network): model to train and select actions
        dqn_target (Network): target model to update
        optimizer (torch.optim): optimizer for training dqn
        transition (list): transition information including 
                           state, action, reward, next_state, done
    """

    def __init__(
        self, 
        network_type: str,
        env: gym.Env,
        name_llm_model:str,
        memory_size: int,
        batch_size: int,
        target_update: int,
        warm_up_llm_episodes = 25,
        seed: int = None,
        LLM_epsilon_decay: float = 0.0001,
        max_LLM_epsilon: float = 1.0,
        min_LLM_epsilon: float = 0.1,
        epsilon_decay: float = 0.0001,
        max_epsilon: float = 1.0,
        min_epsilon: float = 0.1,
        gamma: float = 0.99,
        nb_actions_LLM: int = 10,
        description_game: str = "This is a game",
        dict_action: dict = {},
        max_context_LLM:int = 1000,
        
    ):
        """Initialization.
        
        Args:
            network_type (str): "Cnn" or "Mlp"
            env (gym.Env): openAI Gym environment
            memory_size (int): length of memory
            batch_size (int): batch size for sampling
            target_update (int): period for target model's hard update
            epsilon_decay (float): step size to decrease epsilon
            lr (float): learning rate
            max_epsilon (float): max value of epsilon
            min_epsilon (float): min value of epsilon
            gamma (float): discount factor
        """

        assert network_type in ["Cnn", "Mlp"]
        self.network_type = network_type

        # obs_dim = [2] # To modify if necessary
        obs_dim = env.observation_space.shape
        action_dim = env.action_space.n
        self.env = env
        self.memory = ReplayBuffer(obs_dim, memory_size, batch_size)
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.LLM_epsilon = max_LLM_epsilon
        self.LLM_epsilon_decay = LLM_epsilon_decay
        self.warm_up_llm_episodes = warm_up_llm_episodes
        self.seed = seed
        self.max_LLM_epsilon = max_LLM_epsilon
        self.min_LLM_epsilon = min_LLM_epsilon
        self.epsilon = max_LLM_epsilon
        self.epsilon_decay = epsilon_decay
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.target_update = target_update
        self.gamma = gamma
        self.LLM_next_selected_actions = list()
        self.nb_actions_selected_by_LLM = nb_actions_LLM
        self.description_game = description_game
        self.dict_action = dict_action
        self.max_context_LLM = max_context_LLM
        
        # device: cpu / gpu / mps
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(self.device)

        # networks: dqn, dqn_target
        if self.network_type == "Mlp" :
            self.dqn = MlpNetwork(obs_dim[0], action_dim).to(self.device)
            self.dqn_target = MlpNetwork(obs_dim[0], action_dim).to(self.device)
        elif self.network_type == "Cnn" :
            self.dqn = CnnNetwork(obs_dim[-1], action_dim).to(self.device)
            self.dqn_target = CnnNetwork(obs_dim[-1], action_dim).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()
        
        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters())

        # transition to store in memory
        self.transition = list()
        
        # mode: train / test
        self.is_test = False

        # LLM
        self.name_llm_model = name_llm_model
        self.llm = LLM(name_model = name_llm_model, device = self.device, max_tokens = 4, max_context = self.max_context_LLM)


    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # epsilon greedy policy
        if len(self.LLM_next_selected_actions) > 0:
            selected_action = self.LLM_next_selected_actions.pop(0)
        elif self.LLM_epsilon > np.random.random() and self.id_current_episode >= self.warm_up_llm_episodes:
            selected_action = self.select_action_LLM(state)
        elif self.epsilon > np.random.random():
            selected_action = self.env.action_space.sample()
        else: 
            selected_action = self.dqn(
                torch.FloatTensor(state).to(self.device)
            ).argmax()
            selected_action = selected_action.detach().cpu().numpy()
        
        if not self.is_test:
            self.transition = [state, selected_action]
        
        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated

        if not self.is_test:
            self.transition += [reward, next_state, done]
            self.memory.store(*self.transition)
    
        return next_state, reward, done

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        samples = self.memory.sample_batch()

        loss = self._compute_dqn_loss(samples)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
        
    def train(self, num_frames: int, wandb_monitor = True, name = "LLM_DQN", save_csv = True):
        """Train the agent."""
        
        if save_csv:
            df = pd.DataFrame(columns = ["Time", "Episode", "Score"])
            print(df.head())

        if wandb_monitor :
            wandb.login()
            
            config = {
                "network_type": self.network_type,
                "env": self.env,
                "name_llm_model":self.name_llm_model,
                "memory_size": self.memory_size,
                "batch_size": self.batch_size,
                "target_update": self.target_update,
                "warm_up_llm_episodes":self.warm_up_llm_episodes,
                "seed": self.seed,
                "LLM_epsilon_decay":self.LLM_epsilon_decay,
                "max_LLM_epsilon":self.max_LLM_epsilon,
                "min_LLM_epsilon":self.min_LLM_epsilon,
                "epsilon_decay":self.epsilon_decay,
                "max_epsilon": self.max_epsilon,
                "min_epsilon": self.min_epsilon,
                "gamma": self.gamma,
                "max_context_LLM": self.max_context_LLM,
            }
            run = wandb.init(
                # Set the project where this run will be logged
                project="DQN_LLM",
                name = name,
                # Track hyperparameters and run metadata
                config=config,
            )
        self.is_test = False
        
        state, _ = self.env.reset(seed=self.seed)
        update_cnt = 0
        epsilons = []
        losses = []
        scores = []
        score = 0
        self.llm.reset()
        self.id_current_episode = 0

        t0 = time()

        for frame_idx in range(1, num_frames + 1):
            action = self.select_action(state)
            self.llm.update(self.env, state, action)
            next_state, reward, done = self.step(action)
            
            state = next_state
            score += reward

            # if episode ends
            if done:
                state, _ = self.env.reset(seed=self.seed)
                scores.append(score)
                print("Score : ", score)
                self.id_current_episode += 1
                wandb.log({"score": score, "episode": self.id_current_episode, "global_step":frame_idx})
                self.llm.add_episode_done(score)

                if save_csv :
                    df.loc[len(df)] = [time() - t0, self.id_current_episode, score]
                t0 = time()
                score = 0

            # if training is ready
            if len(self.memory) >= self.batch_size:
                loss = self.update_model()
                losses.append(loss)
                update_cnt += 1
                
                # linearly decrease epsilon
                if self.id_current_episode >= self.warm_up_llm_episodes:
                    self.LLM_epsilon = max(
                        self.min_LLM_epsilon, self.LLM_epsilon - (
                            self.max_LLM_epsilon - self.min_LLM_epsilon
                        ) * self.LLM_epsilon_decay
                    )

                self.epsilon = max(
                    self.min_epsilon, self.epsilon - (
                        self.max_epsilon - self.min_epsilon
                    ) * self.epsilon_decay
                )

                wandb.log({"loss": loss, "epsilon":self.epsilon, "epsilon_LLM":self.LLM_epsilon, "global_step":frame_idx})

                epsilons.append(self.epsilon)
                
                # if hard update is needed
                if update_cnt % self.target_update == 0:
                    self._target_hard_update()

                    if save_csv:
                        df.to_csv(name + ".csv")
                
        self.env.close()
                
    def test(self) -> None:
        """Test the agent."""
        self.is_test = True
        
        naive_env = self.env
        
        state, _ = self.env.reset(seed=self.seed)
        done = False
        score = 0
        
        while not done:
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward
        
        print("score: ", score)
        self.env.close()
        
        # reset
        self.env = naive_env

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        """Return dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        curr_q_value = self.dqn(state).gather(1, action)
        next_q_value = self.dqn_target(
            next_state
        ).max(dim=1, keepdim=True)[0].detach()
        mask = 1 - done
        target = (reward + self.gamma * next_q_value * mask).to(self.device)

        # calculate dqn loss
        loss = F.smooth_l1_loss(curr_q_value, target)

        return loss

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())
                

    def select_action_LLM(self, state, nb_actions_selected = 10):
        """
        Call the LLM and return an action
        """

        self.LLM_next_selected_actions = self.llm.select_action(self.env, state, nb_actions_selected)

        return self.LLM_next_selected_actions.pop(0)
    
