import gymnasium as gym
import pygame

def get_human_player_move():
    
    action = None

    while action == None :
        for event in pygame.event.get():       
            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_RIGHT:
                    action = 0
                if event.key == pygame.K_LEFT:
                    action = 1
                if event.key == pygame.K_DOWN:
                    action = 2

    
    return action

env = gym.make("MiniGrid-Empty-16x16-v0", render_mode="human")
observation, info = env.reset(seed=42)
for _ in range(100):
   action = get_human_player_move()
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()
env.close()


