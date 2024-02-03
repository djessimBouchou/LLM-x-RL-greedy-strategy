################################################
###              VERSION 2.0                 ###
################################################

# Goal :
# Create an environment with one door, one key and one goal so the agent must get the key before reaching the door to then access to the final goal postion
# Observations : Box() = Image with different colors for the different objects
# Action space : Discrete(4) = UP, RIGHT, DOWN, LEFT



import gymnasium as gym
from gymnasium import spaces
import pygame
import random
import seaborn as sns

import numpy as np
import copy

NB_ROOMS = 5

SIZE_ONE_BLOCK = 20 # Size of one block in pixels
FPS = 64

COLOR_SET = sns.color_palette(None, NB_ROOMS).as_hex()

def create_walls(grid_width, grid_height, num_rooms):
    """
    Notations : 0 = empty, 1 = wall, -1 = not available for select new corner
    """
    # Empty map
    map = np.zeros(shape=(grid_width, grid_height), dtype = np.int32)

    # Create the walls around
    map[0:grid_width, 0] = 1
    map[0:grid_width, grid_height-1] = 1
    map[0, 0:grid_height] = 1
    map[grid_width-1, 0:grid_height] = 1
    
    # And prevent the future corners to be chosen next to a wall
    map[1:grid_width-1, 1] = -1
    map[1:grid_width-1, grid_height-2] = -1
    map[1, 1:grid_height-1] = -1
    map[grid_width-2, 1:grid_height-1] = -1

    
    # Create rooms
    for idx_room in range(1, num_rooms):

        # Choose corner
        possibilities = np.argwhere(map == 0)
        if len(possibilities) > 0:
            corner_coords = random.choice(possibilities)
        else :
            num_rooms = idx_room
            break

        # Choose directions and expand walls AND restrict the area around the walls to be used as future corners
        potential_directions = ["left", "right", "top", "bottom"]
        direction_1, direction_2 = np.random.choice(potential_directions, size = 2, replace= False)
        for i in range(corner_coords[0]-1, corner_coords[0]+2):
            for j in range(corner_coords[1]-1, corner_coords[1]+2):
                map[i, j] = -1
        map[corner_coords[0], corner_coords[1]] = 1
        
        for direction in (direction_1, direction_2):
            i, j = corner_coords
            if direction == "left":
                i -= 1
                while not map[i, j] == 1:
                    map[i, j] = 1
                    if map[i, j-1] == 0:
                        map[i, j-1] = -1
                    if map[i, j+1] == 0:
                        map[i, j+1] = -1
                    i -= 1
            elif direction == "right":
                i += 1
                while not map[i, j] == 1:
                    map[i, j] = 1
                    if map[i, j-1] == 0:
                        map[i, j-1] = -1
                    if map[i, j+1] == 0:
                        map[i, j+1] = -1
                    i += 1
            elif direction == "top":
                j -= 1
                while not map[i, j] == 1:
                    map[i, j] = 1
                    if map[i-1, j] == 0:
                        map[i-1, j] = -1
                    if map[i+1, j] == 0:
                        map[i+1, j] = -1
                    j -= 1
            elif direction == "bottom":
                j += 1
                while not map[i, j] == 1:
                    map[i, j] = 1
                    if map[i-1, j] == 0:
                        map[i-1, j] = -1
                    if map[i+1, j] == 0:
                        map[i+1, j] = -1
                    j += 1

    # Transform all the -1 to normal empty spaces (0)
    map = np.where(map == -1, 0, map)
    return map

def number_rooms(map):

    map = np.where(map == 1, -1, map)
    current_room = 1

    # While there is non numbered rooms
    while 0 in map:
        position = np.argwhere(map == 0)[0]
        stack = [position]
        while len(stack) > 0:
            position = stack.pop(-1)
            map[position[0], position[1]] = current_room
            if map[position[0] - 1, position[1]] == 0:
                stack.append([position[0] - 1, position[1]])
            if map[position[0] + 1, position[1]] == 0:
                stack.append([position[0] + 1, position[1]])
            if map[position[0], position[1] - 1] == 0:
                stack.append([position[0], position[1] - 1])
            if map[position[0], position[1] + 1] == 0:
                stack.append([position[0], position[1] + 1])
        
        current_room += 1
    map = np.where(map == -1, 0, map)

    return map


def create_graph(map, nb_rooms):

    adjancy_list = {num_room : dict() for num_room in range(1, nb_rooms+1)} # nb 1 : nb 2 : list wall coordinates

    for i in range(1, map.shape[0]-1):
        for j in range(1, map.shape[0]-1):
            top = map[i+1, j]
            bottom = map[i-1, j]
            right = map[i, j+1]
            left = map[i, j-1]
            
            if top != bottom and top!=0 and bottom!=0:
                if bottom not in adjancy_list[top].keys():
                    adjancy_list[top][bottom] = [(i, j)]
                    adjancy_list[bottom][top] = [(i, j)]
                else :
                    adjancy_list[top][bottom].append((i, j))
                    adjancy_list[bottom][top].append((i, j))

            elif left != right and left!=0 and right!=0:
                if left not in adjancy_list[right].keys():
                    adjancy_list[right][left] = [(i, j)]
                    adjancy_list[left][right] = [(i, j)]
                else :
                    adjancy_list[right][left].append((i, j))
                    adjancy_list[left][right].append((i, j))

    return adjancy_list


def create_doors(map, adjancy_list, nb_rooms):

    # Choose the path to go from one room to another, using Kruskal

    final_map = copy.deepcopy(map)
    
    new_number_rooms = dict()
    new_number_rooms[1] = 1
    curr_room_new_idx = 2

    rooms_explored = [1]

    while len(rooms_explored) != nb_rooms:

        possible_next_rooms = []
        for room in rooms_explored:
            for next_potential_room in adjancy_list[room].keys():
                if next_potential_room not in rooms_explored:
                    possible_next_rooms.append((next_potential_room, adjancy_list[room][next_potential_room]))

        next_room, coord_potential_doors = random.choice(possible_next_rooms)

        # New room
        rooms_explored.append(next_room)
        final_map = np.where(map == next_room, curr_room_new_idx, final_map)
        
        # Create door:
        coord_door = random.choice(coord_potential_doors)
        final_map[coord_door[0], coord_door[1]] = -curr_room_new_idx

        curr_room_new_idx += 1

    return final_map


class KeyDoorsEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, grid_width, grid_height, nb_rooms = 2, render_mode = "rgb_array", seed = None):
        super(KeyDoorsEnv, self).__init__()

        # Action space : Discrete(4) = RIGHT, DOWN, LEFT, UP
        self.action_space = spaces.Discrete(4)

        self._action_to_direction = {
                0: np.array([1, 0]), # RIGHT
                1: np.array([0, 1]), # DOWN
                2: np.array([-1, 0]), # LEFT
                3: np.array([0, -1]), # UP
            }
        

        # Observation space
        self.grid_width = grid_width 
        self.grid_height = grid_height 
        
        self.observation_space = spaces.Box(low = 0, high = 255, shape=(self.grid_height, self.grid_width, 3), dtype = np.uint8)
        # Note : PyGame inverses x and y coordinates, so we follow their representation
        
        # Other
        self.render_mode = render_mode
        self.seed = seed
        self.nb_rooms = nb_rooms


    def step(self, action):
        """ Execute one time step within the environment"""
        
        if type(action) != int:
            action = int(action)
        assert action in self._action_to_direction.keys() # Check if action exists

        action_direction = self._action_to_direction[action]

        # New potential location of the agent (clipped so it stays on the grid)
        new_location = np.zeros((2,), dtype=int)
        new_location[0] = int(np.clip(self._agent_location[0] + action_direction[0], 0, self.grid_width-1))
        new_location[1] = int(np.clip(self._agent_location[1] + action_direction[1], 0, self.grid_height-1))

        # If we are on the position of the key
        for e, key_location in self._keys_location.items():
            if np.array_equal(key_location,new_location):
                self._has_key[e] = 1
                self._agent_location = new_location

        # Else, if there is no wall we also make the move 
        if self.map[new_location[0], new_location[1]] > 0 or self.map[new_location[0], new_location[1]] == -1:
            self._agent_location = new_location
        
        # Or finally, if we want to pass a door and we have the key
        elif self.map[new_location[0], new_location[1]] < -1:
            if self._has_key[-self.map[new_location[0], new_location[1]]] == 1:
                self._agent_location = new_location

        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else -0.01 # Penalty at each step
        info = self._get_info()
        truncated = False

        self.render(mode = self.render_mode)
        observation = self._get_obs()

        self.nb_steps += 1

        return observation, reward, terminated, truncated, info
        

    def _get_obs(self):
        return self._observation
    
    def _get_info(self):
        return {}
    
    def reset(self, seed=None):
        """Reset the state of the environment to an initial state"""
        
        # Seed the environment if we want to keep the same one
        if seed :
            super().reset(seed=seed)
            np.random.seed(seed = seed)
            random.seed(seed)
        elif self.seed :
            super().reset(seed=self.seed)
            np.random.seed(seed = self.seed)
            random.seed(self.seed)

        self.nb_steps = 0


        map = create_walls(self.grid_width, self.grid_height, self.nb_rooms)
        map = number_rooms(map)
        if self.nb_rooms > np.max(map):
            self.nb_rooms = np.max(map)
        adjancy_list = create_graph(map, self.nb_rooms)
        self.map = create_doors(map, adjancy_list, self.nb_rooms)

        # Create initial position agent
        self._agent_location = random.choice(np.argwhere(self.map == 1))
        self.map[self._agent_location[0], self._agent_location[1]] = -1

        # Set keys
        self._keys_location = {}
        for room_number in range(2, self.nb_rooms+1):
            if room_number == 2 and len(np.argwhere(self.map == room_number - 1)) == 0: # In case the first room is only one square ... 
                self._keys_location[room_number] = self._agent_location
            else :
                key_location = random.choice(np.argwhere(self.map == room_number - 1))
                self._keys_location[room_number] = key_location

        # Set goal
        final_room_number = self.nb_rooms
        self._target_location = random.choice(np.argwhere(self.map == final_room_number))

        # If agent has keys or not
        self._has_key = {e: 0 for e in self._keys_location.keys()}
        if self.map[self._keys_location[2][0], self._keys_location[2][1]] == -1: # In case the first room is only one square ... 
            self._has_key[2] = 1

        # Render 
        pygame.init()
        width = self.grid_width 
        height = self.grid_height 
        self.screen = pygame.Surface((width, height))
        self.clock = pygame.time.Clock()

        if self.render_mode == "human":
            pygame.display.set_caption("Key Door Game")
            self.display = pygame.display.set_mode((width* SIZE_ONE_BLOCK, height* SIZE_ONE_BLOCK))
    
        self.render()

        observation = self._get_obs()    
        info = self._get_info()

        return observation, info


    def get_human_player_move(self):
        
        action = None

        while action == None :
            for event in pygame.event.get():       
                if event.type == pygame.KEYDOWN:

                    if event.key == pygame.K_RIGHT:
                        action = 0
                    if event.key == pygame.K_LEFT:
                        action = 2
                    if event.key == pygame.K_DOWN:
                        action = 1
                    if event.key == pygame.K_UP:
                        action = 3
        
        return action

    def render(self, mode=None):

        if mode is None:
            mode = self.render_mode
        
        #Background
        self.screen.fill((67,70,75))

        # Position target
        pygame.draw.rect(self.screen, "RED", (self._target_location[0], self._target_location[1], 1, 1))

        # Position walls and doors
        for i in range(self.grid_width):
            for j in range(self.grid_height):
                if self.map[i][j] == 0:
                    pygame.draw.rect(self.screen, "WHITE", (i, j, 1, 1))
                elif self.map[i][j] < -1 :
                    pygame.draw.rect(self.screen, COLOR_SET[-self.map[i][j] - 1], (i, j, 1, 1))

        # Position keys
        for e, key_location in self._keys_location.items():
            if not self._has_key[e]:
                pygame.draw.rect(self.screen, COLOR_SET[e - 1], (key_location[0], key_location[1], 1, 1))

        # Position agent
        pygame.draw.rect(self.screen, "GREEN", (self._agent_location[0], self._agent_location[1], 1, 1))

        # Observation (we have to transpose as Pygame swaps axes)
        self._observation = pygame.surfarray.array3d(self.screen).transpose(1, 0, 2)

        if mode == "human":
            self.display.blit(pygame.transform.scale(self.screen, self.display.get_rect().size), (0, 0))
            pygame.display.update()


if __name__ == '__main__':

    print("Let's play a game")
    env = KeyDoorsEnv(14, 14, nb_rooms=NB_ROOMS, render_mode="human", seed = 10)

    env.reset()
    for i in range(1000):
        env.render()
        action = env.get_human_player_move()
        # action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        if done :
            break
    env.close()
    print(obs, reward, done, truncated, info)
    print(obs.shape)
    print("OK")