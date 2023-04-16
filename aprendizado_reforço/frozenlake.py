import gym

gym.envs.register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78, # optimum = .8196
)

env = gym.make('FrozenLakeNotSlippery-v0', render_mode='ansi')

S_n = env.observation_space.n
print("n_posições", S_n)

print("sorteando posições aleatórias no grid:")
for i in range(10):
     print("posição: ", env.observation_space.sample())

# print("ações posiveis no espaço: ", env.action_space.n)
# print( "ações:   1. LEFT: 0 ,2. DOWN = 1 , 3. RIGHT = 2 , 4. UP = 3")
# actions = dict({0: "L", 1: "D", 2:"R", 3:"U"})
# for i in range(10):
#     print("executando ação: ", actions[env.action_space.sample()])


##executando ações no grid
env.reset()
action = 0
(observation, reward, v, done, prob) = env.step(action)
grid = env.render()
print(grid)
print(observation, reward, done, prob)


#andando pelo grid
env.reset()
for x in [1,1,2,2,1,2]:
    print(env.step(x))
    print(env.render())


# Make the environment based on deterministic policy
environment = gym.make("FrozenLake-v1", is_slippery=False, render_mode='ansi')
#environment.seed(8)
environment.reset()
for i in range(10):
    action = 2
    print(environment.step(action))
    print(environment.render())