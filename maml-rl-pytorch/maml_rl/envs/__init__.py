from gym.envs.registration import register

# Bandit
# ----------------------------------------
for k in [5, 10, 50]:
    register(
        'Bandit-K{0}-v0'.format(k),
        entry_point='maml_rl.envs.bandit:BernoulliBanditEnv',
        kwargs={'k': k}
    )

# TabularMDP
# ----------------------------------------
register(
    'TabularMDP-v0',
    entry_point='maml_rl.envs.mdp:TabularMDPEnv',
    kwargs={'num_states': 10, 'num_actions': 5},
    max_episode_steps=10
)

# Grid World
# ----------------------------------------
# entry_point: The Python entrypoint of the environment class (e.g. module.name:Class)
register(
    'GridWorld-v0',
    entry_point='maml_rl.envs.gridworld:GridWorldEnv',
    kwargs={'prob_name': 'gw10Two1'},
)

register(
    'GridWorld-v1',
    entry_point='maml_rl.envs.gridworld:GridWorldEnv',
    kwargs={'prob_name': 'gw20Three1'},
)

register(
    'KeyDoor-v0',
    entry_point='maml_rl.envs.keydoor:KeyDoorEnv',
    kwargs={'prob_name': 'ky10One'},
)

register(
    'Treasure-v0',
    entry_point='maml_rl.envs.treasure:TreasureEnv',
    kwargs={'prob_name': 'it10'},
)

# Mountain-Car
# ----------------------------------------
register(
    'MountainCar-v3',
    entry_point='maml_rl.envs.mountaincar:MountainCarEnv',
    kwargs={},
)

# Mujoco
# ----------------------------------------
register(
    'AntVel-v2',
    entry_point='maml_rl.envs.utils.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'maml_rl.envs.mujoco.ant:AntVelEnv'}
)

register(
    'AntDir-v2',
    entry_point='maml_rl.envs.utils.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'maml_rl.envs.mujoco.ant:AntDirEnv'}
)

register(
    'AntPos-v1',
    entry_point='maml_rl.envs.utils.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'maml_rl.envs.mujoco.ant:AntPosEnv'}
)

register(
    'HalfCheetahVel-v2',
    entry_point='maml_rl.envs.utils.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'maml_rl.envs.mujoco.half_cheetah:HalfCheetahVelEnv'}
)

register(
    'HalfCheetahDir-v2',
    entry_point='maml_rl.envs.utils.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'maml_rl.envs.mujoco.half_cheetah:HalfCheetahDirEnv'}
)

# 2D Navigation
# ----------------------------------------
register(
    '2DNavigation-v0',
    entry_point='maml_rl.envs.navigation:Navigation2DEnv',
    max_episode_steps=100
)
