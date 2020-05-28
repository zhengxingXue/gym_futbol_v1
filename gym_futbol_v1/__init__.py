from gym.envs.registration import register

register(
    id='futbol-v1',
    entry_point='gym_futbol_v1.envs:Futbol',
    kwargs={'number_of_player': 2},
)
