from gym.envs.registration import register

register(
    id='futbol-v1',
    entry_point='gym_futbol_v1.envs:Futbol',
    kwargs={'number_of_player': 2},
)

register(
    id='futbol-v1-5v5',
    entry_point='gym_futbol_v1.envs:Futbol',
    kwargs={'number_of_player': 5},
)

register(
    id='futbol-v1-10v10',
    entry_point='gym_futbol_v1.envs:Futbol',
    kwargs={'number_of_player': 10},
)
