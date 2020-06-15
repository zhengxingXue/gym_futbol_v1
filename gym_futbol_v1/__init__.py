from gym.envs.registration import register

register(
    id='futbol-v1',
    entry_point='gym_futbol_v1.envs:Futbol',
    kwargs={'number_of_player': 2},
)

register(
    id='futbol2v2-v1',
    entry_point='gym_futbol_v1.envs:Futbol',
    kwargs={'number_of_player': 2},
)


register(
    id='futbol5v5-v1',
    entry_point='gym_futbol_v1.envs:Futbol',
    kwargs={'number_of_player': 5},
)

register(
    id='futbol10v10-v1',
    entry_point='gym_futbol_v1.envs:Futbol',
    kwargs={'number_of_player': 10},
)
