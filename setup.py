from setuptools import setup

setup(name='gym_futbol',
      version='1.0.0',
      install_requires=['gym', 'pymunk', 'numpy', 'matplotlib', 'opencv-python', 'stable-baselines[mpi]==2.10.0',
                        'tensorflow==2.6.4', 'imageio', 'IPython']  # And any other dependencies foo needs
)