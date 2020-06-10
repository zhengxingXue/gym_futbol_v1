from stable_baselines.common.policies import FeedForwardPolicy


class CustomPolicy2v2(FeedForwardPolicy):
    """
    Custom MLP policy for 2v2
    """
    def __init__(self, *args, **kwargs):
        super(CustomPolicy2v2, self).__init__(*args, **kwargs,
                                              net_arch=[256, 256, dict(pi=[128, 128],
                                                                       vf=[128, 128])],
                                              feature_extraction="mlp")


def create_custom_policy(net_arch):
    """
    function to create custom policy
    eg. net_arch = [64, 64]
    eg. net_arch = [64, dict(pi=[64],vf=[64])]
    """
    class CustomPolicy(FeedForwardPolicy):
        def __init__(self, *args, **kwargs):
            super(CustomPolicy, self).__init__(*args, **kwargs, net_arch=net_arch, feature_extraction="mlp")

    return CustomPolicy
