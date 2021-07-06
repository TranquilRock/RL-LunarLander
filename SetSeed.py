import torch
import random
import numpy as np


def fixEnvironment(env, seed):
    env.seed(seed)
    env.action_space.seed(seed)


def fixTorch(seed, deteministic=False, benchmark=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.deterministic = deteministic
    torch.set_deterministic(deteministic)


def fixNumpy(seed):
    np.random.seed(seed)
    random.seed(seed)
