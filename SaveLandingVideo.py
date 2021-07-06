import gym
import matplotlib.animation as animation
import matplotlib.pyplot as plt


def saveLandingVideo(Name, env, Agent):  # Save an episode's video
    env = gym.make('LunarLander-v2')
    env.reset()
    fig, ax = plt.subplots()
    imgs = []
    state = env.reset()
    done = False
    while not done:
        Agent.sample(state)
        action, _ = Agent.sample(state)
        state, _, done, _ = env.step(action)
        img = plt.imshow(env.render(mode='rgb_array'), animated=True)
        imgs.append([img])
    ani = animation.ArtistAnimation(
        fig, imgs, interval=50, blit=True, repeat_delay=1000)
    ani.save(Name)
