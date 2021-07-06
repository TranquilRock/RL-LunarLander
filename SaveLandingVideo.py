import gym
import matplotlib.animation as animation
import matplotlib.pyplot as plt


def saveLandingVideo(Name, env, actions):  # Save an episode's video
    env = gym.make('LunarLander-v2')
    env.reset()
    fig, ax = plt.subplots()
    imgs = []
    env.reset()
    for act in actions:
        observation, reward, done, _ = env.step(act)
        img = plt.imshow(env.render(mode='rgb_array'), animated=True)
        imgs.append([img])
        if done:
            break  # Crashes
    ani = animation.ArtistAnimation(
        fig, imgs, interval=50, blit=True, repeat_delay=1000)
    ani.save(Name)
