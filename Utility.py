import numpy as np

def TestAction(env, agent, actions_list):
    agent.network.eval()
    test_total_reward = []
    for actions in actions_list:
        state = env.reset()
        total_reward = 0
        done = False
        done_count = 0
        for action in actions:
            state, reward, done, _ = env.step(action)
            done_count += 1
            total_reward += reward
            if done:
                break
        print(f"Your reward is : %.2f" % total_reward)
        test_total_reward.append(total_reward)
    print(f"Your final reward is : %.2f" % np.mean(test_total_reward))
    return total_reward


def GenerateAction(env, agent, NUM_OF_TEST=5, quite=False):
    agent.network.eval()
    test_total_reward = []
    action_list = []
    for i in range(NUM_OF_TEST):
        actions = []
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action, _ = agent.sample(state)
            actions.append(action)
            state, reward, done, _ = env.step(action)
            total_reward += reward
        test_total_reward.append(total_reward)
        action_list.append(actions)
    distribution = {}
    for actions in action_list:
        for action in actions:
            if action not in distribution.keys():
                distribution[action] = 1
            else:
                distribution[action] += 1
    if not quite:
        print(f"Final reward is : %.2f" % np.mean(test_total_reward))
        print("Action list's distribution: ", distribution)
        np.save("Action_List_test.npy", np.array(action_list))
    return action_list
