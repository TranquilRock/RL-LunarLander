import matplotlib.animation as animation
import time
import random
import gym
from tqdm.notebook import tqdm
from torch.distributions import Categorical
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
from pyvirtualdisplay import Display
import matplotlib.pyplot as plt

from IPython import display
%% capture
virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()


seed = 0xC8763


def fix(env, seed):
    env.seed(seed)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_deterministic(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


%% capture

env = gym.make('LunarLander-v2')

fix(env, seed)

start = time.time()

print(env.observation_space)

print(env.action_space)

initial_state = env.reset()
print(initial_state)

"""利用 `step()` 函式讓 agent 根據我們隨機抽樣出來的 `random_action` 動作。
而這個函式會回傳四項資訊：
- observation / state
- reward
- 完成與否
- 其餘資訊
"""
random_action = env.action_space.sample()
print(random_action)
observation, reward, done, info = env.step(random_action)
print(done)
print(reward)

"""第一項資訊 `observation` 即為 agent 採取行動之後，agent 對於環境的 observation 或者說環境的 state 為何。
而第三項資訊 `done` 則是 `True` 或 `False` 的布林值，當登月小艇成功著陸或是不幸墜毀時，代表這個回合（episode）也就跟著結束了，此時 `step()` 函式便會回傳 `done = True`，而在那之前，`done` 則保持 `False`。
"""


"""### Reward

而「環境」給予的 reward 大致是這樣計算：
- 小艇墜毀得到 -100 分
- 小艇在黃旗幟之間成功著地則得 100~140 分
- 噴射主引擎（向下噴火）每次 -0.3 分
- 小艇最終完全靜止則再得 100 分
- 小艇每隻腳碰觸地面 +10 分

> Reward for moving from the top of the screen to landing pad and zero speed is about 100..140 points.
> If lander moves away from landing pad it loses reward back.
> Episode finishes if the lander crashes or comes to rest, receiving additional -100 or +100 points.
> Each leg ground contact is +10.
> Firing main engine is -0.3 points each frame.

"""


def saveLandingVideo(Name, env, action):
    fig, ax = plt.subplots()
    imgs = []
    env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        observation, reward, done, _ = env.step(action)
        img = plt.imshow(env.render(mode='rgb_array'), animated=True)
        imgs.append([img])
    ani = animation.ArtistAnimation(
        fig, imgs, interval=50, blit=True, repeat_delay=1000)
    ani.save(Name)


"""## Policy Gradient

現在來搭建一個簡單的 policy network。
我們預設模型的輸入是 8-dim 的 observation，輸出則是離散的四個動作之一：
"""
class PolicyGradientNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 4)

    def forward(self, state):
        hid = torch.tanh(self.fc1(state))
        hid = torch.tanh(self.fc2(hid))
        return F.softmax(self.fc3(hid), dim=-1)


"""再來，搭建一個簡單的 agent，並搭配上方的 policy network 來採取行動。
這個 agent 能做到以下幾件事：
- `learn()`：從記下來的 log probabilities 及 rewards 來更新 policy network。
- `sample()`：從 environment 得到 observation 之後，利用 policy network 得出應該採取的行動。
而此函式除了回傳抽樣出來的 action，也會回傳此次抽樣的 log probabilities。
"""


class PolicyGradientAgent():

    def __init__(self, network):
        self.network = network
        self.optimizer = optim.SGD(self.network.parameters(), lr=0.001)

    def forward(self, state):
        return self.network(state)

    def learn(self, log_probs, rewards):
        # You don't need to revise this to pass simple baseline (but you can)
        loss = (-log_probs * rewards).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sample(self, state):
        action_prob = self.network(torch.FloatTensor(state))
        action_dist = Categorical(action_prob)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob

    def save(self, PATH):  # You should not revise this
        Agent_Dict = {
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }
        torch.save(Agent_Dict, PATH)

    def load(self, PATH):  # You should not revise this
        checkpoint = torch.load(PATH)
        self.network.load_state_dict(checkpoint["network"])
        # 如果要儲存過程或是中斷訓練後想繼續可以用喔 ^_^
        self.optimizer.load_state_dict(checkpoint["optimizer"])


"""最後，建立一個 network 和 agent，就可以開始進行訓練了。"""

network = PolicyGradientNetwork()
agent = PolicyGradientAgent(network)
#agent = PolicyGradientAgent()

"""## 訓練 Agent

現在我們開始訓練 agent。
透過讓 agent 和 environment 互動，我們記住每一組對應的 log probabilities 及 reward，並在成功登陸或者不幸墜毀後，回放這些「記憶」來訓練 policy network。
"""

agent.network.train()  # 訓練前，先確保 network 處在 training 模式
EPISODE_PER_BATCH = 5  # 每蒐集 5 個 episodes 更新一次 agent
NUM_BATCH = 400        # 總共更新 400 次

avg_total_rewards, avg_final_rewards = [], []

prg_bar = tqdm(range(NUM_BATCH))
for batch in prg_bar:

    log_probs, rewards = [], []
    total_rewards, final_rewards = [], []

    # 蒐集訓練資料
    for episode in range(EPISODE_PER_BATCH):

        state = env.reset()
        total_reward, total_step = 0, 0
        seq_rewards = []
        while True:

            action, log_prob = agent.sample(state)  # at , log(at|st)
            next_state, reward, done, _ = env.step(action)

            # [log(a1|s1), log(a2|s2), ...., log(at|st)]
            log_probs.append(log_prob)
            # seq_rewards.append(reward)
            state = next_state
            total_reward += reward
            total_step += 1
            rewards.append(reward)  # 改這裡
            # ! 重要 ！
            # 現在的reward 的implementation 為每個時刻的瞬時reward, 給定action_list : a1, a2, a3 ......
            #                                                       reward :     r1, r2 ,r3 ......
            # medium：將reward調整成accumulative decaying reward, 給定action_list : a1,                         a2,                           a3 ......
            #                                                       reward :     r1+0.99*r2+0.99^2*r3+......, r2+0.99*r3+0.99^2*r4+...... ,r3+0.99*r4+0.99^2*r5+ ......
            # boss : implement DQN
            if done:
                final_rewards.append(reward)
                total_rewards.append(total_reward)
                break

    print(f"rewards looks like ", np.shape(rewards))
    print(f"log_probs looks like ", np.shape(log_probs))
    # 紀錄訓練過程
    avg_total_reward = sum(total_rewards) / len(total_rewards)
    avg_final_reward = sum(final_rewards) / len(final_rewards)
    avg_total_rewards.append(avg_total_reward)
    avg_final_rewards.append(avg_final_reward)
    prg_bar.set_description(
        f"Total: {avg_total_reward: 4.1f}, Final: {avg_final_reward: 4.1f}")

    # 更新網路
    # rewards = np.concatenate(rewards, axis=0)
    rewards = (rewards - np.mean(rewards)) / \
        (np.std(rewards) + 1e-9)  # 將 reward 正規標準化
    agent.learn(torch.stack(log_probs), torch.from_numpy(rewards))
    print("logs prob looks like ", torch.stack(log_probs).size())
    print("torch.from_numpy(rewards) looks like ",
          torch.from_numpy(rewards).size())

"""### 訓練結果

訓練過程中，我們持續記下了 `avg_total_reward`，這個數值代表的是：每次更新 policy network 前，我們讓 agent 玩數個回合（episodes），而這些回合的平均 total rewards 為何。
理論上，若是 agent 一直在進步，則所得到的 `avg_total_reward` 也會持續上升，直至 250 上下。
若將其畫出來則結果如下：
"""

end = time.time()
plt.plot(avg_total_rewards)
plt.title("Total Rewards")
plt.show()

"""另外，`avg_final_reward` 代表的是多個回合的平均 final rewards，而 final reward 即是 agent 在單一回合中拿到的最後一個 reward。
如果同學們還記得環境給予登月小艇 reward 的方式，便會知道，不論**回合的最後**小艇是不幸墜毀、飛出畫面、或是靜止在地面上，都會受到額外地獎勵或處罰。
也因此，final reward 可被用來觀察 agent 的「著地」是否順利等資訊。
"""

plt.plot(avg_final_rewards)
plt.title("Final Rewards")
plt.show()

"""訓練時間

"""

print(f"total time is {end-start} sec")

"""## 測試"""

fix(env, seed)
agent.network.eval()  # 測試前先將 network 切換為 evaluation 模式
NUM_OF_TEST = 5  # Do not revise it !!!!!
test_total_reward = []
action_list = []
for i in range(NUM_OF_TEST):
    actions = []
    state = env.reset()

    img = plt.imshow(env.render(mode='rgb_array'))

    total_reward = 0

    done = False
    while not done:
        action, _ = agent.sample(state)
        actions.append(action)
        state, reward, done, _ = env.step(action)
        total_reward += reward
    print(total_reward)
    test_total_reward.append(total_reward)

    action_list.append(actions)  # 儲存你測試的結果
    print("length of actions is ", len(actions))

print(f"Your final reward is : %.2f" % np.mean(test_total_reward))

"""Action list 的長相"""

print("Action list looks like ", action_list)
print("Action list's shape looks like ", np.shape(action_list))

"""Action 的分布

"""

distribution = {}
for actions in action_list:
    for action in actions:
        if action not in distribution.keys():
            distribution[action] = 1
        else:
            distribution[action] += 1
print(distribution)

"""儲存 Model Testing的結果

"""

PATH = "Action_List_test.npy"  # 可以改成你想取的名字或路徑
np.save(PATH, np.array(action_list))

"""### 你要交到JudgeBoi的檔案94這個
儲存結果到本地端 (就是你的電腦裡拉 = = )

"""

files.download(PATH)

"""# Server 測試
到時候下面會是我們Server上測試的環境，可以給大家看一下自己的表現如何
"""

action_list = np.load(PATH, allow_pickle=True)  # 到時候你上傳的檔案
seed = 543  # 到時候測試的seed 請不要更改
fix(env, seed)

agent.network.eval()  # 測試前先將 network 切換為 evaluation 模式

test_total_reward = []
for actions in action_list:
    state = env.reset()
    img = plt.imshow(env.render(mode='rgb_array'))

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
