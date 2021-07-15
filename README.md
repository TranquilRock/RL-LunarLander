# RL-LunarLander
NTUEE ML2020-2021 spring course 
- Some of the code are done by TA or online resources.
- step() returns：
    - observation / state
    - reward
    - done
    - 其餘資訊
- Reward
    - 小艇墜毀得到 -100 分
    - 小艇在黃旗幟之間成功著地則得 100~140 分
    - 噴射主引擎（向下噴火）每次 -0.3 分
    - 小艇最終完全靜止則再得 100 分
    - 小艇每隻腳碰觸地面 +10 分
- Example
    -  ![](https://github.com/AndyDu01/RL-LunarLander/blob/main/Train.mp4)
