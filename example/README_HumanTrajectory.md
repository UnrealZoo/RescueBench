# Human Rescue Trajectory Export

这个文档说明如何使用 `example/Rescue_HumanControl.py` 记录人工救援轨迹，并将每个 `level` 下每个 `episode` 的绝对坐标保存为 `jsonl` 文件。

## 目的

该导出格式适合把人类操作轨迹作为评测基线，与其他模型进行对比，例如：

- 成功率
- 总步数
- 路径长度
- 搜索阶段和搬运阶段的轨迹差异
- 人类与模型的轨迹相似度

## 保存格式

脚本会按如下方式输出：

```text
example/human_trajectories/<env_id>/level_<level>.jsonl
```

例如：

```text
example/human_trajectories/UnrealRescue-SuburbNeighborhood_Day/level_1.jsonl
```

其中：

- 一个 `jsonl` 文件对应一个 `env_id + level`
- 文件中的每一行对应一个 `episode`
- 每个 `episode` 内部包含完整的轨迹序列 `trajectory`

## 运行方式

在项目根目录运行：

```bash
python example/Rescue_HumanControl.py --level 0
```

如果你想指定输出目录：

```bash
python example/Rescue_HumanControl.py \
  --level 1 \
  --trajectory-dir example/human_trajectories
```

如果你希望在采集轨迹的同时保存每一步的 `RGB + Depth`（RGB-D）观测，可开启：

```bash
python example/Rescue_HumanControl.py \
  --level 1 \
  --trajectory-dir example/human_trajectories \
  --save-rgbd
```

如果你想自定义 RGB-D 的输出目录：

```bash
python example/Rescue_HumanControl.py \
  --level 1 \
  --save-rgbd \
  --rgbd-dir example/rgbd_dump
```

## RGB-D 观测导出

启用 `--save-rgbd` 后，脚本会在 `reset` 后保存 `step=0`，并在每次 `step()` 后保存当前观测对应的 RGB 与深度。

默认输出目录（未指定 `--rgbd-dir`）：

```text
<trajectory-dir>/rgbd_frames/<env_id>/level_<level>/episode_<episode_id>/
```

每个 step 会保存三类文件：

- `rgb_000123.png`: 当前观测的 RGB 图像
- `depth_000123.npy`: 当前深度图（float32 原始值）
- `depth_000123.png`: 深度可视化图（归一化到 0-255，便于快速浏览）

说明：

- 深度图会优先从当前 observation 中提取（当 observation 类型包含 depth 通道时）
- 如果 observation 不包含 depth 通道，脚本会通过 UnrealCV 拉取当前主相机的深度图

## 轨迹内容

每一行是一个完整的 `episode` 记录，结构类似：

```json
{
  "env_id": "UnrealRescue-SuburbNeighborhood_Day",
  "level": 1,
  "episode_id": 0,
  "seed": 10,
  "result": "success",
  "terminated": true,
  "truncated": false,
  "timeout_sec": 180,
  "elapsed_sec": 132.4,
  "steps": 132,
  "fps": 18.4,
  "trajectory": [
    {
      "step": 0,
      "x": 620.0,
      "y": -179.0,
      "z": 77.0,
      "roll": 0.0,
      "yaw": 0.0,
      "pitch": 0.0,
      "picked": false,
      "reward": 0.0,
      "timestamp": 0.0
    },
    {
      "step": 1,
      "x": 630.4,
      "y": -178.7,
      "z": 77.0,
      "roll": 0.0,
      "yaw": 3.0,
      "pitch": 0.0,
      "action": {
        "move": [30.0, 200.0],
        "head": 0,
        "anim": 0
      },
      "picked": false,
      "reward": 0.0,
      "timestamp": 0.05
    }
  ]
}
```

## 字段说明

### Episode 级字段

- `env_id`: 当前环境 ID
- `level`: 当前难度等级
- `episode_id`: 当前测试点编号
- `seed`: 随机种子参数
- `result`: `success` 或 `failed`
- `terminated`: 是否成功结束
- `truncated`: 是否超时或失败结束
- `timeout_sec`: 当前 episode 使用的时间上限（秒）
- `elapsed_sec`: 当前 episode 实际消耗时间（秒）
- `steps`: 当前 episode 的交互步数
- `fps`: 当前 episode 的平均运行帧率
- `trajectory`: 当前 episode 的完整轨迹

### Trajectory 级字段

- `step`: 第几步，`0` 表示 reset 后的初始位置
- `x`, `y`, `z`: agent 的绝对世界坐标
- `roll`, `yaw`, `pitch`: agent 的绝对姿态
- `action`: 该步的人类输入动作
- `picked`: 当前是否已经抱起伤员
- `reward`: 该步环境返回的 reward
- `timestamp`: 从当前 episode 开始到该步的秒数

## 坐标来源

轨迹使用环境内部的绝对位姿，而不是根据键盘动作自行推算，因此更适合作为评测基准。

- `reset()` 后的起点直接从 Unreal 中读取当前主角的绝对位置和朝向
- `step()` 后的轨迹点来自 `info["Pose"][0]`

这两个值都来自环境内部维护的 Unreal 世界坐标。

### 关于起点问题的说明

早期版本的轨迹导出脚本在记录 `step=0` 时，直接使用了 `reset()` 返回的 `info["pose"]`。但在当前救援环境里，主角会在 `reset()` 过程中再次被设置到任务指定起点，因此 `info["pose"]` 可能对应的是重定位前或尚未完全同步的旧位置。

这会导致一个现象：

- `step=0` 的位置不是真实起点
- `step=1` 开始的位置才是 agent 的实际任务起点
- 于是 `step=0 -> step=1` 之间会出现一次异常大的空间位移

这个异常大的首帧位移就是文档中提到的“初始跳变”。

目前这个问题已经在采集脚本中修复：`step=0` 不再依赖 `info["pose"]`，而是在 `reset()` 完成后，直接从 Unreal 读取当前主角的真实绝对位姿作为起点。

### 关于 `get_obj_location()` 返回 `error` 的说明

在部分运行环境中，如果在 `reset()` 之后立刻单独调用 UnrealCV 的 `get_obj_location(player_name)`，底层接口可能会返回字符串 `error`，从而在后续解析为浮点数时触发报错。

这通常不是轨迹数据本身有问题，而是因为该时刻对象状态还没有通过这一路单独查询接口稳定返回。

为避免这个问题，当前采集脚本不再使用单独的 `get_obj_location()` / `get_obj_rotation()` 作为起点读取方式，而是改为使用环境内部本来就在持续使用的批量位姿接口 `get_pose_img_batch()` 获取主角 pose。这样和环境每一步更新 observation、pose 的方式保持一致，稳定性更高。

## 时间限制

脚本会在每个 episode 的 `reset()` 完成后开始计时。

- `level 0-2` 默认限制为 180 秒
- `level 3-4` 默认限制为 300 秒

更准确地说，脚本会优先读取每个测试点 `jsonl` 中的 `timeout` 字段；如果没有该字段，才会回退到上述默认规则。

当实际时间超过上限时：

- 当前 episode 会被记为 `result = "timeout"`
- `truncated = True`
- 当前轨迹仍会被完整保存到输出 `jsonl`

## 分析脚本

仓库中还提供了一个分析脚本：

```text
example/analyze_human_trajectory.py
```

它可以完成这些事情：

- 读取一个 `jsonl` 轨迹文件
- 统计每个 episode 的路径长度、时长、是否成功、是否超时、首次抱起伤员的步数
- 导出每个 episode 的 2D 轨迹图
- 导出汇总 `summary.csv`
- 在终端中显示每个 episode 的实际耗时

### 基本用法

```bash
python example/analyze_human_trajectory.py \
  --input example/human_trajectories/UnrealRescue-SuburbNeighborhood_Day/level_1.jsonl
```

### 只分析某个 episode

```bash
python example/analyze_human_trajectory.py \
  --input example/human_trajectories/UnrealRescue-SuburbNeighborhood_Day/level_1.jsonl \
  --episode-id 0
```

### 输出内容

默认输出到：

```text
example/human_trajectories/UnrealRescue-SuburbNeighborhood_Day/analysis/level_1/
```

其中包括：

- `summary.csv`: 每个 episode 的统计结果
- `plots/episode_<id>.png`: 每个 episode 的 2D 轨迹图

分析脚本在终端输出中会显示每个 episode 的：

- `result`
- `steps`
- `time`
- `path2d`
- `pick_step`

如果轨迹中带有时间限制字段，还会显示 `实际耗时/时间上限`。

### 关于旧轨迹文件

分析脚本默认带有一个轻量修正：

- 如果前两个轨迹点之间存在异常大的初始跳变，默认会丢弃第一个点

这是为了兼容你之前已经生成的一些旧轨迹文件。也就是说：

- 新采集的轨迹应当已经修复起点问题
- 旧轨迹如果仍然包含错误的 `step=0`，分析脚本会优先过滤掉这个异常首点

这个阈值可以通过 `--initial-jump-threshold` 调整，设为 `0` 表示关闭。

## 评测建议

如果你后续要做人类轨迹和模型轨迹对比，建议模型也保存成同样的结构，至少保证以下字段一致：

- `env_id`
- `level`
- `episode_id`
- `trajectory[].step`
- `trajectory[].x`
- `trajectory[].y`
- `trajectory[].z`
- `trajectory[].picked`

这样后续更容易统一计算：

- 路径长度
- 首次发现伤员前的搜索距离
- 搬运后回程距离
- 成功率
- 平均完成时间

## 注意事项

- 当前脚本会根据 `gym_rescue/envs/setting/test_jsonl/level_<level>.jsonl` 的行数自动确定测试点数量，不再写死为 `20`
- 每次运行脚本都会向对应 `jsonl` 文件追加新结果，不会覆盖旧记录
- 如果你想避免重复记录，建议在实验前先清理目标 `jsonl` 文件
