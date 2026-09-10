# 点到达与边指令完成：模块边界、迭代和真实 R2R 单点测试

日期：2026-08-31

## 1. 固定模块边界

本轮只优化两个判定模块，不实现“在指令路上/走错了”：

1. `PointNavigationExecutor` 独占“是否物理到达外部所选点”的判定，并只返回
   `point_navigation_arrived` 或未到达。选点策略和指令语义不能反向修改该信号。
2. 收到到达信号后必须建立 node，并把完整 action history 和按时间均匀抽取的
   RGB keyframe 写入 incoming edge。这是功能性 invariant，不计算正确率。
3. `NodeTransitionInstructionCompletionJudge` 只判断该 incoming edge 是否完成当前
   子指令，输出 `completed/unknown`。`unknown` 不暗含 on-route/off-route。

所有回放起点均来自保存的真实 R2R `reference_path` state，模型没有看到最终
测地距离、示范未来点或人工完成标签。

## 2. 物理到达判定

### 2.1 通用错误与迭代

| 版本 | 通用问题 | 修改 |
|---|---|---|
| legacy 3x3 | 九个底边点同时受遮挡/转向影响，易把出画当到达 | 保留为历史基线 |
| dense v1-v3 | 加密 query 会改变控制 tracker；一帧相关丢失仍会误报 | 控制簇和到达簇分工，增加时序确认 |
| v5 | 3x3 导航、3x3 goal、45点到达簇；独立 arrival tracker；消除大部分误报 | 四帧确认、近场入口 |
| v6 | v5 在控制簇先丢失时来不及积累四帧，产生漏报 | 导航/goal/独立稠密 arrival 三簇同时终止时使用终端共识 |
| v7 | 门框遮挡可在尚未到点时触发视觉消失 | 非近场到达增加累计移动量 `>= 0.70 * 初始选点深度`；不可达投影在入口拒绝 |

当前默认是 `dense_stop_motion_guard_v7`：

- 3x3 crop 中心导航簇；
- 3x3 原底边 goal/crop 簇；
- 5x9、共45个吸附地面的稠密 arrival 簇；
- arrival TAPIR 与控制 TAPIR 共享权重但保持独立 causal state；
- 常规 goal+arrival 低可见率四帧确认，或三簇终端共识；
- 非近场使用 70% 初始深度移动门槛；
- 深度 `<=1.0 m` 且点位于图像底部21%的点可在第0步判为近场到达；
- 初始 navmesh 无路径时返回 `selected_point_unreachable`。

### 2.2 结果

| 测试集 | TP | FP | FN | TN | 判定准确率 | 真到达执行率 |
|---|---:|---:|---:|---:|---:|---:|
| legacy，固定20例 | 3 | 5 | 1 | 11 | 70.0% | 15.0% |
| v5，专门漏报8例 | 3 | 0 | 5 | 0 | 37.5% | 37.5% |
| v6，34例开发并集 | 21 | 1 | 0 | 12 | 97.1% | 61.8% |
| v7，同一34例 | 22 | 0 | 0 | 12 | 100.0% | 64.7% |
| v7，冻结后新增10例 | 9 | 0 | 0 | 1 | 100.0% | 90.0% |
| v7，44个互异状态合计 | 31 | 0 | 0 | 13 | 100.0% | 70.5% |

“判定准确率”把正确保持未到达的 TN 计为正确；“真到达执行率”只计确实进入
0.75 m 且正确报到达的 TP。44/44 仍是有限固定状态结果（Wilson 95% 下界约
92%），不能宣称全 R2R 已达到100%。原始结果：

- `outputs/point_arrival_profile_round11_v7_union34/`
- `outputs/heldout_round1_v7_modules10/`

## 3. 到达后建节点 invariant

冻结新增10例中有9次物理到达，9/9 均建立 arrival node；每条边都保存真实
action history 和最多5张时间排序 keyframe。未到达例也可保存 diagnostic stop
node，但不能标成 arrival。该项只报告 `9/9 invariant satisfied`，不写“节点
正确率”。

## 4. 边是否完成子指令

### 4.1 真值与输入

均分示范路径得到的“子指令 endpoint”在人工检查中出现明显矛盾，例如：相机
确实穿门进入卧室但均分 endpoint 判未完成；相机没有完成第二个右侧走廊但
endpoint 距离判完成。因此主真值改为在模型调用前冻结的人工边事件标签：

- 开发集25条有效到达边：7 `completed`、18 `unknown`；
- 冻结后新增验证集9条有效到达边：2 `completed`、7 `unknown`；
- upstream 物理到达失败的边不进入该语义模块分母。

标签：

- `data/instruction_completion_manual_labels_v1.json`
- `data/instruction_completion_heldout_labels_v1.json`

每次判定输入为：当前子指令、前后 node 六视图、Grounded-SAM 环境语义、前后
在线位姿、incoming edge 完整 action、时间排序 keyframe。输出 schema 只有
`completed/unknown`。

### 4.2 通用错误与迭代

1. 当前节点看见目标不等于该 edge 完成：v3 强制比较 KF0、前节点和关系变化。
2. 门在当前/前一节点都可见：v5要求明确 source side -> frame -> destination
   side，而不是对当前房间做分类。
3. “about halfway”没有可见的精确50%边界：单纯加强提示两轮仍漏判。v7 harness
   让 VLM 提取可见证据，并只对近似 partial vertical 形式融合节点高度变化、
   前后多视角楼梯检测和真实边移动量。普通 top/bottom/landing 不用此例外。

### 4.3 结果

| 版本/集合 | TP | FP | FN | TN | 准确率 | completed召回率 |
|---|---:|---:|---:|---:|---:|---:|
| v3，开发25边 | 5 | 0 | 2 | 18 | 92.0% | 71.4% |
| v7，开发25边 | 7 | 0 | 0 | 18 | 100.0% | 100.0% |
| v7，冻结新增9边 | 1 | 0 | 1 | 7 | 88.9% | 50.0% |

冻结验证集唯一错误是 case 21：指令要求绕过床后向右朝拱门；六视图在终点仍
能从后视方向看到床，VLM把“仍可见”误当成“未在身后”，从而漏报 completed。
这说明全景下的“behind”需要显式的视角方位/对象 bearing 变化，而不能只用
对象是否出现。为了保持冻结验证的独立性，本轮没有再针对该例调 v7。

原始结果：

- `outputs/instruction_completion_round13_v7_full_round1/`
- `outputs/instruction_completion_round14_v7_full_partial/`
- `outputs/instruction_completion_round15_v7_full_portals/`
- `outputs/heldout_round1_v7_modules10/instruction_completion_manual_score.json`

代表性两模块视频位于
`outputs/v7_representative_two_modules_v3/dense_stop_motion_guard_v7/case_082/point_arrival.mp4`：
控制帧显式显示 N/G/A 三个角色点簇，结束5帧显示 edge、`COMPLETED/UNKNOWN`
结果、instruction 和实时 top-down；分辨率为固定 960x480。

## 5. 当前默认与限制

- 生产 CLI 默认物理配置：`dense_stop_motion_guard_v7`，每点最多32控制步。
- 默认完成 harness：`v7_structured_partial_extent`。
- 标准 Habitat 语义路径和独立序列策略都只在 point arrival 后调用二值完成判定；
  旧 `SubInstructionNodeMatcher` 只保留诊断，不再决定完成。
- 本轮没有实现 unknown 的途中/走错分类，也没有用它计算整体探索策略准确率。
- 下一轮若继续，应先冻结新的关系型验证集，再增加带视角编号的 landmark bearing
  变化，专门处理 `behind/right/left/between`，不能在当前9例上继续追分。
