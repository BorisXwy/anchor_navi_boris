# Point Tracker / R2R 项目测试规则

## 0. 在线导航的 RGB-only 硬约束（2026-09-09，优先级最高）

本节覆盖本文中所有更早、与其冲突的历史规则和实验说明。正式导航系统在线运行时只能读取：Habitat RGB observation、原始 instruction/子指令、系统自己发出的离散 action history，以及从这些 RGB 计算出的二维地面/物体 mask、检测、点轨迹、crop、视觉 embedding 和纯视觉节点记忆。

以下信息严禁进入选点、点导航/到达判定、节点构建、指令完成判定、探索、回溯或任务 STOP 决策：

- simulator pose、position、rotation、GPS、compass 或真实朝向；
- depth/RGB-D、点深度、由深度反投影的三维坐标；
- pathfinder、navmesh、snap point、ShortestPath、geodesic distance；
- collision state、真实移动距离或根据 simulator pose 得到的 action 是否成功；
- R2R reference path、下一 waypoint、示范 heading 或 goal position。

允许系统根据自己发出的 turn action 累积一个**动作坐标系相对航向**，但它不是 simulator pose，且不得用真实 rotation 校正。允许根据相邻 RGB 的变化计算纯视觉运动量，但不得读取 collision feedback。

Habitat 原始句柄只归测试适配器所有：用于 episode 起点初始化、录制 top-down 视频和系统完成决策后的隐藏几何验证。验证数据必须写入独立的 `evaluation_only/evaluation_geometry.json`，depth 快照也只能位于 `evaluation_only/`；这些数据不得返回导航模块，也不得改变当次选择、动作、到达、节点判定或 STOP。正式入口必须把 `RGBOnlyPolicySimulator` 能力受限句柄交给导航代码；任何特权属性访问都应 fail closed。导航图节点中的 `position_xyz`、`base_yaw_rad` 和绝对视角 yaw 必须为空；边只保存命令 action、RGB 运动量和 RGB keyframe。

当前有效执行配置固定为 `policy_input_contract=rgb_only_v1` 和 `tracking_cluster_profile=rgb_only_dense_stop_v1`。旧的 RGB-D、pose、navmesh、geodesic 执行分支只作为历史代码/离线基准保留，不得从正式 CLI 选择，也不得计入当前结果。

本文件是本项目后续实验的强制规则。除非明确标为纯函数或接口单元测试，任何被称为“单点测试”“单点全流程测试”“单点模块测试”的实验都必须遵守本文件。规则变更必须在实验开始前写入本文件，不能在看到结果后修改口径。

## 1. “单点测试”的含义和测试层级

“单点测试”是从一条真实 R2R 示范轨迹中选取一个机器人状态作为测试起点的统称。它允许只指定一个或若干模块进行测试，不要求每次都运行完整链路，但必须明确属于以下哪一层级：

1. **单点模块测试**：只验证预先指定的模块及其必要上游依赖；不要求执行与目标无关的下游模块。
2. **单点全模块测试**：实际执行完整的语义点导航闭环，并继续验证节点、指令判定和物理回溯。

每次运行前必须在 `manifest.json` 中固定记录 `test_scope`（`module` 或 `full`）、`target_modules`、`required_upstream_modules`、`upstream_artifact_source` 和 `not_run_modules`。不能在看到结果后改变目标模块或测试层级。

单点全模块测试必须真实执行以下链路，不能只检查文件或 mock 返回值：

```text
真实 R2R reference_path 状态
  -> 完整指令拆解和当前子指令对齐
  -> Habitat 当前状态六/八视图 RGB
  -> 三模型 dense 2/3 多数票地面分割 + 独立开放词汇指令目标检测
  -> VLM 六视图地面选点
  -> 点导航执行器移动并停止
  -> 保存到达节点和边 action history
  -> 判断“上个节点到本节点”为 completed / unknown
  -> 使用新旧节点六视图选择回溯点
  -> 点导航物理回溯到原节点
  -> 位置+视觉重访验证和图闭环记录
```

如果其中任一模块没有执行，该实验仍可称为合规的“单点模块测试”，但不能称为“单点全模块测试通过”。

### 1.1 单点模块测试的最低真实输入

单点模块测试只需运行目标模块和得到其输入所必需的真实上游过程，但输入必须来自同一个真实 R2R episode/state，不能用合成图像、随机导航点或 mock 返回值替代：

| 指定测试模块 | 最低真实输入或必要上游过程 |
|---|---|
| 指令拆解 | 真实 episode 的完整原始 instruction |
| 地面感知 | 在真实示范 state 中由 Habitat 采集的六视图 RGB |
| VLM 选点 | 真实 state 的六视图、DINO+SAM mask、当前子指令和真实历史上下文 |
| 点导航执行器 | 真实 state 采集的 RGB，以及本次合规选点产生或既有合规运行保存的二维点、mask 和 crop；必须在 Habitat 中实际执行离散动作，但执行器不得读取 state/depth |
| 节点与边 | 从真实 state 出发并实际执行导航得到的前后 RGB、命令 action history 和 RGB keyframe；节点不存 pose/depth |
| 子指令完成判定 | 真实到达节点的前后六/八视图 RGB、RGB 语义、入边命令 action history 和 keyframe；不得读取节点位姿/depth，也不得用预期子指令 ID 泄漏答案 |
| 回溯 | 实际移动形成的两个真实节点、两端六视图 RGB 和正向命令 action history；必须物理执行回溯，不能 teleport/读取 pose/navmesh |
| 序列恢复 | 从真实 state 运行后自然产生的节点判定和探索结果；不得伪造 off-sequence 分类 |

允许复用先前合规运行保存的真实上游 artifact，以便隔离测试某一模块，但必须记录原运行目录、manifest 哈希、state identity 和 artifact 哈希，并验证它与当前测试 state 完全一致。复用 artifact 不等于复用上游模块的成功结论；本次没有运行的模块统一记为 `not_run`，不能记为成功。

纯 mock、合成数据和只验证类型/接口的测试仍然可以运行，但只能称为“单元测试”或“契约测试”，不能称为单点模块测试。

## 2. 真实示范状态的选取

1. 数据必须来自正式 R2R 数据集 episode，且必须存在非空 `reference_path`。
2. 测试状态必须直接取自 `reference_path[path_index]`，不能随机采样 Habitat 可导航点，不能用模型生成点，也不能为了结果好看手工移动起点。
3. 必须记录：
   - split；
   - episode index 和 episode ID；
   - scene ID；
   - trajectory ID；
   - path index；
   - reference position；
   - reference yaw 及其计算来源；
   - 完整原始 instruction；
   - 当前子指令及 stage ID。
4. 若数据提供该状态的真实旋转，必须直接使用真实旋转。若中间 `reference_path` 只有位置：
   - `path_index == 0` 使用 episode 的 `start_rotation`；
   - `path_index > 0` 使用 `reference_path[path_index-1] -> reference_path[path_index]` 的方向作为 yaw；
   - 必须在 manifest 中标记该 yaw 是由轨迹位置推导，而不是数据集原生旋转。
   - 这里的 `start_rotation` 指 manifest 里 `dataset_path` 所声明的那个数据集文件中的值。
     OpenNav100 端到端评测默认使用第 17 节定义的起始朝向对齐数据集，其 `start_rotation`
     是按第 17 节的 form 级规则离线构建的；运行时仍然只读该文件，不得在线再改。
5. 全模块单点测试优先选择内部状态 `1 <= path_index < len(reference_path)-1`，从而同时存在真实来向和真实下一段方向。若只能使用起点，必须明确标记 `initial_state_only`。
6. 状态选择必须在运行模型前由固定 seed 和确定性采样规则完成。禁止根据 VLM 输出或运行结果事后换状态。

## 3. 子指令和示范进度对齐

1. 测试从完整 R2R instruction 开始，保留完整拆解结果。
2. 当前子指令与 `path_index` 的对齐方法必须写入 manifest。R2R 没有原生短语-路径对齐时，允许使用预先固定的单调对齐规则，但不能在看到选点结果后调整。
3. 若使用按比例对齐，必须保存公式、stage 区间及该状态对应的 stage ID。
4. 对齐不确定的样本不能静默删除；必须标记 `alignment_uncertain` 并单独统计。
5. 单点选点正确率必须说明测试的是：
   - 给定固定子指令后的选点；还是
   - 包含 VLM 指令拆解误差的端到端选点。

## 4. 严禁示范信息泄漏

示范轨迹只能用于状态初始化和模型运行结束后的评分，不得作为选点提示或候选排序输入。

禁止向 VLM、DINO+SAM 候选排序器或导航策略提供：

- 下一示范 waypoint；
- 下一段示范 heading；
- 未来 reference path；
- 到示范轨迹的距离；
- 正确视图编号或正确点坐标；
- `demonstration_progress`、`path_index` 等会暗示示范进度的伪 action history。

允许提供的是在线机器人真实可获得的信息：当前 RGB-D、当前朝向、已执行 action history、前一节点、来向、blocked directions、当前子指令及历史节点记忆。

初始化在内部轨迹点时，可以用前一 reference state 构造“来向”，因为真实机器人在线执行时会知道来向；但该来源必须记录为 `reference_initialization_context`，且不得包含未来信息。

## 5. 各模块的执行规则

单点全模块测试必须执行本节全部模块。单点模块测试只执行 `target_modules` 及第 1.1 节规定的必要上游过程，但凡实际执行的模块都必须满足对应小节的完整规则，不能因为是模块测试而降低输入真实性、记录或评分标准。

### 5.1 指令拆解

- 输入完整 instruction。
- 保存每个子指令的文本、类型、landmark、语义空间目标、到达证据和禁止区域。
- 保存原始 VLM 返回和规则补全后的结果。
- 拆解失败时整个样本计失败，不能跳过后只测选点。

### 5.2 六视图和地面感知

- 必须从初始化后的真实 Habitat state 取得六视图 RGB-D。
- 六视图必须是相对当前 yaw 的 `0/60/120/180/240/300` 度。
- 正式地面 mask 必须使用 OneFormer ADE20K Swin-T、Mask2Former ADE20K Swin-S、SegFormer-B5 ADE20K 三个 RGB dense semantic 模型逐像素投票，至少 2/3 模型共同判为 floor/rug/carpet/ground/stairs/step/landing 等可支撑表面才保留。三个模型必须在 GPU 上运行；manifest 记录完整 checkpoint 和 `dense-majority` 后端名。Grounded-SAM、Transformers DINO+SAM 只可作为显式消融地面后端。
- 综合地面 mask 是选点的不可扩张物理合同：语义关系、物体避让和图像边界规则只能删去像素，不能加入综合 mask 外的像素。严禁在正式后端下用画面下方矩形、深度、navmesh 或其他未分割区域补地面。VLM 编号 anchor、VLM 返回坐标修复、可达性 repair、导航簇和停止簇的每个最终像素都必须重新核验位于该原始综合 mask 内。
- 指令相关物体检测与地面分割解耦：门、沙发等开放词汇语义仍可用 Grounded-SAM 或 DINO+SAM，但物体检测 mask 不得并入综合地面 mask。
- 普通物体分割默认关闭；指令相关的开放词汇 detection/SAM 可以使用。
- 必须保存六张原图、depth、地面 mask、语义检测、候选 mask 和候选拒绝原因。

### 5.2.1 选点与正确路径可视化（诊断专用）

- 每个真实测试案例必须保存一张/一帧包含所有候选视图的可视化：在当前
  observation 上叠加真实 R2R `reference_path` 的投影（青色）和最终 VLM
  选点（红色十字），并保留 `GT PATH` / `VLM POINT` 图例。
- 该投影只在模型完成选点后生成，用于人与结果的对照；`reference_path`、正确
  视图、正确像素和任何由投影计算的距离/heading 严禁进入 VLM、检测器或候选排序
  输入。JSON 审计必须同时保存每个候选视图的 `reference_path_projection`、最终
  `selected_pixel` 和图像文件路径。
- 运行时 Habitat 视频、VLM 选点 contact sheet、序列探索选点图和独立选点评测图
  均遵循同一颜色/坐标约定，确保能直接看出正确路径与实际选点的偏差；无
  `reference_path` 的纯接口/契约测试须标注 `path_projection_visualization=false`。

本项目当前 20-EP 选点回归还保留一个可选的关系感知版本
`v21_relation_aware_route_review`：它不假设首步一定向前，对
around/behind/diagonal/hallway-end/stairs 使用通用 RGB 路由约束，并可在
Grounded-SAM 的 0.15/0.10 弱地面框阈值下按指令形式自适应保留候选。该策略的
hallway 连续地面 tie-break 只使用二维 mask，不能读取深度、navmesh 或示范路径。
所有正式比较必须同时记录 `point_selection_prompt_version`、地面框/文本阈值和
`adaptive_floor_threshold`，不得把本回归的 20-EP 结果冒充 100-EP 总体结论。

### 5.3 VLM 选点

- VLM 必须使用当前可替换 backend 接口，默认 DeepSeek vision。
- 输入必须保存：完整文字 prompt、实际发送的 contact sheet、允许视图、编号地面 anchors、检测证据和真实历史 action。
- VLM 只能在允许视图中的编号地面 anchor 内选择，不能自由生成未经验证的像素。
- 返回必须经过 schema 校验；重试次数和每次错误都要记录。
- 最终点必须在候选地面 mask 内，并投影为有效深度和可导航三维点。

### 5.4 点导航执行器

- 从所选点构造固定水平对称 crop。
- 导航/goal 点簇与物理到达点簇必须分工：前者服务控制和动态 crop，
  后者服务“是否到达外部所选点”的判定，不能因为加密到达点簇而偷偷改变
  控制轨迹。
- 物理到达点簇必须是在有效地面 mask 上吸附的稠密底边点簇；默认实验同时
  保留独立的原底边 goal 簇和稠密 arrival 簇，并保存各自逐帧 visibility。
- 稠密 arrival 簇应使用独立的流式 tracker state，避免增加 query 数量改变原
  导航/goal 簇的跟踪与控制结果。
- 当前默认 `dense_stop_motion_guard_v7` 固定为三个角色：3x3 中心导航簇、3x3
  原底边 goal/crop 簇和 5x9（45点）稠密物理 arrival 簇。arrival 簇使用与
  控制簇共享权重但状态独立的 TAPIR。
- 常规消失需要 goal 簇消失且稠密 arrival 可见率不大于 4%，最多用四帧确认；
  若导航/goal/独立稠密 arrival 三簇同时终止，则允许终端共识，但仍必须满足
  累计真实移动量不小于初始选点深度的 70%。初始深度不大于 1.0 m 且选点位于
  图像底部 21% 的近场点可以在第0步直接到达。
- 投影目标在出发时没有 navmesh 路径时，执行器必须以
  `selected_point_unreachable` 拒绝执行，不能把后续遮挡当作到达。该入口
  可达性和初始 RGB-D 深度是在线输入，不是最终距离真值。
- 使用真实流式 TAPIR 和当前 GNM/ViNT/NoMaD 策略执行 Habitat 动作，不能 teleport 到目标。
- 保存每步 observation、crop、三个角色点簇、visibility、策略输出、实际动作、位姿和碰撞/移动距离。
- “是否到达选点”完全由 `PointNavigationExecutor` 负责，并通过
  `point_navigation_arrived` 返回给调用者；调用者不得用“是否完成指令”反向修改
  这个信号。
- 测试器必须把初始选点投影到 navmesh，并用最终测地距离 `<=0.75 m` 作为隐藏
  参考标签，分别报告到达信号的真到达、提前误报、漏报和未到达。这个隐藏距离只
  用于评测，不能输入执行器。
- 若点簇只是因转向离开画面而消失、但物理距离不满足要求，应记录为 `premature_offscreen_stop`，不能算导航成功。

### 5.5 节点和边

- 起始示范状态必须先建立 origin node。
- 收到 `point_navigation_arrived` 后必须建立新 node；这是接口/持久化功能约束，
  不作为“正确率”模块。执行器失败时可以额外保存 diagnostic stop node，但不能
  把它标成 arrival node。
- node 必须保存：六视图、环境语义、视觉 embedding、出发目的子指令、位姿、yaw、时间步和到达信号。
- edge 必须保存完整 action history、移动距离、起止节点和执行器停止原因。
- edge 必须保存从上个节点到本节点按时间均匀抽取的原始 RGB keyframe（默认
  首帧、末帧及至多三个中间帧），供指令完成判定使用。
- 测试必须核对 edge action history 与真实执行日志一致，不能使用预填充历史。

### 5.6 子指令完成判定

- 必须在到达选点并建立节点后重新采集六视图，再判断“上个节点到本节点的真实
  转移是否完成当前子指令”。
- 判定输入必须同时包含：上个节点六视图和环境语义、本节点六视图和环境语义、
  两节点真实在线位姿、入边完整 action history、入边按时间排序的 keyframe、
  当前子指令的语义空间目标和完成证据。
- 不得把 `departure_purpose_sub_instruction_id` 直接当作判定真值或形成必然匹配；测试判定时必须屏蔽会直接泄漏预期 ID 的字段。
- 自 2026-09-01 起，新实验恢复为两个语义输出：`completed` 和 `unknown`。这是
  对“上个节点 -> 当前节点”真实边事件的判定，不是只看当前节点的场景分类：
  - `completed`：前后全景、时序 keyframe 与真实 action history 共同证明该边
    完成了当前子指令的语义地标、空间目标或完成边界；
  - `unknown`：除 `completed` 外的所有情况，包括仍在路上、走错、反向、停滞、
    证据不足或语义歧义。本模块不再输出或考核 `on_route`。
- 单纯动作方向合理、移动距离较大、看见目标、到达点导航所选点，或处于示范轨迹
  中间状态，均不足以判为 `completed`。必须证明完整语义完成边界；否则统一为
  `unknown`。
- `unknown` 后继续探索、重选点还是回溯仍由外层策略决定，不得反向修改本模块
  标签或点导航到达信号。
- harness 可以把“有进展但未完成”作为内部证据用于校验完整边界，但模块对外不得
  返回 `on_route`；该内部状态必须在 `NodeTransitionInstructionCompletionJudge`
  边界折叠为 `unknown`，探索策略只能看到 `completed/unknown`。
- 预测结果必须与调用模型前冻结、且不暴露给模型的人工“边完成事件”语义标签
  比较，报告 completed/unknown、confidence、二类混淆矩阵、completed precision
  和 recall。
  把示范轨迹、未来 waypoint、path index 或评分标签输入 VLM 均属于结果泄漏；
  它们只能在模型调用结束后用于隐藏评分。
- 对“about halfway/part way”这类近似垂直范围，允许 harness 将 VLM 提取的
  图像证据与节点高度变化、前后楼梯检测和真实边移动量融合为操作性判定；仍
  禁止使用未来示范点。普通 top/bottom/landing 不得套用该例外。
- “到达选点”、节点里保存的出发目的、单纯看见地标或仅有动作距离，都不能
  单独作为“完成指令”的证据。

### 5.7 真实物理回溯

- 前向节点建立后必须执行一次回到 origin node 的真实回溯测试。
- 回溯选择器必须使用 origin node 保存的六视图、当前节点六视图、正向 action history 的反向上下文及图几何信息。
- 必须重新进行地面选点并调用同一个点导航执行器；禁止直接 `set_pose` 回到 origin。
- 必须保存回溯 VLM/混合选择记录、每步动作、回溯边和失败重试。
- 默认回溯成功条件同时要求：
  - 当前位置到 origin node 的平面距离不大于 0.75 m；
  - 六视图视觉相似度不低于 0.75；
  - graph 写入有效的 backtrack/revisit edge；
  - loop closure 指向实际重访的逻辑节点。

### 5.8 序列恢复和方向 block

- 单点全模块测试必须至少执行一次节点序列判定并记录策略 directive。
- 若自然触发 off-sequence，必须继续执行“一次额外探索 -> 再错则回溯 -> block 错误方向”的真实分支。
- 若当前样本自然判定正确，仍必须完成第 5.7 节的显式物理回溯模块测试；但不得伪造错误分类来宣称序列恢复成功。
- 自然未触发的 off-sequence 分支应另用真实示范状态专项测试，不能仅凭 mock 单元测试宣称系统级通过。
- `unknown` 包含途中和走错；之后如何探索或回溯属于整体探索策略，不得混入本
  模块准确率。

## 5.9 严格轮次推进闭环（2026-09-04 起强制）

后续十 EP 导航优化必须按子指令逐段推进，不得用“最终 R2R 距离变小”替代闭环
验收。每个子指令都必须在真实示范状态/真实在线节点上按以下顺序执行，并在当前
轮次产物中保存对应证据：

```text
人工观察六视图 + 完整 instruction/拆解 + 示范轨迹（仅用于人工基准/事后评分）
  -> 人工确定当前视角下最佳地面/楼梯地面目标及理由
  -> 系统 VLM/检测/地面候选选点，和人工目标比较视角、像素和语义接近度
  -> 只有系统选点足够接近才进入点导航
  -> 点跟踪导航真实移动；人工根据 RGB、动作历史、在线位姿复核是否到达所选点
  -> 若未到达，先只优化执行器并在同一真实 state 重测
  -> 到达后人工复核该点是否是当前可见信息下的最佳可执行终点
  -> 若不是，回到选点轮次；若是，写入 node/edge/action history/keyframes
  -> 调用系统 completion judge，禁止人工结论直接替代系统判定
  -> 系统判定与人工/示范事后审计一致后，才冻结该子指令并进入下一段
  -> unknown 时人工标记走错/在路上（仅供外层策略），走错验证真实回溯后回到选点；
     在路上继续当前子指令；完成则进入下一子指令
```

### 闭环冻结条件

1. 选点必须在对应地面/楼梯地面 mask 内；系统最终 ray 与人工基准在记录的角度阈值
   内，且不能只因“能走”而通过。人工基准、系统候选和诊断投影必须同时保存。
2. 点导航必须由 `PointNavigationExecutor` 产生 `point_navigation_arrived`，并由
   在线 RGB/跟踪/位姿证据证明不是提前丢点；人工复核和隐藏的最终测地距离标签只
   用于验收，不能输入执行器。
3. 到达节点后必须重新采集六视图并写入节点/边；节点持久化是 invariant，但缺失
   node/edge/action history/keyframe 时该子指令不能冻结。
4. completion judge 只能判断“上个节点到当前节点”的真实边，输出
   `completed`/`unknown`；系统输出必须和人工观察及示范轨迹的事后语义审计一致。
5. 任何优化只能抽象为 instruction form、空间关系、RGB/2-D 地面证据、动作历史、
   点簇几何或时序一致性规则。禁止 episode/scene/trajectory/固定像素/固定视图的
   条件分支，禁止为了通过当前案例降低判定标准。
6. 已冻结前缀必须在每轮新改动后回归；前缀任何一项退化则该候选不能冻结。十 EP
   的最终通过要求每个适用子指令都完成上述闭环，且系统自身没有遗留 unknown、
   错误回溯终止或人工接管。

详细的每轮字段、人工基准和系统证据模板见
`docs/strict_curriculum_loop.md`。

## 6. 成功判定和正确率

每个样本必须分别报告目标模块及本次实际执行模块的结果，禁止只给一个总 success。未执行模块必须明确记为 `not_run`，不得记作成功：

1. `decomposition_valid`；
2. `ground_candidate_valid`；
3. `point_direction_correct`；
4. `point_on_ground`；
5. `point_depth_valid`；
6. `point_target_arrival_signal`（执行器预测）；
7. `point_target_reference_reached`（隐藏测地参考标签）；
8. `point_target_arrival_correct`；
9. `node_and_edge_persisted`（功能性 invariant，不汇报准确率）；
10. `edge_action_history_and_keyframes_valid`（功能性 invariant）；
11. `instruction_completion_correct`（`completed/unknown` 二分类，仅在到达选点
    并建立节点后计分）；
12. `backtrack_selection_valid`；
13. `backtrack_physical_revisit`；

## 7. 十 EP 从起点开始的轮次推进协议（严格验收）

本节是当前端到端优化任务的专用协议，适用于固定的十个 R2R episode：
`0, 3, 6, 9, 18, 27, 45, 126, 204, 219`。每个 episode 必须从数据集的
`start_position/start_rotation` 开始，不能从旧的 stage node 直接续跑来替代本
协议。旧 stage 结果只能作为回归对照，不能作为本轮已经执行的步骤。

### 7.1 一轮的定义

一轮只推进一个当前子指令。一个 episode 的完整过程是：

```text
episode 起点/上一轮已验收 node
  -> 读取完整 instruction 和已有拆解
  -> 采集当前六视图（必要时八视图辅助观察）
  -> 人工基准观察：确定当前子指令下最优合法地面目标
  -> 系统 VLM/seg 选点
  -> 人工基准与系统选点比较，未接近则只优化通用选点模块
  -> 点导航执行
  -> 人工根据 RGB、action history、位姿和目标点复核物理到点
  -> 当前点是否仍是该视角下的最佳目标复核
  -> 记录 node、edge、keyframe
  -> 调用系统节点-子指令完成判定
  -> 与人工观察和示范轨迹对照，未一致则只优化通用判定模块
  -> completed：进入下一子指令
  -> unknown 且人工判定在路上：在同一指令内继续选点
  -> unknown 且人工判定走错：验证真实回溯，回到上一 node 后重选
```

“人工基准”是本项目的诊断标签，不是给模型的输入。它必须在系统选点和
导航执行前冻结，保存观察依据、合法地面/楼梯地面候选、期望视角、目标像素
和相对方向。示范轨迹只用于初始化、人工复核和运行后评分，不能泄漏下一
waypoint、正确像素、path index 或由示范计算的目标方向给模型。

### 7.2 选点验收门

选点只有同时满足以下条件才算本轮 A 通过：

1. 系统点落在当前六/八视图确认的地面或楼梯地面 mask 内，并可投影为有效
   navmesh 点；
2. 系统选中的视图和人工基准视图在角度上接近，默认差值不大于 30°；若人工
   基准认为存在多个等价地面候选，必须保存等价集合，不能只挑对系统有利的
   一个像素；
3. 对 `PASS/APPROACH/ENTER/STOP/BETWEEN` 等语义目标，必须同时有目标或
   关系证据。没有目标证据时只能记录 `selection_rejected_no_semantic_evidence`，
   不得偷偷退化为普通 floor-only 点并计为通过；
4. 目标选择的判断必须基于当前可见信息和在线历史，不能使用示范 path 的
   未来位置或隐藏距离。

若 A 不通过，只修改形式级的选点规则、prompt、候选生成或视角请求策略；不
允许针对 episode、场景名称、固定像素或固定 waypoint 写分支。A 未通过时不
得把该点送入导航并将后续结果当作该轮通过。

### 7.3 物理到点验收门

只有 A 通过后才执行点导航。B 通过要求：

1. `PointNavigationExecutor` 返回 `point_navigation_arrived`；
2. 人工核对当前 RGB/跟踪簇位置、起止位姿、动作历史和停止原因，确认不是
   点簇因转向、遮挡或 crop 离开画面造成的假到达；
3. 到达后节点中的实际位置与系统选定点一致，且执行器没有提前停止、零动作
   到达或明显偏离选点；
4. 若 B 不通过，优先优化执行器的簇密度、遮挡处理、运动门禁和停止规则，
   不修改 A 的人工基准或把失败点重标成“最佳点”。

### 7.4 到点后的二次选点复核

B 通过后，人工必须重新观察到达点的全景，检查出发前的系统点是否确实是当时
可见条件下最有利于完成子指令的目标。如果当时虽然接近人工点，但到达后发现
它只是普通地面、目标地标未出现、或会导致指令方向不可执行，本轮仍不通过；
必须回到 A，改进通用规则后重测。不能用节点判定的结果反向证明选点正确。

### 7.5 节点/边和判定验收门

二次复核通过后必须建立新 node 和 edge。节点持久化是功能性 invariant，但
必须逐轮检查完整性。然后只能调用系统的
`NodeTransitionInstructionCompletionJudge` 判断当前 edge 是 `completed` 或
`unknown`。人工结论只用于运行后审计，不得替代系统返回值。

判定 C 通过要求系统返回与人工观察和示范阶段边界一致：

- 人工确认已经完成语义目标，系统必须返回 `completed`；
- 人工确认还在路上、走错、停滞或证据不足，系统必须返回 `unknown`；
- `unknown` 不得被自动解释成失败，必须由人工审计进一步标成“在路上”或
  “走错”，但该细分只供外层探索策略使用，不能改变判定模块对外的二分类接口。

若 C 不通过，只允许提取跨形式的通用证据问题（视角时序、portal crossing、
地标持续性、动作方向、节点结构化字段等）来优化判定；不得添加 EP/case 特判。

单段“完成”的统计必须同时满足两个相互独立的条件：一是导航系统确实在物理到点后
建立的新节点及其 incoming edge 上返回 `completed`；二是运行后使用冻结的人工语义
边标签，结合该节点六/八视图、上一节点视图、action history、keyframes、实际轨迹和
示范轨迹完成独立核验，并确认语义目标确实完成。二者缺一，该段均记为未完成。
系统的 `completed` 只能叫 `system_node_completed`，未经核验不得写入
`sub_instructions_completed`、冻结前缀或 EP 成功统计。核验信息不得回流给在线模型。

独立核验还必须设置 `ordered_stage_boundary_verified=true`：本段 incoming edge
既要完成当前语义目标，也不得提前跨过下一子指令的关键空间边界（例如“转身”阶段
直接穿过下一段要求的门）。如果当前目标直到后续 edge 才完成，不能倒算给当前段；
如果下一目标已在前一 edge 被越界执行，也不能把该旧事件补算给下一段。只有机器人
在当前段正确边界建立节点后，才允许开始下一段。这一字段缺失或为 false 时严格
fail-closed。

### 7.6 unknown 后的真实分支

1. 人工判定“在路上”：保留当前 node，继续同一子指令，新的选点必须重新通过
   7.2 的 A 门；不能把“继续走”写成无条件前进；
2. 人工判定“走错”：必须先执行并验收真实物理回溯（7.7），回到上一 node，
   block 已证明错误的方向/候选，再回到 A；
3. 人工与系统都确认 `completed`：才允许推进下一个子指令；
4. 如果人工与系统不一致，本轮不推进子指令，不得把不一致结果写入冻结前缀。

### 7.7 回溯验收门

回溯必须使用两个真实节点的六视图、edge action history 和在线图信息重新
选地面点并调用同一个点导航执行器，禁止 teleport。回到上一节点后要求位置、
视觉重访和 graph loop-closure 同时通过；失败时记录失败类型并先优化通用
回溯策略，不能直接恢复到旧 pose 冒充通过。

### 7.8 轮次记录和冻结规则

每个 episode/子指令必须写入独立的 `round_report.md` 和 `manifest.json`，至少
记录：人工基准、系统候选/拒绝原因、选点角度差、点导航人工复核、node/edge
完整性、系统判定、人工审计、示范 path 运行后对照、是否回溯及下一步 directive。

同一轮的 A、B、C 以及必要的回溯门全部通过后，才冻结该子指令前缀。下一轮
只能从已冻结 node 继续，不能重新跑或修改已通过的前缀。任何优化必须先写入
规则/版本并保留失败 artifact，不能覆盖旧结果。十个 episode 的最终通过条件
是：所有子指令按上述闭环完成，系统自身到达各自 R2R 终点并由系统判定完成；
仅有点到达、节点写入或人工认为“应该完成”都不算端到端通过。

### 7.9 终点命中统计口径（强制）

机器人进入 Habitat/R2R 标注终点半径，只能记录为
`goal_radius_hit_diagnostic`，它是用于排查轨迹的几何诊断量，不是成功指标。
只有从 episode 原始起点出发，完整指令拆解中的所有子指令都由系统按顺序完成
A/B/N/C 闭环、没有未解决的 `unknown`、没有人工接管，并最终进入终点半径时，
才允许记录 `instruction_validated_r2r_success=true` 并计入终点成功率。

系统在最后一个子指令的到达节点被在线判定为 `completed` 后，必须立即发出唯一的
Habitat 任务级 `STOP`，此后不得再执行移动。`simulator_reported_success` 的定义固定为
“主动发出该 STOP 且 STOP 时刻的最终 geodesic distance 位于 episode goal radius
内”；只进入半径但未主动 STOP，或尚未完成最后子指令就停止，均为 0。由于当前直接
集成使用 Habitat-Sim locomotion API，STOP 作为任务级终止动作写入 `task_stop`，并以
当时 pose 计算几何结果，不伪装成 Habitat-Sim 的移动 action。该模拟器指标与更严格的
`instruction_validated_r2r_success` 分开汇报，后者仍额外要求独立语义核验。

随机探索、走错分支、越序移动、未完成首段/中间段，或只运行了截断的前若干段后
偶然进入终点半径，均标记为 `invalid_goal_radius_hit=true`。这类样本在正式成功数中
恒计为 0，不得以“终点到达”“Habitat success”或其他名称另行计入成功率；它只能
进入失败分析。正式汇报中的“到达终点”默认且只能指
`instruction_validated_r2r_success`。
14. `loop_closure_valid`；
15. `all_modules_success`（仅单点全模块测试可填写布尔值；单点模块测试必须为 `not_applicable`）。

选点主正确率沿用项目约定：所选可导航三维点相对当前状态的 heading，与示范下一段 heading 的夹角 `<90°`。同时必须报告 `<30°/<45°/<60°`。

准确率分母必须是所有预先选定的合规样本。API失败、无候选、无效深度、导航失败、判定失败和回溯失败均计入对应模块的错误，不能只在 valid 样本上计算主正确率。必须同时报告样本数和95%二项置信区间。

单个状态只能叫 case result，不能据此声称“正确率”。正确率必须来自预先确定的一组真实示范状态。

## 7. 强制输出

每个单点测试目录都必须包含：

```text
manifest.json
module_results.json
```

单点模块测试还必须保存目标模块及必要上游过程对应的所有原始输入、中间结果和评分证据。以下完整清单对单点全模块测试全部强制；单点模块测试只要求其中与实际执行模块相关的项目，未运行项写入 `manifest.json:not_run_modules`：

```text
manifest.json
instruction_decomposition.json
initial_six_views/
initial_depths/
ground_masks/
semantic_detections.json
vlm_prompt.txt
vlm_contact_sheet.jpg
vlm_response.json
point_selection.json
goal_crops/
forward_action_history.json
arrival_six_views/
instruction_completion.json
navigation_graph/navigation_graph.json
backtrack_selection.json
backtrack_action_history.json
trajectory.json
module_results.json
exploration.mp4
```

凡目标模块包含视觉决策、导航、节点建立或回溯，必须保存视频。视频继续使用固定三画面布局：左侧实时 observation，右上黑底白字 instruction/当前阶段/模块状态，右下实时 top-down。选点阶段必须显示六视图、地面 mask、允许/排除状态、编号 anchor 和最终选择；前向及回溯阶段必须显示导航/goal/稠密 arrival 三个角色点簇、crop、动作、可见率和目标节点。纯指令拆解模块测试不强制生成视频。

## 8. 可复现性

- 保存随机 seed、代码版本或文件哈希、模型名/权重、DeepSeek配置、DINO+SAM阈值、Habitat步长和转角。
- 同一测试 manifest 必须能够原样重跑。
- 模型温度、thinking模式、重试次数和任何 fallback 必须写入结果。
- 运行期间发生的异常不得覆盖原始输出；失败样本也必须保留完整日志。

## 9. 命名边界

- `tests/test_*.py` 中使用 mock 或合成数据的测试叫“单元测试”或“契约测试”。
- 从真实示范 state 初始化、满足第 1.1 节并明确指定目标模块的测试，叫“单点模块测试”，例如“单点 VLM 选点模块测试”或“单点回溯模块测试”。
- `evaluate_r2r_point_selection.py` 若满足本规则，可叫“单点 VLM 选点模块测试”或“真实示范状态选点消融”，但不能叫“单点全模块测试”。
- 只执行前向点导航、不执行节点判定和物理回溯的实验可叫“单点导航模块测试”或“单点前向测试”。
- 只有完整满足本文件第1节和第5节的实验才能叫“单点全模块测试”。

## 10. 对现有结果的说明

本规则写入前生成的结果继续保留，但不得按新规则重新命名：

- `outputs/deepseek_point_accuracy_100` 是真实示范状态上的选点消融，不能代表导航、节点判定或回溯正确率。
- `outputs/deepseek_smoke_ep0_14` 是单点前向测试，没有执行真实物理回溯，不是全模块测试。
- 当前43个 mock/合成输入测试是模块契约测试，不是系统级单点正确率。

从本文件生效后，所有新单点系统实验必须按本规则生成 manifest 和目标模块结果；只有目标包含回溯或测试层级为 `full` 时才强制提供完整回溯证据。

## 11. 100-case 全模块批量评测冻结协议（2026-08-30）

本节只描述历史 100-case 冻结实验，不得覆盖第5.4--5.6节的新模块边界。
其中旧 `belongs_to_sequence/off-sequence` 指标不能作为当前边状态模块的正确率。
本节所述 `completed/unknown` 仅保留为历史 100-case 协议；2026-08-31 之后的新
2026-08-31 的三分类协议仅保留为历史结果；2026-09-01 之后的新实验必须遵循
第 5.6 节的 `completed/unknown` 人工边事件标签。

本协议用于 `evaluate_r2r_all_modules.py`，在任何模型调用前固定，运行中及
看到结果后不得修改。样本必须与已保存的 V3 选点评测
`outputs/vlm_point_selection_100ep_v3_soft_semantic/sample_manifest.json`
身份一致：100 个不同的 R2R `val_unseen` episode，每个 episode 一个内部
`reference_path` state，固定 seed 17。

每个样本真实执行：DeepSeek 指令拆解、Habitat 六视图 RGB-D、DINO+SAM、
V3 选点、水平对称 crop、双 TAPIR 点簇、GNM 控制、停止节点与真实 edge、VLM
节点序列判定、自然序列恢复状态机，以及使用新旧节点六视图的 VLM 物理回溯。
每次前向/回溯点导航最多 30 个控制步；序列探索最多 3 跳；回溯目标固定为
本次真实 state 建立的 `node_0000`，禁止 teleport。

新增并冻结以下评分细节：

1. `decomposition_valid` 只表示可执行拆解有效性，不冒充人工语义准确率：至少
   一个子指令，且每段的文本、类型、semantic spatial target、spatial relation、
   visual arrival evidence 和 forbidden target 均非空。
2. `ground_candidate_valid` 要求六视图完整、至少一个 DINO+SAM 地面实例、
   至少一个非空地面候选，并最终产生地面 anchor。它是可用候选率，不是像素
   IoU；R2R/MP3D 当前没有与该开放词汇地面查询逐像素完全等价的真值。
3. VLM 选点继续用与下一示范段 heading 夹角 `<90°` 为主正确率，并同时报告
   `<30°/<45°/<60°`。完整链路结果包含实时 DeepSeek 拆解误差；既有 81/100
   结果是给定预先固定子指令后的隔离选点率，两者不得混称。
4. 点导航物理成功严格要求执行器内部 arrival 信号成立且最终到所选 navmesh
   目标的测地距离不大于 0.75 m。只因点簇转出视野而停止仍为失败。
5. 节点分类的事后真值不得输入模型。对每一跳，先从 graph edge 得到真实起止
   state；`ground_truth_belongs` 同时要求：选点相对局部示范下一段方向 `<90°`、
   实际位移沿该方向投影至少 0.20 m、停止点到当前及未来示范 polyline 的 XZ
   距离不大于 3.0 m。预测为正时还必须 matched ID 等于运行前固定的 expected
   ID 且 confidence 不低于 0.5，之后与该真值比较。
6. 回溯必须是非平凡节点对：显式回溯开始节点到 origin 的 XZ 距离至少 1.0 m。
   主回溯率分母仍是全部 100 个预选样本；未形成有效节点对计失败。同时另报
   有效节点对上的条件成功率。选择正确要求选中 yaw 与 origin bearing 夹角
   `<90°`；物理重访仍同时要求 0.75 m 位置阈值、0.75 六视图相似度、真实
   backtrack edge 和 loop closure。
7. 序列恢复不伪造 off-sequence。根据第5条逐跳构造隐藏真值状态机：真值正例
   应 advance/complete；第一处真值负例应 `explore_once_more`；连续第二处真值
   负例应 `backtrack_and_block`。`sequence_recovery_policy_correct` 要求所有实际
   directive 与该隐藏状态机一致，并在出现真实恢复机会时完成物理回溯且写入
   对应 blocked yaw。没有恢复机会的样本只要不误触发且 directive 全部正确可
   通过策略级评分；另外单列真实恢复机会上的条件成功率。
8. API失败、进程失败、缺文件、无候选或上游失败均保留在100样本主分母中。
   每项报告成功数、总数、成功率、Wilson 95% 区间和失败原因。crop、双点簇、
   TAPIR 和 image-goal policy 作为执行器内部模块也分别统计，不用契约测试结果
   代替真实 episode 成功率。

## 12. 子指令完成判定 holdout-20 冻结验证（2026-09-01）

`outputs/instruction_completion_holdout20_v13_20260901` 是第 5.6 节二分类判定器的
独立专项验证集。它在任何 VLM 调用前固定 10 条与既有 30 例开发集 episode 和
trajectory 均不重叠的 R2R `val_unseen` 示范轨迹，并从每条轨迹构造一个完整
正向 `completed` 边和一个 `unknown` 边；后者固定交替使用完整反向边或只到
首个内部状态的途中边。因此共有 20 个 case、10 个正例和 10 个负例。

所有 case 必须使用真实 Habitat replay 产生两节点一边的图，包含实际 action
history、按时间排序的 keyframe、两端 8 视图、DINO+SAM 语义、深度和视觉
embedding。标签、case category、reference path index、未来 waypoint 和人工
标签理由不得进入模型输入。全部构建产物和 prompt 必须在首次模型调用前通过
`pre_vlm_leakage_audit.json` 审计。

v13 在该冻结集上的一次性结果为 17/20（85.0%），completed F1 84.21%，unknown
F1 85.71%，macro-F1 84.96%；完整正向边 8/10、完整反向边 5/5、途中边 4/5，
API/schema 失败 0/20。该数值只代表子指令完成判定模块，不代表选点、点导航或
全系统端到端成功率。

本次结果产生后，holdout-20 不得继续用于 v13 的 prompt、阈值、规则或 case 级
修补。若后续 v14 使用这 20 例分析或调参，它们必须明确改名为开发集；v14 的
最终准确率必须在另一组预先冻结、轨迹不重叠且未查看结果的新测试集上报告。
详细协议、误差分析和统计限制见同目录 `REPORT.md`。

## 13. 正式导航回归前的三个困难单点门槛（2026-09-01）

在再次运行 episode indices `12,21,30,48,75` 的正式端到端导航前，必须先分别
完成语义选点、物理到达判定和在线节点边完成判定的困难单点通用优化。上述五个
任务 episode 必须从所有调参集排除，不能用其新一轮结果选择 prompt、阈值或规则。

三个模块都必须使用真实 R2R episode/reference state 和 Habitat RGB-D；示范未来
信息只可在模型运行结束后用于隐藏评分。任何优化必须是按指令形式或几何失效模式
定义的通用规则，禁止按 episode、scene、case、具体物体名或期望结果修补。

1. **语义选点/地面候选 hard-30**：固定 seed 31，从 30 个不同 `val_unseen`
   episode 的 `reference_path[0]` 和第一个按定义拆解的子指令构造，按指令形式
   分层；排除正式五任务。主指标为所选可导航三维地面点相对首段示范 heading
   误差 `<90°`，通过线为至少 24/30（80%）；同时要求有效选择、地面吸附、有效
   深度与 navmesh 投影均为 30/30，并报告 `<30°/<45°/<60°`。
2. **物理到达 hard-30**：复用 hard-30 已冻结的真实 state、地面 mask 和选点，
   在 Habitat 中实际执行 GNM+流式 tracker。隐藏真值仍为最终到选点的测地距离
   `<=0.75 m`。通过线为 arrival accuracy、precision 均不低于 90%，recall 不低于
   85%，且 30 例中 `premature_offscreen_stop` 不超过 3；不可达点必须在执行前
   拒绝并计 TN。
3. **在线节点边完成判定 hard-30**：10 个完整正向完成边、10 个方向/区域错误边、
   10 个未到语义边界的途中边。每条边必须从真实示范 state 出发，在 Habitat 中
   产生实际 action history、keyframe 和双端 8 视图；人工语义事件标签在任何
   completion VLM 调用前冻结。通过线为 accuracy 和 macro-F1 均不低于 80%，且
   completed precision、recall 均不低于 80%。

三组调参完成后还必须运行既有 100 例选点集、34 例 arrival 集、20 例 completion
holdout 和 breadcrumb 回溯契约作为回归保护。只有新 hard-30 达标且既有保留集
不发生超过 3 个百分点的主指标下降，才允许重新运行正式五任务；否则继续留在
对应单点模块迭代，不得用五任务结果调参。

### 13.1 本轮冻结结果与正式五任务（2026-09-01）

本轮所有规则均按指令形式或几何失效模式实现，没有 episode/scene/case/具体物体名
分支。候选版本即使在 hard-30 达标，只要回归集失败也不得成为默认：V6 在 hard-30
为 25/30、100 例为 74/100，V8 为 25/30、77/100，均被拒绝；V9 单阶段双图为
21/30，未进入 100 例。最终选点 `v10_approach_relation_router` 只对单地标接近使用
两阶段，`BETWEEN_OBJECTS/CIRCUMNAVIGATE` 保留跨视图单阶段比较；它在 hard-30 为
24/30（80%），在既有 100 例为 78/100，相对 V3 的 81/100 恰好下降 3 个百分点，
按预先阈值边界通过。两组有效选择、地面吸附、有效深度和 navmesh 投影均为全数
通过。该结论必须表述为“边界通过”，不得表述为显著优于 V3。

物理到达实际使用另一个在模型调用前冻结的、轨迹去重 hard-30：它从既有真实
all-modules 运行中按初始目标测地深度分层抽取 10/10/10，并排除既往到达调参轨迹
和正式五任务。它没有复用选点 hard-30，因此是本节第2项原计划的明确协议偏差，
报告时不得隐去或把两组样本说成相同。V7 基线为 TP=20、FP=3、FN=3、TN=4；
`dense_stop_geodesic_guard_v9` 为 TP=23、FP=2、FN=0、TN=5，accuracy 93.33%、
precision 92%、recall 100%。V9 在既有 34 例回归集为 34/34。其证据目录分别为
`outputs/point_arrival_hard30_v9_round1_20260901` 和
`outputs/point_arrival_v9_regression34_20260901`。

在线节点边完成判定继续采用 `v13_structured_node_edge_binary`：冻结 30 例为 26/30
（TP=8、FP=2、FN=2、TN=18，accuracy 86.67%、completed precision/recall 均
80%、macro-F1 85%）；轨迹不重叠 holdout-20 为 17/20、macro-F1 84.96%。V14
虽在开发 30 例达到 27/30，但 holdout-20 只有 16/20、completed recall 60%，因此
被拒绝且不得设为默认。breadcrumb 与全部模块契约共 74 项单元/契约测试通过。

满足上述门禁后，正式固定 EP `12,21,30,48,75` 使用 V10 选点、V9 物理到达、
V13 节点边完成和 `breadcrumb_budget_v3` 回溯，从 R2R 数据集起点运行。输出目录为
`outputs/r2r_formal_navigation_5ep_v10_qualified_20260901`。结果为 R2R success 0/5、
SPL 0、平均测地进展 +1.0137 m；25 次目标中隐藏真实到点 20 次，到达混淆
TP=19、FP=2、FN=1、TN=3（accuracy 88%、precision 90.48%、recall 95%）；
18 个子指令完成 6 个，内部序列完成 1/5，恢复回溯 11/13。EP21 虽内部 4/4 完成，
但最终仍距真实终点 8.59 m，因此本轮主要瓶颈是在线语义序列闭环及其假完成，
其次是无地面候选（2 EP）和回溯失败（2 EP），而非点执行器本身。

## 14. 八视角纯 RGB 严格选点协议（2026-09-01）

自本节生效后，新选点主指标从 `<90°` 改为：最终所选地面像素反投影得到的三维
heading 与当前 R2R 示范下一段 heading 的误差 **不超过 45° (`<=45°`)**。旧
`<90°` 仅保留为兼容诊断指标，不得再作为选点通过条件。

选点输入固定为八个同时采集的 RGB 视角，相对当前朝向依次为
`0,+45,+90,+135,180,-135,-90,-45°`。DINO+SAM 地面/地毯 mask、语义检测、
候选关系约束、跨视角比较、VLM prompt、anchor 排序和精细视角请求必须全部为
纯 RGB/二维图像信息：

- 不得向 DINO+SAM、语义检测或关系候选生成传入 depth；
- 不得向 VLM 暴露 depth 数值、depth 统计、3D 坐标、navmesh、示范 path 或评分；
- 指令拆解里的旧 `depth` perception/rank 字段必须在 VLM 选点接口处覆盖；
- depth 只能在 VLM 已冻结最终像素后，用于相机反投影、navmesh 投影、实际执行和
  隐藏事后评分，且不得反向改变视图或 anchor。

VLM 第一轮必须返回 `use_existing` 或 `request_refined`。请求精细视角时同时返回
最佳现有 fallback view 和期望相对 yaw；精细 yaw 与 fallback 中心的差不得超过
30°，否则按非法非局部重选拒绝。Habitat 在原位置采集该 yaw 的新 RGB（depth
仍只封存供最终像素确定后的投影），重新执行纯 RGB DINO+SAM/检测。第二轮只在
确认的现有或精细视角内选择编号地面 anchor；优先画面中央 40% 内、沿可见路线的
连通地面，避免正确相机扇区被边缘像素拉成大于45°的三维 heading。

在查看正式结果前冻结两道门槛：

1. 继续使用排除正式 EP `12,21,30,48,75` 的 initial-state hard-30 身份清单；
   `<=45°` 至少 21/30（70%），有效选择、地面吸附和 navmesh 投影均须 30/30。
2. 使用既有轨迹 state 100-case 身份清单回归；`<=45°` 至少 65/100。旧六视图
   V3 在同一身份清单为 62/100，因此新方法必须产生实际提升而不是仅不退化。

失败选择、API/schema 失败、无地面 anchor 均进入主分母。困难 30 和 100 例完成
前不得查看或用正式五 EP 的新八视图结果调 prompt、门禁或阈值。

### 14.1 本轮候选结果与深度隔离结论（2026-09-01）

本轮所有新选点调用均实施代码级隔离，而不只依赖 prompt：传给
`select_ground_target` 的候选字典不得含 `depth`，否则立即报错；Habitat/evaluator
把深度保存在 VLM 候选之外，只能在 `view_index + pixel` 已冻结后读取。所有 prompt
版本均覆盖分类表里遗留的 depth 策略文字，并只暴露 RGB、DINO+SAM 二维 mask、
二维检测和 anchor。新 manifest 必须记录 `rgb_only_depth_prohibited`。隐藏深度只能
用于冻结后反投影和事后评分，不能换视角、换 anchor 或重试。

在相同的 initial-state hard-30 身份清单上，八视角候选结果如下：

- V11 `v11_eight_view_refinement`：`<=45°` 17/30（56.67%），有效选择 29/30；
- V12 `v12_eight_view_dual_candidate`：`<=45°` 15/30（50.00%），有效选择 30/30；
- V13 `v13_eight_view_native_rgb`：八张独立 RGB + clean panorama + 二维标注 panorama，
  复合路线使用“当前转移与首个可见后续地标同/相邻视区”的通用一致性规则；结果
  `<=45°` 18/30（60.00%，Wilson 95% CI 42.32%--75.41%），有效选择 29/30，
  `<30°` 17/30、`<=60°` 20/30、兼容 `<90°` 23/30。

V13 相对旧六视图 V3 的 14/30 和 V11 的 17/30 有数值提升，但低于冻结的 21/30
门槛；唯一无效例为最终冻结像素的隐藏深度无效，按规则不得预看深度换点。其余
主要失败是 VLM 在多门/多区域场景选择了语义合理但与示范方向不同的分支，以及
`CROSS_SPACE/PASS_LANDMARK/APPROACH_LANDMARK/FOLLOW_PATH_BOUNDARY` 等关系方向偏差。
V13 的 30 例中 VLM 没有主动请求精细视角，因此“可二次取景”接口已实现但尚未
产生实测收益。

按冻结协议，V11/V12/V13 均不得成为默认策略，V13 不进入 100-case 回归，也不得
据此重跑正式五 EP。完整证据目录为
`outputs/point_selection_hard30_initial_v13_native_rgb_sequence_20260901`；本轮全部
82 项单元/契约回归通过。

### 14.2 V14/V15 纯 RGB 优化、困难集通过与 100-case 拒绝（2026-09-01）

本轮继续遵守第14节的纯 RGB/二维输入边界，未使用 episode、scene、case、示范
答案或具体期望方向作分支。V14 `v14_rgb_evidence_refinement` 实施以下通用修改：

- 过滤覆盖大部分画面或把动作短语当物体标签的开放词汇检测；
- 对 exit/enter/through/down/pass/follow/approach 形成紧凑二维检测证据计划；
- 显式左/右/掉头使用更窄的承诺扇区；
- 目标物或候选地面未居中时，最多请求一次相对 fallback 不超过30度的新 RGB 视角；
- 最终 anchor 优先画面中央40%的 DINO+SAM 地面。

V14 在冻结 initial-state hard-30 上达到 22/30（73.33%），但中央40%被误作硬
有效性条件，导致4例在仍有合法地面时被判无点，有效选择仅26/30，因此未通过
覆盖门槛。V15 `v15_rgb_center_preferred` 将其改成“中央优先”：若一次局部新视角
后中央仍无 mask 支持，则在同一已选 DINO+SAM 地面 mask 中保留最靠近水平
中轴的合法 anchor；不得借助隐藏 depth 换视角或换点。同时修复精确30度经三角
运算表示成 `30.000000000000018` 时被误拒绝的浮点边界。

V15 最终 hard-30 复测证据为
`outputs/point_selection_hard30_initial_v15_rgb_center_preferred_tolerance_20260901`：

- `<=45°` 25/30（83.33%，Wilson 95% CI 66.44%--92.66%）；
- `<30°` 23/30，`<=60°` 26/30，兼容 `<90°` 28/30；
- 有效选择、地面吸附和 navmesh 投影均为30/30；
- 与 V13 的18/30相比净增7例（8例由失败变成功、1例由成功变失败），与旧 V3
  的14/30相比增11例；
- 规则触发6次局部取景、接受2次；VLM原生主动请求0次。中央 fallback 使用4次，
  其中3次命中。剩余5个失败分别来自 `CROSS_SPACE` 77.9度、
  `ADVANCE_STRAIGHT` 135.1度、`APPROACH_LANDMARK` 60.2度、
  `CIRCUMNAVIGATE` 93.3度和 `TURN_AROUND` 53.3度。

V15 因此只通过第14节第一道困难集门槛。随后在与旧 V3 完全相同、manifest 哈希
一致的100个内部轨迹 state 上运行冻结回归，证据目录为
`outputs/vlm_point_selection_100ep_v15_rgb_center_preferred_20260901`。结果为
`<=45°` 53/100（53.0%，Wilson 95% CI 43.29%--62.49%），有效选择99/100；旧
V3 同清单为62/100。V15 未达到65/100门槛且回退9个百分点，必须拒绝为默认，
不得据此重跑正式五 EP。

100-case 中，28次规则触发局部取景，16次获得合法新 RGB 视角，但接受新视角的
16例仅6例最终 `<=45°`；VLM自身主动请求仍为0次。该统计只说明接口/触发器的
覆盖和相关性，因没有对同一模型响应运行冻结 fallback 反事实，不能声称新视角
造成了6次成功。形式级最弱项为 `PASS_LANDMARK` 和 `TURN_RIGHT`（各1/6）、
`CIRCUMNAVIGATE` 与 `VERTICAL_DOWN`（各1/5）；最稳为 `TURN_AROUND`、
`VERTICAL_UP`（各5/5），其次为 `APPROACH_LANDMARK`（5/6）。47个主失败包含
21个45--90度相邻射线错误、25个不小于90度的错误分支和1个最终冻结像素的隐藏
depth无效。主要问题是内部 state 的具体分支/关系选向泛化，而非地面候选覆盖。

两组运行的 prompt 审计均未出现 depth数值、depth统计、3D/navmesh或示范路径；
manifest记录 `rgb_only_depth_prohibited`。depth仅在 view+pixel 冻结后用于隐藏
反投影、navmesh校验和评分。当前全部86项单元/契约测试通过。正式系统默认选点
版本保持不变。

### 14.3 左右转三视角硬门（2026-09-01）

左右转向在八视角语义复核之前显式限制方向候选：`TURN_LEFT` 的扇区槽位固定为
`+45/+90/+135`度，`TURN_RIGHT` 固定为 `-45/-90/-135`度；VLM只能在对应一侧
且具有 DINO+SAM 地面 anchor 的视角中做语义排序，后续 anchor 阶段不得重新
打开正前、反侧或正后方向。`TURN_AROUND` 继续使用 `+135/180/-135`度后三视角。

直接实现该结构的 V16 `v16_turn_three_view_gate` 在相同 hard-30 上为23/30
（76.67%），有效选择29/30。两个右转仍为2/2，但两个裸 `Turn left` 分别因选择
浅转 `+45`度视角导致隐藏深度无效和74.5度方位误差，左转为0/2。该结果说明三
视角硬门本身不足以阻止 VLM 把视觉上更开阔的斜前方误当成已经完成转弯的路线。

V17 `v17_turn_three_view_commit_prior` 保留三视角硬门，并增加通用的裸转向语义：
只有指令不含目的地/地标、形式为单独 `turn left/right` 时，`+/-45`度被解释为
浅 veer；若 `+/-90`或`+/-135`度存在地面候选，应优先能实际承诺转弯的视角。
带 `into the hall`、门或其他目的地的转向仍由 VLM 在同侧三视角内根据 RGB
语义自由选择。校验器保留相同的通用 fallback，但冻结 hard-30 中 VLM 在 prompt
约束下已直接选择承诺视角，事后强制调整实际为0次。

V17 证据目录为
`outputs/point_selection_hard30_initial_v17_turn_three_view_commit_prior_20260901`：
总结果25/30（83.33%），有效选择、地面吸附和 navmesh 投影30/30；两个左转为
24.2度、13.7度，两个右转为19.2度、10.3度，左右转合计4/4。它恢复了 V15 的
hard-30总指标，同时满足本节的三侧视角结构。V17尚未通过第14节100-case回归门，
不得设为正式默认或用于重跑正式五 EP。当前全部88项单元/契约测试通过。

## 15. 真实 failure case 通用优化复测（2026-09-02）

本轮只从既有真实 R2R 运行记录提取失效模式；不得读取隐藏示范方向后修改单例，
不得添加 episode/scene/case/物体名分支。选点仍严格执行 RGB-only 决策边界：深度
只在 view+pixel 冻结后用于反投影和隐藏评分。

### 15.1 选点：历史方向贯穿到精细视角与最终像素射线

100-case V15 的六个折返错误暴露出两个结构漏洞：局部精细视角只检查
`hard_excluded` 而未检查软来路排除；相机中心可在50度排除区外，但图像边缘 anchor
的最终射线可重新滑入来路。另有真实导航 EP48 把厨房台面误作地面、EP75 把抬高
表面/阶面误作可走区域，说明 DINO+SAM 绿色 mask 只能作为候选，不能当作 RGB
真实性证明。

V18/V19 实施以下通用规则：

- 精细视角不得重新进入 incoming/blocked/显式方向门禁；检测证据只能在距合法
  地面视角45度以内触发局部取景；
- VLM 的复核和最终 anchor prompt 必须用 clean RGB 否决台面、桌面、床、沙发座面、
  墙、架子和抬高物体表面的伪地面；
- V19 在 anchor 生成阶段用二维像素 bearing 计算最终 RGB 射线，在普通指令中删除
  重新进入来路或 blocked 扇区的 anchor；不使用 depth；
- 显式 `TURN_AROUND` 是唯一允许覆盖软来路排除的形式，但仍不得覆盖 sequence
  blocked 硬门；左右转继续固定在各自 `45/90/135` 三视角。

相同 manifest（SHA256
`6be1b1964ef9366567e7db3a9a24ddc82318ab4473fa5f59019af821e0582f6f`）的正式
100-case 对照如下：

- V15：53/100，valid 99/100，中位 heading error 31.13度；普通指令折返6次；
- V18：54/100，valid 92/100，中位 error 24.00度；仍有4次普通折返；
- V19 `v19_pixel_ray_history_and_reverse_override`：55/100（Wilson 95% CI
  45.24%--64.39%），valid 94/100，中位 error 24.00度，普通指令折返0次；记录的
  3次来路方向选择全部是显式 `TURN_AROUND`。

V19 的六个无效样本为两个无非折返地面候选、左/右/后硬门内各一个无地面候选、
一个冻结像素隐藏深度无效。它改善了安全性并净增2个严格45度成功，但仍低于第13
节已接受 V10 的78/100，也低于第14节65/100候选门槛，因此不得替换正式默认。
证据目录为 `outputs/vlm_point_selection_100ep_v19_pixel_history_20260902`。

### 15.2 节点边完成判定：端点交换偏置与调用方差

V14 的 holdout 错误显示，无序端点角色分类对 X/Y 位置标签存在偏置：同一物理节点
交换到另一位置后角色会随字母翻转。V15
`v15_structured_with_swap_consistent_reverse_veto` 因此以两种相反 X/Y 排列独立调用，
映射回物理 previous/current 后，只有两次都以不低于0.70置信度一致认为
“previous 是 completion、current 不是 completion”才可否决主判定。这里的一致性
只比较 completion/non-completion 极性；`source_context` 与 `no_clear_relation` 的
子类型差异不应取消同一反向证据。

V15 在一次开发30复测为25/30；独立 holdout-20 为17/20（85%，completed F1
85.71%，unknown F1 84.21%，macro-F1 84.96%）。V16 放宽复合指令时序措辞后开发
26/30、holdout 16/20，因 unknown 误报增加而拒绝。V17 对结构化主判定先做两次
独立调用、仅在分歧时第三次多数决，开发集达到27/30，但 holdout 仍为16/20，说明
多数决能抑制随机波动，不能修复稳定的语义误判，且调用成本更高，故也拒绝为默认。

按第13节既有回归门禁，正式在线判定仍保持已接受的
`v13_structured_node_edge_binary`（开发26/30、holdout17/20）；V15/V17保留为可显式
选择的研究版本，不得静默改变正式五任务配置。本轮代码级契约测试最终为97/97。

## 16. 十 EP 子指令轮次推进协议（2026-09-04 起生效）

本节定义“十个 EP 逐步选点、导航、建图、判定，直到完整端到端成功”的唯一推进
方式。目标不是先运行一次不可解释的全序列调参，而是沿每条 instruction 的子指令
序列逐段建立能力，并在每一段冻结后保护已经完成的前缀。

### 16.1 固定测试集和阶段坐标

1. 在第一次模型调用前固定十个真实 R2R `val_unseen` episode、数据集 SHA256、
   episode index/id、scene、trajectory、起点和完整 instruction。默认复用本项目已
   冻结的多场景集合 `0, 3, 6, 9, 18, 27, 45, 126, 204, 219`；若更换集合必须
   新建 manifest，不能根据上一轮结果替换样本。
2. 先对十条完整 instruction 完成一次拆解，保存所有子指令的固定 ID、文本、形式、
   语义空间目标、完成边界和阶段顺序。每个 EP 的子指令数可以不同；第 `k` 轮只
   处理存在第 `k` 个子指令的 EP，其余 EP 标记 `not_applicable`，不能从分母中静默
   删除。
3. 第 1 轮从数据集真实 `start_position/start_rotation` 开始。第 `k>1` 轮必须从
   第 `k-1` 轮冻结配置实际产生的到达节点开始：优先通过保存的真实 action history
   在 Habitat 中 replay，或复用同一 state identity 的合规上游 artifact；禁止
   teleport、重新从示范 waypoint 初始化或使用未来 reference path。
4. 当前子指令的路径对齐公式、stage 区间、`alignment_uncertain` 和前缀 artifact
   来源必须写入该轮 manifest。示范轨迹只能用于初始化/最终隐藏评分，不能进入
   在线选点、导航或判定输入。

5. **每次真实测试和优化必须同步覆盖固定十个 EP。** 对当前阶段 `k`，同一轮必须
   为 `0, 3, 6, 9, 18, 27, 45, 126, 204, 219` 逐 EP 运行相同的选点、导航、节点/
   判定或回溯流程；某 EP 没有第 `k` 个子指令时只能显式记为
   `not_applicable`，不能从测试集或失败分母中静默移除。任何候选 prompt、阈值、
   执行器或判定规则都必须以十 EP 的汇总失败 taxonomy 为依据，再在同一十 EP 上
   回归验证。单 EP 运行只能作为不计入阶段门槛的诊断/可视化检查，不能冻结配置、
   不能宣称该阶段通过，也不能据此改变固定测试集。

### 16.2 每一轮的固定顺序

每个阶段 `k` 必须拆成三个有明确冻结边界的子轮次，并按以下顺序推进：

```text
Rk-A  选点：冻结导航、建图和判定，优化通用选点策略
      选点 -> 地面/候选审计 -> 记录选点节点输入
Rk-B  导航：冻结 Rk-A 选点结果，优化点导航执行器
      选定点 -> crop/TAPIR/GNM -> 物理到达 -> 建立 node/edge
Rk-C  判定：冻结 Rk-A/Rk-B，优化上个节点到当前节点的完成判定
      previous/current node + edge history/keyframe -> completed/unknown
```

子轮次的输出必须成为下一子轮次的输入，不能用事后挑选的“正确点”、人工节点或
人工动作历史替代真实输出。Rk-C 判定完成后才允许把该节点作为 R(k+1) 的真实
起点；未到达点不能伪装成已完成前缀。

### 16.3 冻结、回归和阶段通过门槛

1. 每次候选优化只能修改当前子轮次负责的通用模块；当前模块以前的已冻结代码、
   prompt、阈值、模型版本和随机种子写入 `frozen_parent_config`，不得在后续子轮
   静默改变。
2. Rk-A 选点的主门槛是所有适用 EP 均满足：最终 anchor 在合法 dense 2/3 多数票
   地面候选上、可投影到可导航点，且隐藏评分的局部示范方向误差不超过项目当前
   30 度验收标准。VLM 仍不得看到深度、navmesh 或示范方向；这些只用于运行结束
   后的评分。
3. Rk-B 导航的主门槛是所有 Rk-A 通过的 EP 都真实执行到选定点：执行器返回
   `point_navigation_arrived`，隐藏最终测地距离满足项目阈值，并建立 arrival node
   和完整 edge action history。提前离屏、不可达、最大步数耗尽和漏报必须分开统计；
   不能用“节点已写入”代替物理到达。对 `VERTICAL_UP/DOWN` 等长路径目标，
   中途楼梯点簇丢失不得直接视为到达；必须使用远端可见楼梯/landing 锚点、足够的
	   在线运动比例和有限的最近观测 coast，直到终点全景显示 level landing 或明确
	   的目标边界，否则返回未到达并不建立完成节点。
	   对任意形式的丢簇停止都必须有起点估计行程上限：若视觉簇消失或控制仍在前进却
	   已超过该上限，执行器必须返回非到达（例如 `initial_geodesic_travel_cap_exceeded`），
	   不能用全簇消失或预算耗尽伪造 `point_navigation_arrived`，也不能把这段动作历史
	   作为下一子指令的有效前缀。
4. Rk-C 判定的主门槛是所有已建立的真实边都得到正确的 `completed`/`unknown`
   二类结果，并且已完成语义边推进到下一个子指令。人工完成标签必须在模型调用
   前冻结且不输入模型；报告 precision、recall、F1、混淆矩阵和逐 EP 结果。
5. 阶段 `k` 只有在 A、B、C 三个子轮次都通过后才能冻结并进入 `k+1`。如果目标
   是“十 EP 全部通过”，任何适用 EP 失败都不能把该阶段标为通过；可记录部分通过，
   但下一阶段只能对真实已通过前缀继续执行。
6. 每次冻结都必须做前缀回归：使用冻结的前 `k-1` 阶段配置重跑同一十 EP（或
   合规 replay 同一真实状态），任何已通过 EP 不得变为失败，已通过阶段的成功数、
   物理到达数和判定 F1 不得下降。出现退化时回退候选版本，不得牺牲旧阶段换取
   新阶段指标。

### 16.4 通用性和禁止补丁

- 禁止 episode index、scene ID、trajectory ID、具体物体名称、固定像素、固定视图
  编号或某条 instruction 原文的条件分支；不得以“这个 EP 单独处理”修复失败。
- 允许的优化必须表达为可迁移的 instruction form、空间关系、视角扇区、地面/物体
  证据、历史动作约束、点簇几何或时序一致性规则。垂直形式的“远端锚点优先”和
  “中途丢失不算到达”同样是形式级规则，不得退化为某个场景的固定像素或固定
  楼梯步。所有修改都必须在至少一个未参与调参的真实
  state 上做回归。
- 示范 path、未来 waypoint、正确 heading、path index 和隐藏标签只用于结果评分；
  它们不能进入 prompt、候选排序、控制器或完成判定。
- 不能因为一轮中的某个模块较弱而同时改动其他模块；若发现跨模块问题，先记录为
  failure taxonomy，等轮到对应模块时统一处理。
- 中间视觉/导航模型不得回退到 CPU 推理。Grounded-SAM/DINO+SAM、TAPIR/点跟踪、
  GNM/VINT/NOMAD 等模型以及视觉编码器必须使用可用 CUDA 设备；当前测试默认
  `cuda:0`，每轮 manifest 必须记录实际 `device`，显存不足时应停止并调整并发或
  模型批量，不能静默改用 CPU。生产入口在启动时校验 CUDA 可用性、设备编号并设为
  当前设备；请求 CUDA 但不可用时直接失败。只有显式的模块单元测试才允许传入 CPU。

### 16.5 每轮必须记录的产物

组合式完成基准已由
`scripts/build_ten_ep_completion_benchmark.py` 固化为
`outputs/r2r_curriculum_10ep_20260904/benchmark/ten_ep_completion_benchmark_v1.json`。
该基准定义了从子指令 A/B/N/C 门槛到 EP 级成功的合成关系：只有每个适用子指令
都真实选点、物理到点、完整写入 node/edge 并由系统判定 `completed`，且没有未处理
`unknown`、人工接管或失败回溯，最后 Habitat 目标也被系统判定完成，EP 才算成功；
十个固定 EP 全部成功才算十 EP 基准通过。基准中的人工选点是冻结前置和事后审计
标签，不能进入模型输入；没有人工基准或任一阶段 artifact 的 EP 不能按成功处理。

每轮使用独立目录，例如：

```text
outputs/r2r_curriculum_10ep_20260904/
  round_00_baseline/
  round_01_stage00_selection/
  round_02_stage00_navigation/
  round_03_stage00_judgment/
  round_04_stage01_selection/
  ...
```

每个轮次目录必须包含 `manifest.json`、`process.log`、`summary.json`、逐 EP 结果、
失败案例索引、候选配置差异、前缀回归结果和视频/节点/边 artifact 路径。manifest
至少记录：

- `round_id`、`stage_index`、`subround`（selection/navigation/judgment）；
- 固定十 EP 清单、当前适用 EP、跳过原因和子指令哈希；
- `frozen_parent_round`、代码/prompt/模型/阈值/seed、上游 artifact 哈希和 state
  identity；
- 本轮只允许改动的模块、通用问题假设、实际改动摘要和未采用候选；
- 选点、物理到达、节点持久化、判定和前缀回归的逐 EP 指标；
- 每个失败的统一错误类型、证据文件和下一轮待验证假设；
- `stage_gate_passed`、`prefix_regression_passed`、`all_modules_success`。

每轮结束必须追加 `docs/curriculum_10ep_round_log.md` 的一条记录。记录只能在该轮
运行和评分完成后填写，不能事后修改测试口径。最终“十 EP 端到端完成”定义为：十
个 EP 的每个适用子指令都依次经过真实选点、物理点到达、节点/边写入和
`completed` 判定，且没有未处理的 unknown、错误回溯终止或人工接管；仅最终位置
靠近 R2R 目标、仅执行器返回到达、或仅内部游标完成均不算端到端成功。

### 16.6 Active-STOP 十 EP 重新推进最终门槛（2026-09-07）

从本轮重新推进开始，固定 EP 仍为
`0, 3, 6, 9, 18, 27, 45, 126, 204, 219`，不得按结果换样本。最终配置只有在同一
冻结版本、同一轮从十个 episode 的数据集真实起点完整运行，并且 **10/10 均满足
`simulator_reported_success=true`** 时才允许标记通过。该指标固定要求：最后一个
子指令由在线系统在真实到达节点判定 `completed` 后立即主动发出任务级 `STOP`，且
STOP pose 的最终 geodesic distance 位于 episode goal radius 内。9/10、仅几何
命中、仅游标完成、STOP 在半径外、人工接管或事后补写 STOP 均不通过。

推进仍严格采用 16.2 的 Rk-A/Rk-B/Rk-C 顺序：当前轮必须同时覆盖十 EP，先归纳
通用 failure taxonomy，再只修改该子轮负责的模块；已通过前缀必须回归且不得退化。
可以记录阶段性 simulator success 数量，但只有最终同配置的完整起点运行达到10/10
才算本次优化任务完成。

## 17. OpenNav100 起始朝向对齐数据集（2026-09-11 起生效）

### 17.1 问题

官方 R2R VLN-CE `val_unseen.json.gz` 的 `start_rotation` 与指令开局转向词、GT
`reference_path` 的初始方向并不统一。对 100 条 OpenNav id 的量化（左为正，
Δ = GT 初始方向 − 官方 yaw）：`TURN_LEFT` 6 条 Δ 在 +43°~+149°、`TURN_RIGHT`
10 条中 7 条在 −60°~−166° 但 3 条在 +106°~+116°、`TURN_AROUND` 6 条 |Δ| 在
81°~180°；更多的是非转向开局却背对路径（如 1133 "Walk straight…" Δ=−179°、
842 "Go straight…" Δ=−94°、207 "Walk out of the room…" Δ=−179°）。这种起点
朝向让"按指令开局"的选点在开局就没有可用地面候选，与模型能力无关。

### 17.2 对齐数据集的构建规则（form 级，不允许逐条手改）

- 文件：`data/datasets/opennav100_start_aligned/val_unseen_opennav100ids_start_aligned.json.gz`，
  由 `scripts/build_opennav100_start_aligned_dataset.py` 从官方 `val_unseen.json.gz`
  按 `data/opennav100_episode_ids.json` 的 100 个 id（保持清单顺序）生成；除
  `start_rotation` 外**所有字段逐字节等于官方值**，`instruction_vocab` 原样保留。
- 开局 form 由确定性分类 `instruction_taxonomy.decompose_by_definition(text)[0]["form"]`
  给出，不使用 VLM。
- GT 初始方向 = 起点到 `reference_path` 上第一个沿折线累计距离 ≥ 1.0 m 的 waypoint 的
  方位角（整条路径不足 1.0 m 时取末点）；仅用 x/z 分量。
- 转向角：`TURN_LEFT = +90°`、`TURN_RIGHT = −90°`、`TURN_AROUND = 180°`，其余非转向
  form 一律 0°；对齐后的 `start_yaw = GT 初始方向 − 转向角`，即执行完开局转向后正对
  GT 初始方向。
- `TURN_TO_LANDMARK` 与 `OTHER` 无法定义转向角，**保留官方 `start_rotation`**，审计标
  `kept_official`。当前 100 条为 rewritten 96 / kept_official 4。
- 四元数 `[x, y, z, w]` 只绕 +y，`yaw = 2·atan2(y, w)`，与
  `habitat_point_navigation.yaw_from_coeffs` 一致（已交叉验证）。
- 构建产物必须一并提交：`build_manifest.json`（源文件与输出 sha256、规则参数、逐 form
  计数）和 `start_rotation_alignment_audit.jsonl/.md`（逐条审计表）。
  `tests/test_opennav100_aligned_dataset.py` 校验提交产物与规则的一致性；改规则必须重建
  文件、更新本节并升级 `rule_version`，不得只改数据。

### 17.3 使用与报告

- `run_e2e_eval.sh` / `scripts/run_end_to_end_eval.py` 新增 `--start-pose-source
  {aligned,official}`，**默认 `aligned`**；`official` 使用官方全量 `val_unseen.json.gz`；
  显式 `--r2r-data` 优先于两者并记录为 `explicit`。
- `--episode-indices` 指官方全量 split 的行号，与 `aligned` 同用时 preflight 直接拒绝；
  固定十 EP（`0, 3, 6, 9, 18, 27, 45, 126, 204, 219`）协议不受本节影响，仍在官方全量
  数据上按 index 运行（`--start-pose-source official`）。
- 每轮 `manifest.json` 必须记录 `start_pose_source` 与 `dataset_sha256`；报告 OpenNav100
  结果时必须标注使用的是 `aligned` 还是 `official`，两者不得混在同一张成功率表里比较。
- 对齐后的 `start_rotation` 只用于 episode 初始化；第 0 节和第 4 节的 RGB-only 与
  示范信息泄漏约束不变——在线模块仍不得读取任何 pose / reference_path。
- 已知局限：regex 分类会把 "Take a right at the large clock and travel straight"（id 52）
  这类"先走到地标再转"的句子判为开局 `TURN_RIGHT`；这是 form 级规则的代价，按 16.4
  不做逐条覆盖，必要时只能通过改进通用分类规则并重建数据集解决。
