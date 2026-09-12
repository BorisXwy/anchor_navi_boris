# 十 EP 子指令轮次记录

本文件是 `project_rulle.md` 第 16 节规定的逐轮记录模板。每完成一个真实轮次，
在下表末尾追加一条记录；不要覆盖历史记录或在看到结果后改变门槛。

## 轮次记录模板

### Round `<round_id>` — stage `<k>` / `<selection|navigation|judgment>`

- 日期（Asia/Shanghai）：
- 测试目录：
- manifest：
- 固定十 EP manifest/hash：
- 当前适用 EP：
- `frozen_parent_round`：
- 上游 artifact/state identity：
- 本轮目标模块：
- 未运行模块（`not_run_modules`）：
- 本轮通用问题假设：
- 允许改动的代码/配置边界：
- 实际改动摘要：
- 未采用候选及原因：

#### 指标

| EP index | stage | 选点 | 物理到达 | 节点/边 | 判定 | 前缀回归 | 失败类型 |
|---:|---:|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |

- 选点汇总：
- 导航汇总：
- 判定汇总（completed/unknown、precision、recall、F1、混淆矩阵）：
- 已冻结前缀逐段成功数：
- `stage_gate_passed`：
- `prefix_regression_passed`：
- `all_modules_success`：

#### 失败归因和下一轮

- 统一错误类型计数：
- 证据文件/视频：
- 是否发现跨模块问题（只记录，不在本轮越界修改）：
- 下一轮待验证的通用假设：
- 评审结论（冻结 / 继续优化 / 拒绝候选）：

## 使用约束

- 每条记录必须对应一个可原样重跑的 manifest 和独立输出目录。
- 只允许填写真实 R2R/Habitat 运行结果；mock 或契约测试另行记录，不能混入本表。
- 逐 EP 的 `选点/物理到达/判定` 必须引用实际 artifact，不能用事后人工替换的
  点、节点或 action history。
- 若阶段未通过，保留该记录并在下一轮递增 round id；不得删除失败轮次。

## Round 00 baseline — stage 0 prefix（2026-09-04）

- 测试目录：`outputs/r2r_curriculum_10ep_20260904/round_00_baseline/`
- 固定 EP：`0, 3, 6, 9, 18, 27, 45, 126, 204, 219`
- 子轮次：baseline（尚未优化任何当前模块）
- 结果摘要：选点 heading `8/10 <=30°`；执行器到达信号 `6/10`；隐藏物理到点
  `5/10`；到达判定准确率 `5/10`（TP=3, FP=3, FN=2, TN=2）；第一段完整
  通过 `1/10`。
- 主要通用问题：`turn towards <landmark>` 被归为普通
  `APPROACH_LANDMARK`，缺少目标方位转向约束；`VERTICAL_UP` 在楼梯视图与
  初始 heading 不一致时缺少通用 RGB 视角/anchor 复核；提前离屏和漏到达归入
  后续导航轮，完成判定误报/漏报归入后续判定轮。
- 阶段结论：`stage_gate_passed=false`，不进入下一子指令；下一轮只修改通用
  选点模块并回归这十个真实起点。

## Round 01 — stage 0 selection（2026-09-04）

- 测试目录：`outputs/r2r_curriculum_10ep_20260904/round_01_stage00_selection/`
- 固定父轮：`round_00_baseline`；代码边界只改选点/形式定义，导航与节点判定冻结。
- 通用改动：新增 `TURN_TO_LANDMARK`，对“turn towards/to <named landmark>”使用
  地标方位射线；`VERTICAL_UP` 纳入 RGB 楼梯连续性和同侧 anchor 复核；VLM 仍只接收
  RGB/二维信息。
- 选点结果：隐藏 navmesh 点 heading `<=30°` 为 **10/10**，选点地面/投影/初始可达
  为 **10/10**。EP 204 从基线约 179° 降至 1.21°，EP 3 为 17.69°。
- 冻结下游观测：执行器信号 7/10，隐藏物理到点 6/10，V13 节点边完成输出 5/10；
  这些仅用于归因，不改变本轮门禁。
- 阶段结论：`stage_gate_passed=true`、`prefix_regression_passed=true`；冻结
  `v23_landmark_turn_stair_ray`，进入 stage 0 的 point-navigation executor 轮。
- 证据：该目录的 `manifest.json`、`round_report.md`、`summary.json` 以及每个 EP 的
  `trajectory.json`/视频。

## Round 22 — stage 0 selection/navigation regression candidate（2026-09-04）

- 测试目录：`outputs/r2r_curriculum_10ep_20260904/round_22_stage00_selection_navigation/`
- 固定父轮：`round_01_stage00_selection`；选点使用 `v23_landmark_turn_stair_ray`，
  执行器冻结为 `dense_stop_motion_recovery_v11`，地面分割为 Grounded-SAM。
- 通用改动：仅保留 portal 形式的 RGB lower-floor support fallback；全形式 fallback
  因导致已通过的 EP204 选点漂移而拒绝。
- 选点结果：10/10 满足隐藏示范方向误差 `<=30°`，10/10 选点可达。
- 导航结果：10/10 `point_navigation_arrived`，10/10 隐藏最终测地距离 `<=0.75m`，
  10/10 节点和边写入完整。选点/导航前缀回归通过。
- 阶段结论：A/B 通过，C 尚未运行；不进入下一子指令。详见该目录
  `manifest.json` 与 `round_report.md`。

## Round 30 — stage 0 judgment（2026-09-04）

- 测试目录：`outputs/r2r_curriculum_10ep_20260904/round_30_stage00_judgment_v19_orientation/`
- 固定父轮：Round 22 的真实节点/边；选点、点导航、节点构建和 action history 全部冻结。
- 通用改动：`v19_vertical_guard_consensus` 增加完整楼梯终点的 level-landing/no-
  remaining-flight/stability 证据门控；对 `TURN_AROUND`、`TURN_TO_LANDMARK` 使用
  节点航向差与八视角关系补足执行器初始原地转向未进入边 action 名称的问题。
- 结果：判定准确率 10/10；completed P/R/F1=`1.00/1.00/1.00`，unknown
  P/R/F1=`1.00/1.00/1.00`，macro-F1=`1.00`。EP3 的“仍在楼梯上”被正确判为
  unknown，EP45 的“门仍在前方”按语义冻结为 unknown；没有修改上游点和轨迹。
- 阶段结论：stage 0 的 A/B/C 均通过，前缀回归通过；stage 0 冻结。下一轮只能
  优化子指令 1 的选点，并从每个 EP 已真实建立的当前节点继续。
- 证据：该目录的 `manifest.json`、`summary.json`、逐 EP `instruction_completion.json`
  和 VLM prompt/image artifact；完整输入节点/边见 Round 22。

## Round 41 — stage 1 R1-A selection candidate（2026-09-04）

- 测试目录：`outputs/r2r_curriculum_10ep_20260904/round_41_stage01_selection_candidate/`
- 固定父轮：Round 30；十个 EP 均从 Round 22 对应真实 `node_0001` 启动，恢复入边
  action history/来向，并复用已冻结的指令拆解 artifact。
- 通用选点改动：对近前向 `PASS_LANDMARK` 候选增加独立RGB远端通路二次判定；
  若语义侧视角更符合“越过地标”，在原/侧视角之间请求局部RGB视角；已有侧向
  候选不被改写。仍为 Grounded-SAM 地面、无物体检测、VLM RGB-only。
- 结果：阶段1选点隐藏示范方向误差 `<=30°` 为 **10/10**，地面合法/投影/可达
  为 **10/10**，提示中的上一阶段 action history 恢复为 **10/10**。运行同时产生
  的下游观察中，物理到达和隐藏最终测地距离 `<=0.75m` 均为10/10，但尚未作为
  固定选点的 R1-B 独立门禁；R1-C 完成判定未运行。
- 阶段结论：`R1-A passed; stage1 not frozen`。下一轮固定本目录最终选点，独立
  验证 R1-B 点导航，再进行 R1-C 判定；阶段0配置和真实 artifact 不变。

## Round 48 — stage 1 R1-C judgment gate（2026-09-04）

- 测试目录：`outputs/r2r_curriculum_10ep_20260904/round_48_stage01_judgment_v21_transitionfix/`
- 固定父轮：Round 41 的真实 `node_0000 -> node_0001` 节点/边；Round 41
  的选点和 Round 42 的导航输入、Grounded-SAM 地面分割、执行器和 action
  history 全部冻结。期望状态在 Round 42 manifest 中于 VLM 调用前冻结，未向
  VLM 暴露。
- 通用判定改动：`v21_stage_endpoint_recovery_structured` 在不读取示范轨迹的
  情况下，按形式聚合节点语义视角：间隙指令按物体跨左右视角汇总，普通无序
  doorway 用前节点前方到当前节点后方的跨门时序证据，pass 指令在目标仍处于
  部分/前方时拒绝升级，含楼梯的纯转向在前方仍有楼梯时拒绝完成。所有规则
  均为形式级规则，不含 EP/case 分支。
- 结果：完成/未知二分类准确率 **10/10**；completed P/R/F1=`1.00/1.00/1.00`，
  unknown P/R/F1=`1.00/1.00/1.00`，macro-F1=`1.00`。R1-A、R1-B、R1-C
  均通过；Round 30 的 stage-0 冻结 artifact/config 未改动，前缀回归通过。
- 阶段结论：**stage 1 冻结**。下一轮只允许从各 EP 的真实 stage-1
  `node_0001` 继续优化 stage-2 选点；不得重新选点或重跑前缀。
- 证据：`manifest.json`、`summary.json`、每 EP 的
  `instruction_completion.json`、VLM prompt/call 记录和 Round 41 的真实
  节点/边/视频。

## Round 49 — stage 2 R2-A selection candidate（2026-09-04）

- 测试目录：`outputs/r2r_curriculum_10ep_20260904/round_49_stage02_selection/`
- 固定父轮：Round 48；每个 EP 从 Round 41 真实 stage-1 `node_0001` 开始，
  恢复入边 action history；stage 0/1 的选点、导航和判定均未重跑、未修改。
- 本轮只运行 stage-2 VLM 选点（`v24_compound_turn_ray_center`，Grounded-SAM
  floor，semantic detector none），不运行 stage-2 导航/判定。
- 方向门禁按源节点最近的真实 R2R reference-path state 计算到下一参考状态
  的 heading，阈值 ≤30°。可评分 9 个 EP 中通过 **5/9**；点可达 **9/10**，
  观察到物理到达 **9/10**。EP9 的冻结 stage-1 节点已越过该 EP 的最后参考
  waypoint，标为父段过冲候选而非伪造的 stage-2 失败。
- 失败形式：STOP_WAIT 侧向选点（EP6）、ADVANCE_STRAIGHT 的反投影不可达
  （EP18）、ENTER_REGION 过侧门（EP27）、BETWEEN_OBJECTS 未沿连续红毯
  方向（EP204）。下一轮只允许修改 stage-2 选点的通用连续通路/可达性规则，
  不得触碰已冻结前缀；通过后才进入 R2-B 导航。

## Round 52 — stage 2 selection geometry audit（2026-09-04）

- 固定父轮：Round 49。针对失败 taxonomy 先做 v26 几何复核，方向评分仍只在
  运行结束后使用冻结源节点与下一真实 reference state 计算，VLM 不见示范方向和
  深度。结果为 8 个可评分 EP 中 4/8 通过，说明仅靠连续通路共识不能解决“语义
  关系方向与示范局部方向不一致”。本轮不通过，artifact 保留用于回归。

## Round 53–55 — stage 2 final ray iterations（2026-09-04）

- v27 将 `PASS/ADVANCE/ENTER/STOP` 的形式约束放到最终二维 ray 上，并在几何锁定
  时跳过会覆盖方向的第二次 RGB 中心重写。Round 53/54 分别为 6/8、6/8；Round 55
  达到 7/8，唯一剩余问题是 STOP_WAIT 无近前方地面时错误退回后向候选（EP6）。
- 这些候选只改当前 stage-2 选点，未触碰 stage-0/1；未通过轮不作为冻结父轮。

## Round 56 — EP6 generic probe（2026-09-04）

- v28 的独立 STOP_WAIT side-ray probe 将无前向地面时的候选限定为非后方左右视角，
  选出约 -30° 的地面射线，隐藏方向误差 24.6°，可达且到达。该 probe 不是最终
  统计，随后在十 EP 上做完整 R2-A。

## Round 57 — stage 2 R2-A selection（2026-09-04）

- 测试目录：`outputs/r2r_curriculum_10ep_20260904/round_57_stage02_selection_v28/`
- 固定父轮：Round 48；冻结 stage-0/1 的所有选点、导航、节点和判定 artifact。
- 通用改动：`v28_stage2_side_aware_stop_ray`，STOP_WAIT 在无可靠近前方地面时
  只请求非后方左右 RGB 视角；其余使用 v27 最终 ray 几何。Grounded-SAM 仅分割
  ground/floor/carpet，semantic detector=none，VLM RGB-only，禁止深度。
- 结果：隐藏方向阈值 30° 的 8 个可评分 EP 为 **8/8**，地面候选和可达为
  **10/10**；EP9、EP126 因冻结父节点已越过最终 reference waypoint 标记
  `not_scored`。前缀回归通过，R2-A 通过并冻结 v28。

## Round 58 — EP204 executor probe（2026-09-04）

- 在冻结 v28 目标上单独测试 `dense_stop_motion_recovery_v13`。通过增加停止点簇
  的 coast/recovery 帧，EP204 从边界误报恢复为 `all_stop_cluster_points_disappeared`，
  到达信号与隐藏物理距离均通过。probe 结果随后纳入完整 R2-B。

## Round 59 — stage 2 R2-B navigation（2026-09-04）

- 测试目录：`outputs/r2r_curriculum_10ep_20260904/round_59_stage02_navigation_v13/`
- 固定 Round 57 的十个选点，唯一改动为点导航执行器 profile
  `dense_stop_motion_recovery_v13`；不重新选点、不使用未来轨迹。
- 结果：10/10 返回 `point_navigation_arrived`，10/10 隐藏最终测地距离 ≤0.75 m，
  10/10 持久化 arrival node、edge action history 和 keyframes。无 off-screen、
  unreachable、max-step 或 arrival-misreport。R2-B 通过并冻结真实节点/边。

## Round 60 — invalid label audit（2026-09-04）

- 初次判定运行误把十条 edge 都标成 completed；这是调用前标签构造错误，目录和结果
  保留但明确标记 **invalid/provisional**，不计入任何门禁，也不修改测试代码口径。

## Round 61 — stage 2 R2-C judgment audit（2026-09-04）

- 重新按真实 edge 语义冻结标签：EP9 为 completed，其余 9 条为 unknown（边尚未
  证明完成目标地标）。v21 预测全 unknown，得到 accuracy **9/10**；唯一错误是
  ENTER_REGION 跨门后 VLM 的 temporal self-report 过于保守。该错误归因于通用规则
  过度依赖 `keyframe_support/motion_fit`，不是 case 修补。

## Round 63 — stage 2 R2-C judgment（2026-09-04）

- 测试目录：`outputs/r2r_curriculum_10ep_20260904/round_63_stage02_judgment_v22_enter_transition_fix/`
- 通用改动：`v22_stage2_enter_transition` 对 ENTER_REGION 使用“目标在当前前/后
  半球持续 + 连续移动 ≥1 m + 非反向/非矛盾 + 源节点开口或足够跨越距离”的
  结构化迁移判定，不再把 VLM 自报的 ambiguous temporal fields 当硬门槛；不读取
  深度、未来 path 或 episode 名称。
- 结果：冻结标签 1 completed + 9 unknown，预测完全一致；accuracy **10/10**，
  completed/unknown P/R/F1 均 **1.00**，macro-F1 **1.00**。R2-C 通过，stage 2
  A/B/C 全部通过，前缀回归通过，stage-2 选点 v28、执行器 v13、判定 v22 冻结。

### 当前冻结前缀（Round 63）

- stage 0：A/B/C 全部 10/10，判定 v19。
- stage 1：A/B/C 全部 10/10，选点 v24、执行器冻结配置、判定 v21。
- stage 2：A 8/8 可评分方向（10/10 可达），B 10/10 到达/节点，C 10/10
  completed/unknown 分类；v28/v13/v22。
- 尚未宣称十 EP 全部 instruction sequence 完成：stage-2 C 中仅 EP9 的边已被
  冻结为 completed，其余 edge 的 unknown 是诚实的“当前节点尚不能证明该子指令已
  完成”。下一轮应在不改变上述冻结配置的前提下，从真实 stage-2 节点继续处理
  下一适用子指令，或先为 unknown 设计统一的再探索/延伸规则。

## Round 64 — stage 3 R3-A selection candidate v28（2026-09-04）

- 测试目录：`outputs/r2r_curriculum_10ep_20260904/round_64_stage03_selection_v28/`
- 固定父轮：Round 63；适用 EP 为 0、6、9、27、45、126、204、219，EP3/18
  只有三个子指令，按规则 `not_applicable`。所有适用 EP 从 Round 59 真实
  stage-2 `node_0001` 启动，不重跑前缀。
- R3-A 仅运行选点。8/8 地面点可达；隐藏局部方向阈值 30° 的 6 个可评分 EP
  通过 **2/6**（EP45、EP219）。EP9/126 的父节点已越过最终参考 waypoint，按
  规则不评分。主要问题是 STOP_WAIT 关系被统一压成中央 ray，以及 doorway/portal
  目标的局部侧向 bearing 与父节点偏移不一致。阶段门禁不通过，未进入 R3-B。

## Round 65 — stage 3 R3-A candidate v29（2026-09-04）

- 通用候选：有 RGB 地标证据时保留非后方 landmark bearing，明确 left/right 请求
  近侧 45° ray；没有改变 Grounded-SAM、VLM RGB-only 或冻结前缀。
- 结果仍为 2/6、8/8 可达：EP6 改善到 8.3°，但 EP219 从 26.4° 退化到
  105.8°，EP0/EP27/EP204 仍未过阈值。R3-A 仍不通过；v29 不冻结。

## Round 67–69 — stage 3 relation probes（2026-09-04）

- v30 仅做通用关系/portal probe：高置信 SELECT_PORTAL 保留最多 ±135° 的
  目标 bearing；STOP_WAIT 侧向 refinement 允许一个 45° local view。EP204 放宽
  portal ray 后为 60.6°，不纳入；EP27 的 local probe 为 70.8°（比 141.1°改善但
  仍失败）；EP219 恢复为前向保守结果 26.4°。所有 probe 只使用真实节点、RGB、
  地面 mask 和 action history，未读取深度或未来 path，不构成阶段通过。
- 当前状态：stage 0、1、2 已冻结；stage 3 仍停在 R3-A，未执行导航和判定，因而
  没有把候选选点伪装成下一阶段输入。

## Round 70 — 从起点开始的严格闭环协议（2026-09-04）

- 本轮由用户重新定义十 EP 端到端推进规则，固定 episode 集合为
  `0, 3, 6, 9, 18, 27, 45, 126, 204, 219`。
- 新规则已写入 `project_rulle.md` 第 7 节：每个 episode 必须从
  `start_position/start_rotation` 开始，逐轮只推进一个子指令，并依次通过
  人工基准选点、系统选点贴近性、点导航物理到点、到点后二次选点复核、node/edge
  持久化、系统指令判定和必要的真实回溯；所有优化必须形式级通用且保留失败
  artifact。
- 本轮暂未宣称任何新子指令通过，也未修改 stage 0/1/2 的冻结 artifact。下一步
  是生成十个起点的真实六视图与拆解记录，先完成首个子指令的人工基准选点和系统
  选点对齐测试；未通过 A 门之前不执行后续导航或把结果写入冻结前缀。

## Round 71 — 旧 stage-2 选点/动作历史根因审计（2026-09-04）

- 审计目录：`outputs/r2r_curriculum_10ep_20260904/round_59_stage02_navigation_v13/`。
  该目录的十条边均真实返回 `point_navigation_arrived` 并写入 action history，
  但这不能证明子指令语义完成。
- 十个 stage-2 调用的候选视图合计 **0 个语义检测实例**，其中 8/10 个场景所有
  候选均为 `floor_only`，另外 2/10 仅为 `small_seg_safe_floor`。因此 VLM 实际
  在“只有地面、没有地标证据”的输入上选点；“到点”主要表示到达一个可走地面点，
  而不是到达 pass/enter/stop/around 等语义边界。
- 典型误差不是 TAPIR 或 crop 导致的动作漂移，而是上游目标定义错误：EP0 仅移动
  0.40 m 仍在 couch 前，EP27 仅移动 0.30 m 未到箱子房间，EP45 未完成右转且
  未到 bathroom，EP219 零平移即被物理到达。即使执行器报告成功，产生的历史也
  是错误方向/不足进度的历史。
- 本审计不改写旧结果，也不把这些失败归咎于导航执行器。下一候选必须先通过
  R0-A：人工基线与系统选点在同一真实起点对齐；对于需要地标关系的形式，禁止
  无语义证据的普通地面 fallback 冒充目标；若 RGB 仍无法确认关系，必须请求局部
  新视角或返回未决，不能直接执行并污染后续 action history。

## Round 72 — stage-0 起点 selection-only 对照（2026-09-04）

- 目录：`outputs/r2r_curriculum_10ep_20260904/round_72_r0_start_selection_query_expansion_cpu8_10ep/`。
  十个固定真实起点、8 视角、Grounded-SAM floor + RGB-only DeepSeek，未执行导航。
- 选点结果：10/10 地面候选/可达，隐藏局部方向阈值 30° 为 **9/10**，中位距离
  0.61 m；失败为 EP18 `CIRCUMNAVIGATE`（55.2°）。系统视角与人工基线的扇区差
  仍在 45° 内，说明主要误差来自最终像素 anchor 的 ray，而非 Habitat 执行器。
- 通用查询展开已合入 `semantic_detector.extract_detection_queries`：关系短语同时
  提供核心类别和同义词，保留原短语作为语义上下文；不使用深度、示范路径或 EP
  条件。该改动在 EP0 单例将隐藏距离 1.58 m 降至 0.78 m，但 doorway 仍常为 0 检测。

## Round 73–76 — `v31_circumnavigate_forward_competitor` 候选（不冻结）

- 目的：修复无明确左右时把“绕过障碍”硬压成侧向视角，并增加
  `CIRCUMNAVIGATE` 的相机扇区/像素同侧一致性检查。规则仅按 instruction form 生效，
  未使用 EP/scene/path 条件。
- EP18 三次真实起点试跑依次得到 94.0°、代码变量错误导致的无效结果、66.8° 和
  94.0°（最终一次选择后侧右向细化）；候选未达到 30°，全部不冻结。审计说明
  “放开前向”本身不能让 VLM辨认正确绕行侧，且过度的首步前向硬门会把真实约 70°
  左转路线拒掉。当前仅保留同侧 anchor 的通用安全修复，待下一轮用人工基线约束
  做正式 selection A-gate，不把这些候选输入导航。

## Round 77 — EP0 stage-0 严格闭环验证（2026-09-04）

- 目录：`outputs/r2r_curriculum_10ep_20260904/round_77_r0_ep0_stage0_full/`。
  从 EP0 数据集 `start_position/start_rotation` 开始，复用已冻结的指令拆解，
  只执行第 0 个子指令 `EXIT_REGION: Exit the bedroom`。
- R0-A：系统选取 view-1 anchor `[195,192]`。人工起点观察确认 view-0/1 都是
  合法出口候选，view-1 的门框与 hallway floor 更居中；隐藏示范方向误差 7.23°、
  选点最终测地距离 0.39 m，A 通过。
- R0-B：TAPIR 双点簇执行 13 个动作，累计移动 2.86 m；返回
  `point_navigation_arrived`，停止原因为 `all_stop_cluster_points_disappeared`，
  最终目标测地距离 0.3902 m，无 off-screen、stall 或漏报，B 通过。
- R0-C：到点后人工查看 node-0001 六视图，卧室门位于后方，hall/开放空间位于前方；
  初始 view-1 确实是当前可见出口的直接且语义一致选择，无需回到选点重试，C 通过。
- 节点/边：已写入 `node_0001`、`edge_0000`、完整 action history、六视图和五个
  keyframe。系统节点判定返回 `completed`（置信度 0.95），与人工观察一致，D 通过。
- 结论：EP0 的第 0 段完成一次真实 A→B→C→node/edge→D 闭环；该 EP 尚未宣称
  全 instruction 完成，十 EP stage-0 也尚未冻结。逐项审计见该目录的
  `strict_loop_audit.json`。

## Round 79 — EP3 旧执行器配置复核（2026-09-04）

- 目录：`outputs/r2r_curriculum_10ep_20260904/round_79_r0_ep3_executor_v20/`。
- 真实起点的 VERTICAL_UP 选点仍落在可见楼梯中段附近；旧的近测地丢失规则在
  累计移动约 `0.63 *` 初始目标测地距离时就返回到达，终点视图仍能看到楼梯，人工
  审计判为未完成。该结果证明“选点语义不够远端 + 执行器提前到达”会共同污染
  action history，不能把系统 `completed` 当作通过。
- 本轮不冻结，作为失败 artifact 保留。

## Round 80 — EP3 垂直目标严格到达门（2026-09-04）

- 目录：`outputs/r2r_curriculum_10ep_20260904/round_80_r0_ep3_vertical_anchor_executor_v21/`。
- 新增通用 VERTICAL_UP/DOWN 远端可见地面锚点规则，并把近测地丢失到达的最小运动比例
  提高到 `0.90`。结果不再误报到达，但在中心点簇丢失后提前返回
  `navigation_cluster_lost`，说明仅收紧停止条件仍不足以穿过楼梯遮挡。
- 本轮不冻结；未写入有效 node/edge 前缀。

## Round 81 — EP3 垂直目标在线 coast 回归（2026-09-04）

- 目录：`outputs/r2r_curriculum_10ep_20260904/round_81_r0_ep3_vertical_coast_v22/`。
- 保留 Round 80 的远端锚点和严格到达门，新增有限的在线轨迹 coast（仅使用最近观测，
  最多 8 帧，过期即失败闭环）。系统选取 view-6、anchor `[95,143]`，与示范首段
  方向误差 `8.71°`，到首段示范折线距离 `0.2765 m`；执行 9 个动作、累计 `2.3538 m`，
  最终目标测地距离 `0.2394 m`，返回 `point_navigation_arrived`。
- 到达后六视图显示水平的上层 landing，楼梯位于下方/后方；系统 completion judge
  返回 `completed`，与人工 C 审计一致。该单段闭环的 strict audit 见该目录
  `strict_loop_audit.json`。
- 该结果只冻结 EP3 stage-0 的候选执行器前缀；十 EP 的 stage-0 尚未冻结，下一步
  继续从其余真实起点逐段做 A/B/C 审计。

## Round 82 — EP0 前缀回归（2026-09-04）

- 目录：`outputs/r2r_curriculum_10ep_20260904/round_82_r0_ep0_executor_v22_regression/`。
- 使用同一 Grounded-SAM、V23 选点和 v22 执行器从 EP0 数据集真实起点重跑。系统选择
  `EXIT_REGION` 的 view-2（相对 `45°`）地面点，17 个动作、累计 `3.7400 m`，
  最终目标测地距离 `0.3750 m`，返回 `point_navigation_arrived`；节点/边和
  completion judge 均正常，`completed` 与人工出口观察一致。
- 相比 Round 77 只增加在线 coast 配置，没有出现零动作、提前离屏或判定退化；v22
  暂作为 EP0/EP3 的候选前缀，但十 EP stage-0 仍需完成其余起点的严格审计后才能冻结。

## Round 83 — EP6 首次失败归因（2026-09-04）

- 目录：`outputs/r2r_curriculum_10ep_20260904/round_83_r0_ep6_stage0_full_v22/`。
- `PASS_LANDMARK + ADVANCE_STRAIGHT` 细化视角的唯一可用 Grounded-SAM 区域被标为
  `stair tread/stair`，系统却把它当作普通 floor；随后 21 步中出现连续
  `blocked_forward`，未到目标，返回 `navigation_cluster_lost`。这是通用的
  “非垂直语义误用楼梯地面”失败，不是 tracker 的随机抖动。
- 本轮不冻结，作为选点失败 artifact 保留。

## Round 84 — EP6 非垂直楼梯掩码过滤（2026-09-04）

- 目录：`outputs/r2r_curriculum_10ep_20260904/round_84_r0_ep6_nonvertical_stair_filter_v23/`。
- 在选点候选生成阶段对非 `VERTICAL_UP/DOWN` 形式剔除 stair/step/landing 地面掩码，
  并在关系掩码为空时使用受限 RGB lower-floor prior。系统改选 view-7、anchor
  `[84,168]`，与首段示范方向误差 `16.79°`、到完整示范折线距离 `0.7012 m`。
- 执行器 24 步连续前进、累计 `5.2798 m`，最终目标测地距离 `0.5089 m`，无
  blocked-forward，返回 `point_navigation_arrived`；终点全景显示 pool 已在后/侧后，
  system judge `completed` 与人工审计一致。该段 strict audit 见其目录
  `strict_loop_audit.json`。

## Round 85 — EP9 首次失败归因（2026-09-04）

- 目录：`outputs/r2r_curriculum_10ep_20260904/round_85_r0_ep9_stage0_full_v22/`。
- 系统从侧向 view-6 进入 living room，执行器无碰撞地完成 20 步，但首段最终射线
  相对示范方向偏差约 `103.16°`；这是选点/portal 证据冲突，不是导航历史的随机
  漂移。该结果不计入 stage-0 通过。

## Round 86–87 — EP9 首段浅角路线仲裁（2026-09-04）

- Round 86（`round_86_r0_ep9_shallow_route_v32`）将首段误差降至 `47.35°`，但
  portal 检测硬 gate 仍排除了正前方，未达到 30°，不冻结。
- Round 87（`round_87_r0_ep9_shallow_route_advisory_v32`）将首段 portal 检测改为
  advisory，并用独立 RGB shallow-route 仲裁保留近前方 floor。系统选择 view-0、
  anchor `[109,170]`，隐藏示范方向误差 `22.20°`，执行 19 步、累计 `4.0486 m`，
  最终目标测地距离 `0.3281 m`；节点六视图显示已在 living room 内，system judge
  `completed` 与人工审计一致。该段 strict audit 见其目录 `strict_loop_audit.json`。
- v32 仅作为当前首段候选，尚未替换十 EP 的冻结版本；仍需在已通过 EP0/3/6 上做
  前缀回归，并继续完成其余真实起点的 A/B/C。

## Round 88 — EP0 v32 首段前缀回归（2026-09-04）

- 目录：`outputs/r2r_curriculum_10ep_20260904/round_88_r0_ep0_v32_prefix_regression/`。
- 使用 v32 首段浅角路线仲裁、Grounded-SAM 地面候选和 v22 点导航执行器，从 EP0
  数据集真实起点重跑。选择 view-1、anchor `[182,176]`，隐藏示范方向误差
  `2.73°`；执行返回到达，最终目标测地距离 `0.3750 m`，节点完成判定与人工观察一致。
- 相比 EP0 已通过的旧前缀没有退化，v32 可作为候选前缀继续回归，但尚未冻结。

## Round 89 — EP3 v32 首段前缀回归（2026-09-04）

- 目录：`outputs/r2r_curriculum_10ep_20260904/round_89_r0_ep3_v32_prefix_regression/`。
- VERTICAL_UP 仍选择远端 endpoint anchor `[95,143]`（view-6），隐藏示范方向误差
  `8.71°`；v22 有限 coast 穿过楼梯遮挡，返回到达，最终目标测地距离 `0.2394 m`，
  上层 landing 六视图和 completion judge 均通过。
- v32 没有破坏 EP3 垂直段的远端锚点规则；该前缀仍待全十 EP 通过后冻结。

## Round 90 — EP6 v32 首段前缀回归（2026-09-04）

- 目录：`outputs/r2r_curriculum_10ep_20260904/round_90_r0_ep6_v32_prefix_regression/`。
- `PASS_LANDMARK + ADVANCE_STRAIGHT` 使用非垂直 stair/step/landing 掩码过滤，选择
  view-7、anchor `[84,168]`，隐藏示范方向误差 `16.79°`；执行返回到达，最终目标
  测地距离 `0.5089 m`，无 blocked-forward，终点节点和 completion judge 通过。
- 该回归确认 v32 的首段浅角规则与非垂直楼梯过滤可共存；剩余六个固定起点仍需
  按同样 A/B/C 顺序验证。

## Round 88–90 汇总：首段候选前缀（2026-09-04）

- 已完成的 EP0、EP3、EP6 以及 Round 87 的 EP9，首段隐藏示范方向误差分别为
  `2.73°/8.71°/16.79°/22.20°`，均小于当前 `30°` 验收线；四段均有真实点导航
  到达、节点建立和 completion judge 与人工审计一致。
- 这是“选点→导航→节点/指令判定”的首段候选结果，不代表十 EP 冻结，也不代表
  全 instruction 已完成；EP18、EP27、EP45、EP126、EP204、EP219 仍需实测。

## Round 91 — EP18 CIRCUMNAVIGATE 关系目标复核（2026-09-04）

- 目录：`outputs/r2r_curriculum_10ep_20260904/round_91_r0_ep18_v32_stage0/`。
- v32 首段关系选择 view-1、约 `+45°`，但该视角的 Grounded-SAM 只留下右侧
  地毯，编号地面点落在错误的内侧；首段方向误差约 `55.2°`，执行器虽移动并
  到达物理点，节点仍未完成绕沙发指令。该轮确认为“语义视角正确但像素候选错误”。

## Round 92–93 — EP18 CIRCUMNAVIGATE RGB 地面补全（2026-09-04）

- Round 92 因代码顺序错误中止；Round 93 目录为
  `outputs/r2r_curriculum_10ep_20260904/round_93_r0_ep18_circum_rgb_prior_v33_retry/`。
- 对关系掩码过窄的 CIRCUMNAVIGATE 形式增加受限 RGB lower/interior floor prior，
  使视角 view-2、约 `+90°` 的地面候选不再被单侧地毯锁死。点导航真实到达，累计
  `2.8697 m`，最终目标测地距离 `0.3406 m`；节点判定为 unknown，人工观察确认
  此时仍在沙发侧面，方向已基本正确但终点尚未越过地标。该轮不是到达失败，属于
  选点终点语义还不够远。

## Round 94 — EP18 CIRCUMNAVIGATE 远端锚点候选（2026-09-04）

- 目录：`outputs/r2r_curriculum_10ep_20260904/round_94_r0_ep18_circum_far_endpoint_v34/`。
- 在同一形式内增加二维“远端通道锚点”规则：浅侧向视角保留能继续绕行的内侧
  像素，直角侧向视角优先外侧且更远的可见地面锚点；不使用深度、navmesh、示范
  路径或 EP 条件。系统将 anchor `[217,170]` 修正为 `[177,158]`，真实点导航到达，
  累计 `4.3136 m`，最终目标测地距离 `0.3709 m`。节点仍为 unknown，人工审计显示
  沙发仍在前/后侧，说明该形式需要后续继续选点，而不能把一个物理到达点冒充已完成
  的“绕到背后”。该候选尚未冻结。

## Round 95 — EP27 出口段执行器伪到达复核（2026-09-04）

- 目录：`outputs/r2r_curriculum_10ep_20260904/round_95_r0_ep27_v32_stage0/`。
- 系统选取右侧出口视角（约 `-75°`），节点判定为 completed，但执行器累计移动
  `6.1428 m` 后目标最终测地距离仍为 `3.1642 m`；因此这是“追踪丢失/持续前进造成
  伪到达”，不是语义判断正确。该 artifact 不进入冻结前缀。

## Round 96–98 — EP27 丢簇上限候选（2026-09-04）

- Round 96 仅增加 near-loss 上限，仍被 budget-terminal 规则绕过；Round 97 仍观察
  到预算低可见伪到达。两轮均不冻结。
- Round 98 目录：`outputs/r2r_curriculum_10ep_20260904/round_98_r0_ep27_v23_cluster_loss_bound_retry/`。
  关闭所有丢簇近似到达后，执行器返回 `max_steps`、无到达信号，明确拒绝伪造完成，
  但仍会在视觉簇持续错误跟踪时走满预算；该轮定位出需要“全程行程上限”的通用问题。

## Round 99–100 — EP27 行程上限与 fail-fast（2026-09-04）

- Round 99 目录：`outputs/r2r_curriculum_10ep_20260904/round_99_r0_ep27_v23_overshoot_failfast/`。
  因 coast 仍保留旧轨迹，单纯丢簇 fail-fast 未触发，结果仍为 `max_steps`，不冻结。
- Round 100 目录：`outputs/r2r_curriculum_10ep_20260904/round_100_r0_ep27_v23_geodesic_cap/`。
  增加形式级起点最短路行程上限（`1.5 ×` 初始估计），在累计 `4.6028 m` 时停止并
  返回 `initial_geodesic_travel_cap_exceeded`，最终目标距离 `1.6243 m`，无到达信号、
  无有效完成节点。该规则成功阻断了错误 action history，但 EP27 首段仍未通过，
  下一步需要修复“出口点被选在门内/错误落点”与执行器的联合前缀。

## Round 97–100 执行器候选结论（2026-09-04）

- `dense_stop_motion_recovery_v23_no_overshoot` 目前只作为诊断候选：它能阻断 EP27
  的伪到达，但会把未解决的追踪偏差尽早报告为非到达；EP0/3/6/9 的已通过前缀尚未
  用 v23 全量回归，因此不能替换 v22 的候选前缀。

## Round 101 — EP45 原拆解回归（2026-09-04）

- 目录：`outputs/r2r_curriculum_10ep_20260904/round_101_r0_ep45_v32_stage0/`。
- 旧 artifact 将 `Finish climbing the stairs ...` 标为 OTHER，系统在错误的普通地面
  目标上物理到达，但节点为 unknown；该轮作为拆解错误基线保留。

## Round 102 — EP45 垂直形式拆解修复（2026-09-04）

- 目录：`outputs/r2r_curriculum_10ep_20260904/round_102_r0_ep45_vertical_taxonomy_v24/`。
- 将 `climb/climbing/climbed/ascend` 统一归入 VERTICAL_UP 后，首段选取约 `+120°`
  的楼梯/门口连通地面，远端锚点修复生效；与首段示范方向误差约 `8.0°`，执行
  累计 `3.2295 m`、最终目标距离 `0.3682 m`，物理到达和节点 completed 均通过。
- 该轮验证的是可迁移的词形/形式规则，不是对 EP45 的特例；后续仍需做其余固定 EP
  和已通过前缀回归。

## Round 103 — EP126 旧执行器回归（2026-09-04）

- 目录：`outputs/r2r_curriculum_10ep_20260904/round_103_r0_ep126_v32_stage0/`。
- TURN_AROUND 首段的 v23 执行器在目标已经进入近场后仍未把导航簇和停止簇联合纳入
  到达判定，走满预算并返回 `max_steps`；该轮不计为物理到达，暴露了“停止簇已满足、
  旧 goal track 仍残留”这一通用问题。

## Round 104 — EP126 双点簇到达判定（2026-09-04）

- 目录：`outputs/r2r_curriculum_10ep_20260904/round_104_r0_ep126_v24_dual_cluster/`。
- v24 将导航依据簇（crop 中心约 1/9）与停止依据簇（crop 底边）分离：停止簇低于
  一半可见、导航簇仍在近场且连续转向/运动约束满足时才发出到达信号。系统以约
  `180°` 的转向完成 TURN_AROUND，completion judge `completed`（0.95），与人工审计
  一致。该通用规则修复了 Round 103 的漏报，尚未完成全十 EP 回归。

## Round 105 — EP204 指令判定边界（2026-09-04）

- 目录：`outputs/r2r_curriculum_10ep_20260904/round_105_r0_ep204_v24_stage0/`。
- 物理点导航到达，但 completion judge 给出 `completed`、置信度仅 `0.60`，理由同时
  明确指出 manel 未居中、只满足部分“面对地标”条件。该轮按严格协议记为判定失败/不
  一致，问题在 Rk-C 的证据阈值与输出一致性，不归因于选点或执行器。

## Round 106–107 — EP219 拆解词形与 GPU0 回归（2026-09-04）

- Round 106（`round_106_r0_ep219_v24_stage0/`）仍使用旧拆解 artifact，把“Start in
  the middle ... head towards the door ...”当作 OTHER；虽物理到达，节点为 unknown。
- 修复为通用的 taxonomy-first 选择：当前置语被合并进动作句时，优先保留第一个可行动
  的 EXIT/ENTER/等形式，而不是 OTHER。Round 107
  （`round_107_r0_ep219_taxonomy_gpu0/`）正确归为 EXIT_REGION，物理到达且节点
  `completed`（0.85），与门框穿越的 keyframe 审计一致；配置明确记录 `device=cuda:0`。

## Round 108 — EP27 v25 双点簇稀疏导航（2026-09-04）

- 目录：`outputs/r2r_curriculum_10ep_20260904/round_108_r0_ep27_v25_dual_sparse_nav/`。
- v25 在 v24 基础上把导航簇确认阈值收紧到 `0.35`，避免“停止簇偶然短暂消失”触发
  伪到达。EP27 首段 EXIT_REGION 真实执行返回
  `all_stop_cluster_points_disappeared`，累计移动 `2.8517 m`，起点估计测地距离
  `2.9601 m`，最终目标测地距离 `0.1276 m`，`navigation_physical_arrival=true`，
  节点 completion `completed`（0.85）。这轮阻断了 Round 95 的 6.14 m 伪到达，同时
  保留真实目标附近的停止信号；需用 EP0/3/6/9 做前缀回归后再冻结。

## 设备执行记录（2026-09-04）

- GPU0 空闲后，所有中间视觉/跟踪/导航模型的默认设备统一改为 `cuda:0`（包括
  Grounded-SAM、DINO+SAM、点跟踪器、GNM/VINT/NOMAD/图像策略与视觉编码器）；评测命令
  也显式传入 `--device cuda:0`。禁止无提示的 CPU fallback，运行 manifest 必须记录设备。

## Round 109–110 — EP0/EP3 v25 GPU0 前缀回归（2026-09-04）

- 目录分别为 `round_109_r0_ep0_v25_gpu0_regression/` 与
  `round_110_r0_ep3_v25_gpu0_regression/`。两轮均显式使用 `cuda:0`，沿用 v32
  选点和 v25 双点簇执行器。
- EP0：14 步、累计 `3.0800 m`、最终目标距离 `0.3587 m`，物理到达、节点
  `completed`（0.85）；EP3：9 步、累计 `2.3536 m`、最终目标距离 `0.2380 m`，
  物理到达、节点 `completed`（0.95）。相对之前 v22 前缀没有退化。

## Round 111 — EP6 v25 回归中的选点漂移（2026-09-04）

- 目录：`round_111_r0_ep6_v25_gpu0_regression/`。检测和 VLM 第一轮都认为正前方
  是连续路线，但紧接着的“紧凑检测邻域”启发式把确认方向改成 `-90°` 侧向地面，
  起点投影测地距离仅 `0.1254 m`；执行器因未满足视觉近场像素门槛而触发行程上限，
  返回 `initial_geodesic_travel_cap_exceeded`，节点 unknown。该轮将问题归因于
  选点后处理的通用方向漂移，而非 CPU/跟踪性能。

## Round 112–113 — 近场执行器与首段 PASS 通用修复（2026-09-04）

- Round 112（`round_112_r0_ep6_v26_nearfield_regression/`）先加入基于起点测地投影
  的近场 witness，虽正确发出物理到达信号，但暴露出 EP6 原始侧向选点仍不能完成
  “past pool”语义，因此不冻结。
- 随后在 v32 的首段 PASS/straight 规则中保留 VLM 已确认的正前方路线，只有在该
  路线确无中心地面时才请求一次局部 RGB 细化。Round 113
  （`round_113_r0_ep6_v32_forward_commit_v26/`）选择局部 `-26.1°` 连续地面，执行
  27 步、累计 `5.9205 m`、最终目标距离 `0.4680 m`，物理到达且节点
  `completed`（0.92）。规则只依赖指令形式、RGB 和 2-D 地面掩码，不依赖深度、示范
  轨迹或 episode 条件。

## Round 114 — EP9 v32/v26 前缀回归（2026-09-04）

- 目录：`round_114_r0_ep9_v32_forward_commit_v26/`。系统选择正前方 `0°` 路线，
  19 步、累计 `4.0374 m`、最终目标距离 `0.3186 m`，物理到达，completion
  `completed`（0.95）。因此当前四个已通过前缀 EP0/3/6/9 在 v26 上均保持通过；
  近场 profile 仍是候选，尚未宣布十 EP 冻结。

## Round 115–116 — EP0/EP3 v26 完整前缀回归（2026-09-04）

- 目录分别为 `round_115_r0_ep0_v26_full_regression/` 和
  `round_116_r0_ep3_v26_full_regression/`，两轮均显式 `device=cuda:0`。
- EP0 14 步、累计 `3.0800 m`、最终目标距离 `0.3587 m`，节点 `completed`（0.85）；
  EP3 9 步、累计 `2.3536 m`、最终目标距离 `0.2380 m`，节点 `completed`（0.95）。
  加入近场 witness 和首段正前方提交规则后，四个已通过前缀 EP0/3/6/9 均保持物理
  到达与语义完成。

## Round 117 — CUDA 设备硬约束（2026-09-04）

- GPU0 空闲后固定生产运行设备为 `cuda:0`。`habitat_point_navigation.py` 和
  `track_cluster.py` 的 CLI 在启动时校验 CUDA 是否可用、设备编号是否可见，并调用
  `torch.cuda.set_device`；若请求 CUDA 失败则立即报错，不会把中间小模型静默放到 CPU。
- 模块单元测试仍可显式使用 CPU 作为隔离环境；生产 manifest 继续记录 `config.device`。
- 代码编译通过；`tests/test_instruction_graph_modules.py`：30 passed。

## Round 118–129 — 十 EP stage-00 真实状态基准（2026-09-04）

- 固定 EP 集合 `0,3,6,9,18,27,45,126,204,219` 均从真实 R2R 起点运行首个
  子指令，记录八视角、Grounded-SAM/DINO+SAM 地面掩码、VLM 点、执行器到达信号、
  节点/边和 completion judge。产物分别位于
  `outputs/r2r_curriculum_10ep_20260904/round_118_*` 至 `round_129_*`。
- EP0/3/6/9/45/126/204/219 的首段物理到达与选点基准通过；EP18 为
  CIRCUMNAVIGATE 的中途状态，EP27 暴露了侧向 EXIT 门口的通用 ray 选择问题。
  EP204 仍有低置信度判定一致性问题，未宣称十 EP stage-00 全通过。

## Round 130–132 — EP27/EP18 子指令序列与点端点见证（2026-09-04）

- EP27 side-portal 序列（`round_130`）和 EP18 CIRC 路由（`round_131`）均暴露
  “选点已到但视觉跟踪簇未发到达信号”的执行器漏报。加入仅使用所选点自身在线
  navmesh 端点见证的通用补偿后，EP18 `round_132_r0_ep18_sequence_endpoint_witness`
  达到 `instruction_sequence_complete`，4/4 子指令 completed，含两次合法回溯。
- 该端点见证不读取示范轨迹、VLM 结果或隐藏目标，只在停止簇比例和所选点的在线
  测地距离同时满足阈值时发出 `point_navigation_arrived`。

## Round 133–136 — PASS 后侧八视角规则（2026-09-04）

- EP27 多轮序列定位到 PASS_LANDMARK 的通用误判：VLM 在 `rear/rear_left/rear_right`
  已满足、动作持续前进且 keyframe 支持时仍返回 unknown。结构化 harness 和适配器
  统一加入后侧八视角 PASS 规则；`round_136_r0_ep27_sequence_adapter_pass_rule`
  中 PASS 节点记录 `pass_directional_override=true` 并按序完成。
- 该规则不对单个场景/物体打补丁，也不改变选点和物理到达口径；31 个图级/节点级
  单测在后续轮次继续通过。

## Round 137–143 — STOP_WAIT 侧向关系迭代（2026-09-04）

- `round_137`–`round_140` 逐轮尝试侧向 STOP 的近侧 ray、局部视角基准、对侧/下部
  地面锚点；真实 EP27 仍在最后 STOP_WAIT 出现“到达门口但完成关系 unknown”，
  说明问题主要是侧向语义、相机朝向和门槛距离耦合，而非执行器到达漏报。
- `round_141_r0_ep27_query_heads` 将关系短语通用展开为地标头词（如 `doorway`）和
  目的地短语（如 `living room`），Grounded-SAM 检测记录确认 query 生效；该轮因
  DeepSeek 非确定输出导致早期 PASS 结果波动，不能冻结为前缀配置。
- 当前候选 `v33_stop_relation_near_side` 保留 VLM 选定的侧视角，不强制固定 +45°，
  并在 `STOP_WAIT` mask 有效时只从图像下部近侧支撑带取点；`round_142`、`round_143`
  的 EP27 全序列仍记为失败（存在 unknown/错误回溯），因此没有宣称该阶段通过。
- 所有真实轮次命令均显式 `--device cuda:0`；CUDA 不可用时入口直接失败，禁止
  CPU fallback。最新 `tests/test_instruction_graph_modules.py`：31 passed。

## Round 144–147 — EP27 STOP_WAIT 真实节点隔离与低带锚点复测（2026-09-04）

- 从 Round 143 的真实 `node_0011`（ENTER 子指令完成后建立的节点）隔离启动
  `STOP_WAIT`，没有回放起点或向模型暴露示范轨迹。Round 144 两次点导航均发出
  `point_navigation_arrived`，终点在线测地距离约 0.26/0.39m，但完成判定为
  `unknown`；人工检查确认门口/目的地关系未被稳定证实。
- Round 145 保留同一视角与执行器，仅验证下部近侧 2-D 锚点规则；原有采样锚点仍
  没有落入低带，结果与 Round 144 相同。
- Round 146–147 加入通用的低带地面像素合成，并限制侧向 STOP 暴露的所有锚点均满足
  `y>=0.82H`。首个点从 `[160,185]` 调整为 `[177,205]`，在线目标距离从约
  3.34m 降至约 2.53m，物理到达仍稳定；但当前节点仍被 VLM 判为 partial/unknown，
  随后探索方向和回溯未完成该子指令。因此这不是成功冻结：选点的门槛距离有所改善，
  语义端点仍需下一轮通用优化。
- 变更只使用 RGB/地面掩码，不使用深度、navmesh、示范轨迹或 EP 条件；模块单测
  31 passed。上述真实运行均固定 `cuda:0`，禁止中间模型 CPU fallback。

## Round 148–153 — unknown 在路上分流与选点安全边界（2026-09-04）

- Round 148–151 在同一真实 EP27 `node_0011` 上验证：当 completion judge 返回
  `unknown` 且当前 endpoint 为 partial、无反向/静止证据并有真实移动时，外层策略不再
  立即回溯，而是继续当前子指令。Round 149/150/151 前三次点导航均物理到达，证明
  action history 没有被过早截断；但后续 VLM 视角漂到无关门/楼梯，仍未完成 STOP。
- Round 152/153 将 v33 接入通用选点后可达性修复，并增加在线候选测地上限 4m。若同
  一视角/允许邻近视角没有安全候选，记录 `no_reachable_ground_anchor`，不伪造修复；
  Round 153 仍暴露“VLM 选中远处不可执行候选后没有请求新视角”的待修问题，未冻结。
- 这些轮次均从真实保存节点启动用于隔离诊断，不能替代十 EP 从起点的最终验收；模型
  调用固定 `cuda:0`，CPU 仅负责 I/O/统计，单元测试保持 31 passed。

## Round 154 — 不可达候选重选边界复测（2026-09-04）

- 在相同真实 `node_0011`/STOP_WAIT 状态执行一次安全重选：前三个低带点均物理到达；
  第四次 VLM 候选在线测地距离超过 4m 且同视角/邻近视角没有安全候选，系统没有把它
  伪装成“已到达”，而是记录 `no_reachable_ground_anchor` 并重选一次。重选后的候选
  虽可达但处于静止/未完成状态，仍判 `unknown`，本轮不冻结。
- 该轮证明选点-执行边界已能阻断明显远点，但还需要在“无有效门框检测”的情况下
  产生新的可验证视角/地面目标，才能完成 STOP_WAIT 闭环；不把失败归因于 CPU 或
  执行器漏报。

## Round 155–159 — 连续在路上推进与 STOP 近场支撑（2026-09-04）

- Round 155 暴露并修复了接口参数错误；未产生有效导航结果，随后 Round 156 重新执行。
- Round 156 保持 on-route 状态跨越一次物理失败，避免在下一 hop 产生近零重复点；但仍
  出现无安全候选，未冻结。
- Round 157 将连续失败候选保留为诊断但仍执行了第二次不安全点，随后收紧策略；
  Round 158 起第二次仍无安全候选时直接拒绝执行，不创建污染节点。该边界符合严格
  “选点未过关先回选点”的流程。
- Round 159 恢复 Grounded-SAM 原始掩码的中心下部支撑带后，前三个真实 hop 均物理
  到达，点像素出现 `[133,239]`、`[190,239]` 等近场候选；其中一次静止/未完成
  unknown 和后续方向仍未完成 STOP_WAIT，因此当前子指令仍未冻结。
- 上述变更均为 form/掩码/状态机级通用规则；真实运行固定 `cuda:0`，单元测试继续
  31 passed。

## Round 160–162 — 门户语义视角覆盖问题（2026-09-04）

- Round 160/161 的视频审计发现：VLM 已在八视角中选中正确的门口（EP27 起点
  `view 5`），但后续“portal/region 最小偏移”规则把它改成壁炉视图，导致选点
  看似落地面、实际语义方向错误。这是通用后处理覆盖错误，不是单个场景补丁。
- Round 162 将高置信度且包含地标证据的 EXIT/ENTER/TRAVERSE/SELECT_PORTAL 视角
  保护为语义视图；EP27 第一段真实选点保持 `view 5`，2.91m→0.29m，物理到达并
  正确判定 `Exit...` 完成。第二段到达后六视角仍显示楼梯，completion `unknown`
  且 disposition=`on_route`，未提前完成。

## Round 163–165 — PASS 地面候选稀疏的通用兜底（2026-09-04）

- Round 163 从真实 `node_0002` 隔离复测 PASS，Grounded-SAM 在门廊/走廊视图没有
  地面候选，系统因此无安全点而停止；视频确认是分割空洞，不是 VLM 语义失败。
- Round 164 对所有 PASS_LANDMARK（而不只是复合形式）启用已有的受限 RGB 下部
  地面先验，并仍由 DINO+SAM 物体掩码和 VLM 选择最终锚点。真实 node_0002→node_0004
  连续完成 PASS 与 ENTER：3.49m→0.32m、2.90m→0.08m，均物理到达且顺序判定正确。
- Round 165 从真实 `node_0004` 继续 STOP：第一次是合理的 on-route unknown，第二次
  1.78m→0.10m 后判定 STOP 完成，状态 `instruction_sequence_complete`。已通过的
  前缀没有被修改。

## Round 166 — EP27 完整四段端到端回归（2026-09-04）

- 从 EP27 原始起点、无人工点注入运行当前代码，8 次点导航全部物理到达并创建节点；
  中间的 3 个 unknown 均被判为 on-route，未错误回溯，随后在新节点完成对应子指令。
- 四段子指令全部由系统按序完成：`completed_sub_instructions=4`、
  `instruction_sequence_complete`、`recovery_backtracks=0`。各点初始/最终在线测地
  距离为：2.91→0.29、2.92→0.09、3.49→0.32、2.90→0.08、1.99→0.07、
  3.37→0.24、1.89→0.15、1.78→0.10m。
- R2R 原始 goal 的最终测地距离仍为 7.69m，因此 benchmark `r2r success=false`；
  这与“按自然语言四段完成”的序列成功不同，不能宣称 R2R 终点指标通过。视频和
  轨迹分别见 `round_166_r0_ep27_full_latest/exploration.mp4` 与
  `trajectory.json`。
- 本轮及前述轮次的中间模型均固定 `cuda:0`；CPU 仅用于 I/O/统计和 checkpoint
  反序列化，没有 CPU 推理回退。图级测试保持 31 passed。

## Round 168–171 — EP0 TURN_LEFT 近零点、墙面伪地面与失败节点（2026-09-04）

- Round 168 在 EP0 的真实 `node_0001` 发现 TURN_LEFT 候选在线距离仅 0.25m，执行器
  原地近场到达，未产生真正分支进入；加入 TURN/PASS/EXIT/ENTER 等形式的最小
  0.75m 初始进展要求，STOP_WAIT 保留近侧点例外。
- Round 169 的视频显示 Grounded-SAM 约 90% 全幅伪地面覆盖墙面；加入宽且上下均匀
  的二维 mask sanity check，拒绝墙/门板类 proposal，并在 v33 prompt 中显式要求
  TURN 必须看到足够宽的连续走廊。
- Round 170/171 验证了重选链路；物理失败不再创建 node/edge，只保存失败尝试，
  防止错误 action history 污染后续判定。尚未冻结 TURN_LEFT，因为首个候选仍需
  人工与示范轨迹的最佳性复核。

## Round 172–176 — EP0 EXIT 浅层化与动态继续（2026-09-04）

- Round 172 将 EXIT_REGION 的首个门户候选限制为浅层门外点：系统最终点在线距离
  1.216m（示范首个门外状态约 1.44m），方向差约 2.83°，终点 0.150m；这次才
  通过人工最佳性和物理到点双重门槛。
- 为避免浅层 EXIT 后无法继续，后续 on-route 状态动态放宽到 4m，同时要求新的
  0.75m 进展。真实 EP0 Round 176 完成 EXIT 与 TURN_LEFT；TURN 后节点方向与
  示例路径仍需逐点审计，下一 PASS 选择曾因候选安全边界终止。
- 全过程没有把人工示范路径输入模型；示范只用于本后验审计。模块测试保持 31
  passed，推理固定 `cuda:0`。

## Round 177–178 — EP0 PASS 地面/物体 mask 联合约束（2026-09-04）

- Round 177 从真实 `node_0003` 复测 `Walk straight passing the gray couch`，VLM
  把门/墙误识为 gray couch，暴露原先 `object_mask` 始终为空的实现缺陷。
- Round 178 将 instruction-related DINO+SAM detection mask 纳入地面安全约束，
  家具、门和结构面作为禁区，走廊/地板/开口语义保留；该轮因此拒绝了无可靠地面
  的候选，没有伪造导航成功。当前 PASS 子指令尚未冻结，等待下一轮基于真实节点
  的有效地面候选/新视角重选。

## Round 167 — EP0 首个子指令严格闭环（2026-09-04）

- 从 EP0 数据集真实起点开始，仅推进首个子指令 `Exit the bedroom`。人工检查六视图
  和完整拆解后，将含连续走廊地面的门口视图 2/8、地面候选约 `[280,200]` 作为基准；
  示范首步方向与该目标的平面方向差约 2.83°。
- 系统最终保留语义门户视图 8，点 `[285,196]`，与人工基准像素距离约 7.81px，
  在地面候选内；因此选点阶段通过，不是仅凭可达性通过。
- 点导航真实执行 17 控制步，在线目标距离 3.759m→0.253m，发出
  `point_navigation_arrived`，隐藏参考到点标签为 true。node_0001/edge_0000、完整
  action history 和 keyframes 已写入并与轨迹一致；终点六视角仍显示门框在后、走廊在
  前，人工审计认为该点是当前起点可见信息下的最佳可执行目标。
- 系统 completion judge 输出 `completed`，序列节点匹配首个子指令，和人工/示范事后
  结论一致，EP0 子指令 0 可冻结。人工基准明细见该轮 `manual_baseline.json`；本轮只
  运行一个目标，其余模块记为 `not_run`，不能据此宣称 EP0 或十 EP 完成。

## Round 179–180 — 污染前缀审计（2026-09-04）

- 从未冻结的历史节点复测 PASS/TURN 暴露了错误前缀污染：VLM 在错误的 node 上可
  看到相似家具/墙面，不能把这些结果当作当前轮次的选点或判定结论。
- 两轮均保留为失败审计，不纳入冻结结果；后续轮次改为只从 Round 188 的真实 EXIT
  到达节点继续，避免失败 action history 进入新一轮。

## Round 181–187 — EP0 EXIT 门外点与地面候选修复（2026-09-04）

- Round 181 的动态拆解把 EXIT 与 TURN 合并，因不满足冻结拆解可比性而作废。
- Round 182–186 显示 EXIT 门外点过浅、门户视图重试时被错误屏蔽等通用问题；增加
  首次 EXIT 的 1.25m 进展下限、门户扇区地面形态扩展和失败候选审计。
- Round 187 虽选到正确门户方向，但执行器越过目标并未到点；随后启用在线端点距离
  与底边停止簇联合的 v27 执行器。

## Round 188 — EP0 EXIT 子指令冻结（2026-09-04）

- 从真实 EP0 起点、冻结拆解运行。选点 `[13.460,-4.439]`，与示范门外首状态误差
  约 0.27m；在线目标距离 1.613m→0.219m，停止簇可见比例 0.489，执行器返回
  `point_navigation_arrived`，节点/边/action history/keyframes 完整持久化。
- completion judge 与人工审计均判定 `Exit the bedroom` 完成，因此仅子指令 0 前缀冻结。

## Round 189–196 — EP0 TURN 方向、重试和地面掩膜失败（2026-09-04）

- Round 189–191 的负号方向门控把正确侧向走廊排除；Round 192 发生近前方错误分支，
  Round 194 在“太近”重试时屏蔽了正确视角，转入错误的 +45° 分支。
- Round 195 首次重跑漏传冻结的 v33 配置，0 步结果作废；Round 196 记录正确左侧
  `-90°` 语义视角但 Grounded-SAM 将走廊大部分误归为楼梯，残余地面约 6%，无安全点。
- 这些轮次均未创建有效冻结节点，不能作为 TURN 成功率；失败记录保存在各轮
  `selection_failures.json`，视频保留用于人工审计。

## Round 197 — TURN 低面积地面掩膜补全（2026-09-04）

- 对侧向 TURN 的低支撑掩膜加入通用下方 RGB 地面先验并保留可达性过滤。第一跳
  产生合法节点但选点朝向仍不稳定，后续 hop 未完成 TURN；该轮不冻结，说明地面
  候选恢复后还需校验物理 yaw 与语义视角一致性。

## Round 198–199 — 物理左正号与过强 +90° 覆盖（2026-09-04）

- 根据真实 EXIT 边 action history（`turn_left_forward` 对应正 yaw 增量）统一方向门控：
  物理左为正相对 yaw、右为负；图像像素左右不再直接当作 yaw 符号。Round 198/199
  的 VLM 仍把 +90° 视角选为走廊，导航到错误/未知节点，未完成 TURN。
- 同时发现后处理会因存在 90° 地面候选而覆盖 VLM 的 +45° 走廊判断，已改为只有低
  置信且无连续走廊证据时才升级到 90°。这两轮用于回归和问题定位，TURN 仍未冻结。
- 方向门控、低掩膜补全、最小推进修复均通过 `31 passed`；所有模型推理固定使用
  `cuda:0`，CPU 仅负责 I/O、统计与 checkpoint 反序列化，禁止 CPU 中间模型推理。

## Round 200–201 — 错误来向修改的反证（2026-09-04）

- Round 200/201 曾临时把来向方向改成未取负号的 `atan2(x,z)`，随后 VLM 选择了
  视觉上像走廊但实际接近来向的 `+135°` 射线；与示范下一状态的方向误差约 131°。
  虽然三次物理到点，均不能通过“选点接近人工最佳目标”门槛，相关节点不计入冻结前缀。
- 根据点投影公式 `forward=[-sin(yaw),-cos(yaw)]`，该修改已撤销，来向计算恢复为
  `atan2(-x,-z)`。真正待修的是阶段边界：EXIT 边已包含约 29° 左转，下一 TURN 应
  基于 action history 采用已转弯后的前向延续，而不是重复施加侧向转弯。
- 两轮保留为失败反证；所有本地模型调用均为 `cuda:0`，没有 CPU 中间模型回退。

## Round 202 — 从真实起点重新验证 EP0 stage-0（2026-09-04）

- 按本次任务要求从 EP0 的真实 `start_position/start_rotation` 重启，保留完整原始
  instruction 和冻结拆解。人工后验基准为门外首个示范状态 `[13.6494,-4.2417]`；
  系统选择门外地面点 `[13.4602,-4.4391]`，平面误差约 0.272m、方向误差约 2.83°，
  选点阶段通过“贴近人工最佳目标”门槛。
- v27 点导航真实执行 8 控制步，在线目标距离 `1.613m→0.219m`，返回
  `point_navigation_arrived`；停止节点、edge、完整 action history、keyframes 和
  六视角均写入。系统 completion judge 以 0.85 置信度判定 `Exit the bedroom`
  完成，与人工观察一致，stage-0 单段闭环通过。
- 本轮只证明 EP0 stage-0，尚未进行十 EP stage-0 汇总、前缀回归或端到端终点验收；
  对应 `manifest.json`、`summary.json`、`process.log` 已补齐，设备固定为 `cuda:0`。

## Round 204–211 — 续接图与选点重试安全审计（2026-09-04）

- Round 204/205 暴露续接初始化缺陷：旧节点位置/入边被读取，但图被重新建成空图，
  0 控制步仍产生新节点并被 VLM 判为完成；两轮均作废，不计成功率。
- 增加 `NavigationGraphMemory.import_graph`，续接时恢复原节点、边、视图和 keyframe，
  并要求 `initial_node_id` 是导入图最新节点；同时在选点安全失败重试前恢复原始 yaw，
  防止预览转向污染下一次 panorama。31 个模块测试仍全部通过。
- Round 206–208 进一步确认 Grounded-SAM 正确走廊视角的首个锚点可能不可达/过近；
  以前的二次随机换视角会从前方走廊跳到侧墙。后处理增加同视角地面锚点候选，且安全
  失败不再自动换语义视角，失败只保留为当前真实节点审计。
- Round 209–211 是该问题的对照：Round 209 的错误侧视角向东移动约 0.81m，属于
  选点重试错误；Round 211 首次 VLM 明确选择正确前方 view-0，但锚点在在线几何上无
  安全可达点，系统停止而未写错误边；均不冻结。

## Round 212–213 — 正确视角后的执行器/判定分离（2026-09-04）

- Round 212 保留正确 view-0，向示范方向实际移动 2.64m，在线目标距离
  `2.72m→0.08m`，物理到点和节点/边持久化均通过；completion judge 仍 unknown，原因
  是本条边只含前向动作，而左转已发生在上一条 EXIT 边。
- Round 213 将上一条边的通用 turn-carryover action-history 写入当前边及 completion
  prompt 后，仍保持 view-0，但 GNM 执行出现约 196° 反向旋转，24 步后判为 unknown；
  该轮主要是底层执行策略漂移，不是初始视角选择错误。以上两轮均不宣称完成 TURN。
- 所有失败/对照视频、选点图、轨迹和 `selection_failures.json` 均保留；GPU 中间模型
  推理固定在 `cuda:0`，CPU 只用于 I/O、统计和 checkpoint 反序列化。

## Round 215–225 — EP0 前缀继续推进与错误 rug 分支（2026-09-04）

- Round 215 的 TURN_LEFT 从真实 node_0002 继续执行，选点初始几何距离
  `0.752m→0.055m`，5 步到达，completion judge 以 0.85 判定完成；Round 222 的
  `PASS_LANDMARK(gray couch)` 从 node_0003 选取中线前方地面，`1.666m→0.144m`、7 步，
  人工和系统均通过。因此 EP0 的 0–2 子指令前缀暂时冻结。
- Round 223–224 执行 `STOP_WAIT(rug)`：第一次到 living-room 边缘仍是 unknown/on-route；
  第二次到达 node_0006，位置正好与真实示范中间状态重合，判为“仍在路上”并继续。
- Round 225 从 node_0006 选择了 front-right 的 rug 分支，实际到 node_0007 后 rug 出现在
  后方，位置距示范终点由约 1.6m 变为约 3.6m；人工审计判定错误分支，不能计作完成。
  这不是执行器停止误差，而是地标实例/视角选择错误。

## Round 226–231 — 回溯可达性与 STOP_WAIT 视角候选修复（2026-09-04）

- Round 226 首次物理回溯失败：VLM 像素锚点视觉合理但落在不连通网格，执行器返回
  `selected_point_unreachable`，未产生有效运动。增加同视角已验证地面锚点替换（保留
  VLM 视角、仅替换不可达像素），并加入 `--backtrack-only` 以便真实节点回溯测试。
- Round 227 回溯 node_0007→node_0006 成功：8 步、1.78m，node-revisit 平面误差 0.513m，
  全景相似度 0.992；记录确认替换了不可达锚点，回溯模块通过本次真实状态验证。
- Round 228 从回溯节点重新选 rug，点导航 `3.061m→0.277m` 并返回物理到达；人工查看
  八视角确认 rug 已在相邻视野、距离更近，但旧 completion prompt 仍因“rear”过严而返回
  unknown。已将 STOP_WAIT 的 near/beside 规则改为允许稳定的侧/后视地标，只有显式
  `approach/front` 或时间上明确越过才否决。
- Round 230–231 进一步定位选点根因：Grounded-SAM 在 node_0006 的正确前方/侧方木地板
  视角返回空掩膜，硬语义门控因此只留下 detector 置信度最高的错误 rug 视角。STOP_WAIT
  已改为跨视角软检测证据，并在空掩膜时使用有界下方 RGB 地面先验，让 VLM 能比较所有
  连通地面视角；但同一真实状态上 DeepSeek 仍选错 front-right，说明还需下一轮通用的
  视角/实例消歧，不能宣称 STOP_WAIT 已冻结。
- 所有上述运行均使用 `CUDA_VISIBLE_DEVICES=0`、`--device cuda:0`；31 项模块回归测试
  通过。Round225 的错误节点和 Round227 的回溯成功节点均保留，未删除或覆盖。

## Round 232–238 — STOP_WAIT 宽角视角与候选地面审计（2026-09-04）

- 使用 v33 严格 RGB 复测发现，VLM 的独立路线审查多次选中 view3/view4，但旧的
  `STOP_WAIT ±45°` 几何锁又把它改回 view7；Round235 还暴露 view6 的绿色区域实际覆盖
  沙发，说明“有绿色掩膜”不能替代地面视觉审计。
- 增加通用宽角规则：v33 对地标路线审查置信度达到阈值且有地面锚点时，保留最多 180°
  的已审查视角；检测框触碰图像边界的 STOP 地标降级为软证据；来向排除对有弱地标
  证据的 STOP 视角改为软候选。Round238 仍因 review 置信度/字段位置导致后处理覆盖，
  未冻结。
- 这些轮次均明确记录为选点失败或人工不通过：Round235 的实际点在沙发区域，
  Round232/234/236/237/238 的点均没有通过示范方向/地面最佳性审计；不以物理到达或
  judge 的 optimistic 结果替代选点验收。

## Round 239–242 — EP0 STOP_WAIT 真实终点与节点判定修复（2026-09-04）

- Round239/240/241 在同一个真实 node_0006 复验 v33 view4（相对后方 180°）路线，
  选点投影可达，实际终点 `[13.751,1.702]`，距示范终点约 0.73m；六/八视图中 rug
  在相邻视野清晰且处于安全偏移，人工审计通过选点、执行器和节点建立。
- completion judge 最初因自动生成的 `completion_cue="visible in front and close"` 把
  普通 `stop near rug` 错判为必须正面朝向。新增通用 STOP_WAIT 节点规则：结合真实
  `point_navigation_arrived`、edge 移动历史、同一地标的节点语义面积增长判断 near 关系，
  同时兼容节点中的 `mask_area_fraction` 字段；Round242 判定变为 `completed`，并在
  `validation_overrides.stop_relation_operational_override=true` 留下证据。
- 对照上游错误 Round225：rug 面积没有增长且点在另一侧分支，未满足该规则，仍保持
  unknown/错误分支；因此本修复不是按 EP/坐标补丁。EP0 的四段子指令目前具备真实节点
  前缀和终点证据，但仍需从真实起点做完整前缀回归后，才能冻结到十 EP。
- `31 passed` 模块回归保持通过；本轮所有 Grounded-SAM、DINO+SAM、VLM、GNM 推理均
  使用 `cuda:0`，CPU 仅用于 I/O 和统计。

## Round 244 — 物理到达与节点建立边界回归（2026-09-04）

- 针对 Round243 从真实起点回归暴露的边界问题，统一将“执行器到达信号”和“选定
  navmesh 点物理到达”分开：只有 `point_navigation_arrived` 且最终选定点的网格
  geodesic 距离不超过 0.75m 时，才允许建立节点/边、调用子指令完成判定和推进序列。
  提前的 off-screen/cluster 信号只记录为 `premature_offscreen_stop`，不污染图和
  action history，并保留当前子指令以便重选。
- 在 EP0 真实起点、冻结拆解、只执行首段的回归中，第一次尝试信号为 arrived 但
  `0.883m`，正确被拒绝且未建节点；第二次到达 `0.336m` 后才建立 `node_0001`，
  completion judge 正确判定 EXIT 完成。结果：2 次物理尝试、1 个有效节点、首段闭环
  通过；没有使用示范轨迹给模型或 CPU 中间推理。
- 31 项模块回归测试通过。所有 TAPIR、GNM、DINO+SAM/Grounded-SAM、VLM 推理继续
  固定在 `cuda:0`；CPU 仅用于 checkpoint 反序列化、I/O 和统计，不允许 CPU 推理
  回退。

- 同步将该边界落实到主 Habitat 接口和 BFS 纯探索分支：任何分支都不能仅凭执行器
  的 cluster/off-screen 信号建节点；未满足物理 endpoint 检查的尝试只进入记录并按
  frontier/当前子指令重试。这样不会让失败动作历史伪装成有效节点，也不会改变点导航
  执行器本身的控制接口。

## Round 254–263 — EP0 PASS 地标的通用方向/判定/增量审计（2026-09-04/05）

- Round254 暴露 PASS 从 node2 选到回房间方向且被 completion judge 误判完成。增加
  PASS/ADVANCE 形式级 90° 来向排除与节点边反向证据门；物理到点仍建节点，但语义
  完成可降级为 `unknown`，不再污染指令游标。
- Round255–258 证明单靠后方扇区不能完成通过：当八视角仍有明确 `front` 地标时，
  VLM/harness 与 adapter 两层均禁止 rear-side 自动完成。Round258 的错误分支被识别
  为 `on_route`/`wrong` 并触发真实回溯，未把它计作通过。
- Round259 将 PASS/ADVANCE 的初始目标限制为通用 3.5m 增量，并修正显式最大
  geodesic 在 v10 prompt 下未实际生效的问题；31 项模块测试仍通过。
- Round260–261 发现转向发生在选点阶段、未反映在 edge `turn_deg`，导致真实出向被
  来向门误删。现从已验证 source 节点的上一段 TURN 目的和节点航向恢复
  `turn_carryover`，在该 continuation 模式下解除来向硬排除，但仍保留地面/可达性/
  增量限制。该修复是图结构通用规则，不读取示范路径。
- Round262–263 审计二次 VLM 路线视角：v26 能给出路线审查，但后处理跨视角修复会
  改写语义方向；现对已接受的 PASS/ADVANCE 审查锁定原视角，不可达则整次重选。
  Round263 的首次实现触发零角度评分除零，已修复；重跑按预期返回
  `vlm_selection_returned_no_safe_point`，没有错误移动或建节点。以上轮次均只使用
  `CUDA_VISIBLE_DEVICES=0 --device cuda:0`，CPU 仅用于 I/O/统计/checkpoint 反序列化。

## Round 264–267 — EP0 严格前缀重基线与 TURN 状态边界（2026-09-05）

- Round264 从 EP0 的真实 `start_position/start_rotation` 重做 EXIT_REGION。人工先审计
  起点八视角并冻结“卧室门外连续地面”基准；系统最终点位与示范下一状态约 0.19m，
  物理终点 geodesic 0.132m，节点/边/六视角/语义/embedding/action history/keyframes
  完整写入，completion judge 返回 `completed`。该段六步闭环通过，人工记录见该轮
  `manual_baseline.json`。
- Round265/266 的 TURN 结果不计入冻结：前者使用旧 prompt 误选深处走廊，后者发现
  EXIT 选点阶段的 incidental turn 被错误继承为 TURN 已完成，导致当前 TURN 被改写成
  ADVANCE。已将 carryover 限制到当前 PASS/ADVANCE 等后续形式，当前 TURN 不再跳过。
- Round267 在状态边界修复后保持 `TURN_LEFT`，但合法候选的初始 geodesic 不足，系统
  触发 `minimum_progress_reject` 并返回 `vlm_selection_returned_no_safe_point`，未执行
  移动、未建立新节点；说明当前第二段仍卡在选点验收，不能用旧错误移动结果冻结。
- 本轮次所有实际模型运行均显式 `CUDA_VISIBLE_DEVICES=0 --device cuda:0`；GPU0
  空闲时禁止任何 CPU 中间模型回退。

## 新增硬规则：每轮十 EP 同步测试（2026-09-05）

- 自本条起，所有真实选点、导航、节点判定和回溯优化轮次必须同步运行固定十 EP：
  `0, 3, 6, 9, 18, 27, 45, 126, 204, 219`。当前阶段不存在对应子指令的 EP 必须
  显式标记 `not_applicable`，不能静默删样本。
- 单 EP 运行只能作为诊断/可视化，不计入阶段成功率、前缀冻结或最终结论。Round264
  和 Round267 的 EP0 单独运行因此保留为诊断证据，不构成十 EP 阶段通过；后续从
  下一轮开始必须为十 EP 生成完整 manifest、逐 EP artifact 和汇总 failure taxonomy。

## Round 268–272 — R0-A 第一子指令选点基准建立与冻结前验收（2026-09-05）

- Round268 使用 v19 与确定性指令脚手架作诊断：合法选点 8/10，严格 30° 通过 7/10。
  EP219 出现无效深度像素，EP6 的 PASS 地标候选竞争校验耗尽重试，EP18 的
  CIRCUMNAVIGATE 方向偏到 69.7°；未执行导航，不可冻结。
- Round269 的 v33 虽将合法性提高到 10/10，但 EP9 的 ENTER_REGION 回退到 153°，
  属于前缀回归，拒绝该版本。
- Round270 的 v31 在脚手架输入上达到 9/10 严格通过，但脚手架把整句地标传给模型，
  不代表正式拆解基准；Round271 的 v32 回退到 6/10，均不冻结。
- Round272 接入固定的十 EP/真实 R2R 起点及冻结子指令拆解（benchmark v1），并用 v31
  重测：10/10 地面点，9/10 合法 navmesh 投影，严格 30° 为 8/10。剩余两类均为
  通用问题：EP18 的“目标物体后方/绕行”选成近侧通道（94.0°），EP219 的地面候选
  投影后不可达（角度 14.3°）。无 API、重试或 GPU 失败，未进入 R0-B 导航阶段。
- 因 R0-A 尚未达到 10/10，当前只允许优化形式级绕行后方视角约束和投影后可达候选
  约束，随后仍须在全部十 EP 同步复测；不得进行 case 级修补或提前测试下一阶段。

## Round 273–276 — R0-A 通用修复、通过并冻结（2026-09-05）

- Round273 将 VLM 冻结后的 navmesh 可达性修复接入 v31：只在原视角或模型已审核的
  相邻语义视角内寻找可达地面锚点。EP219 从不可达变为可达，十例均合法/可达，
  严格 30° 由 8/10 提升至 9/10；EP18 绕行仍约 92°，未通过。
- Round274 为 CIRCUMNAVIGATE 等已定义关系形式真正启用 DINO+SAM 地标证据，并加入
  behind/backside/pass/beyond 的远侧关系审查。EP18 由约 92° 降到 60.7°，但仍超出
  30°；其余九例无回归。Round275 的 Grounded-SAM 自适应低阈值没有进一步变化，
  因此不把阈值本身当作解决方案。
- Round276 仅在 CIRCUMNAVIGATE 视角的 Grounded-SAM 地面候选完全为空时，加入已存在的
  保守下半幅 RGB-only 地面先验，且在候选审计中显式标记；它不提供深度、navmesh、
  示范路径或 EP 标识。该通用兜底恢复了漏检的正确侧向候选，EP18 误差降到 2.3°。
- Round276 正式结果：10/10 地面点、10/10 navmesh 投影、10/10 可达、10/10 方向误差
  <=30°、0/10 折返；中位角误差 9.6°。R0-A 已写入 `freeze_manifest.json` 并冻结，
  下一步允许进入 R0-B 物理到达测试。所有视觉模型仍只运行于 `cuda:0`。

## Round 277–278 — R0-B 第一子指令物理到达（2026-09-05）

- Round277 使用 R0-A 的十 EP 固定集合、v31 选点配置和 `dense_stop_motion_recovery_v11`
  执行器。10 个进程均正常启动，9/10 同时满足 `point_navigation_arrived` 与最终
  选定点测地距离 <=0.75m；EP3 在第 4 步导航簇丢失，最终距离 0.957m，未创建错误
  节点。该失败归因为短时跟踪丢簇，不回退修改选点。
- Round278 只替换为形式无关的 `dense_stop_motion_recovery_v13`，增加 8 帧有界
  navigation-loss coast，保持停止簇一半消失、在线端点和 0.75m 物理门槛不变。
  十 EP 结果为 10/10 executor arrival、10/10 最终测地距离、TP/FP/FN/TN=
  10/0/0/0，物理到达率 1.0；无人工接管，视频与逐步 action history 均保存。
- R0-B 已写入该轮 `freeze_manifest.json` 并冻结。下一步只能在这十个真实节点上运行
  R0-N 节点/边字段完整性，然后单独运行 R0-C 二分类指令完成判定；不得重新选择点或
  进入后续子指令。

## Round 279–285 — R0-C 完成/unknown 判定通用优化（2026-09-05）

- Round279 的冻结十 EP 判定基线为 8/10，macro-F1=0.7917。错误集中在两类通用边界：
  TURN_AROUND 的真实执行航向约 140°、但旧几何门槛要求 150° 且把 reverse transition
  当成否决；门到室外的 OTHER 分解没有复用 EXIT_REGION 的双侧门框歧义规则。
- Round280–282 测试了三个更激进的 endpoint-recovery prompt，均回退到 6/10，未接纳，
  说明问题不在于放宽整体完成判定，而在于形式/语义关系级的证据门。
- Round283 先加入 TURN_AROUND reverse 语义和出口歧义门，仍为 8/10；审计确认几何门
  内部仍残留 `not reverse_transition` 条件。该轮不冻结。
- Round284 将 TURN_AROUND 的通用几何支持改为航向变化 >=120°、水平位移 >=0.20m、
  行进 >=0.50m、至少两个控制步且存在转向动作，并让该形式的 reverse transition
  作为正证据；同时将包含 outside/outdoor/doorway/exit 语义的 OTHER 子指令纳入统一
  出口双侧视角歧义规则。Round285 重跑同一冻结十 EP：10/10 正确，completed 与
  unknown 的 precision/recall/F1 均为 1.0，macro-F1=1.0。R0-C 已冻结，未使用标签
  作为 VLM 输入，也未使用示范路径或深度。

## Round 286 — R0-N 节点/边不变量正式十 EP 审计（2026-09-05）

- 对 Round278 已通过物理到达的十个真实 EP 逐一核验：每个均有有效到达节点和边；节点
  包含位置/航向、六视图、环境语义、256 维视觉 embedding、到达信号和子指令字段；边
  包含 source/target、完整 action history、控制步数、行进距离及 chronological
  keyframes；节点的 instruction-completion panorama 均为八视图。
- 结果为 10/10、不变量通过率 1.0，写入 R0-N `freeze_manifest.json`。至此 R0-A 选点、
  R0-B 物理到达、R0-N 建图、R0-C 二分类完成判定均已在同一固定十 EP 基准上通过并冻结。
  下一轮只能推进到 R1 的下一段子指令前缀，并必须保持已冻结首段结果不回归。
## Round 287 — 十 EP 当前完整序列基线（2026-09-05）

- 使用冻结 R0 配置从十个数据集起点执行完整 instruction-sequence-recovery：DeepSeek
  VLM、v31 选点、Grounded-SAM 地面、DINO+SAM 语义、GNM、dense_stop_motion_recovery_v13，
  全部视觉/导航模型运行于 `cuda:0`。
- 十 EP 均正常启动，但完成四个子指令为 0/10、Habitat 终点成功为 0/10；最远 EP9 和
  EP126 各完成 2 段。逐 EP 完成数为：EP0=1、EP3=0、EP6=1、EP9=2、EP18=0、EP27=0、
  EP45=1、EP126=2、EP204=0、EP219=1。
- 失败类型：`vlm_selection_returned_no_safe_point` 5 例、`physical_failure_recovery_failed`
  3 例、`max_sequence_exploration_hops` 2 例。首段主要问题为楼梯顶端未完成、绕障目标
  不可达、纯地标转向物理失败，以及出口判定的视角歧义；后续只能按首个失败子指令
  分轮修复，不能跨段修改。

## Round 288 — R0 首段出口判定前缀回归（2026-09-05）

- 新增通用“出口后侧视角严格占优 + 穿门关键帧”判定，用于解决出口门框在相邻视角
  重叠造成的保守 unknown；平衡前后视角仍保持 unknown。
- 十 EP 同步复测后，EP27 首段从 unknown 提升为 completed，但总体仍未达到前缀冻结门：
  完成四段仍为 0/10，终点成功为 0/10；EP9 出现 VLM 地面视角审查随机回归，EP3/18/204
  继续分别受楼梯、绕障和纯转向物理执行影响。Round288 不冻结。
- 当前下一轮只允许处理首段的通用执行/候选连续性（楼梯、绕障、纯转向），同时保持
  Round288 的出口规则和 R0 冻结结果做前缀回归。

## Round 290 — 首段前缀门槛复测（2026-09-05）

- 使用十个固定真实 R2R EP、`targets=1` 和冻结的 v31/v19/v13 配置进行首段前缀复测。
  该轮确认首段按序完成为 6/10；EP204 的 `TURN_TO_LANDMARK` 因通用 2.5m 目标上限
  返回 no-safe-point，EP3/EP18/EP219 分别受楼梯、绕行和出口视角证据影响。
- 该轮不冻结，不进入第二段，作为后续通用修复的基线。

## Round 291 — TURN_TO_LANDMARK 距离/预算通用修复（2026-09-05）

- 将 `TURN_TO_LANDMARK` 的形式级安全目标范围从 2.5m 放宽到 8.0m，并为该形式提供
  32 步执行预算；这是关系形式级的距离/控制预算，不使用 EP 标识、示范轨迹或深度
  作为 VLM 输入。现有 100 个单元测试全部通过。
- 同一十个真实 R2R EP 首段前缀复测结果：8/10 子指令首段按序完成，首段通过 EP 为
  0、6、9、27、45、126、204、219；未通过为 EP3（楼梯顶端仍未形成稳定的上层落脚
  证据）和 EP18（绕行首段仍在边界/执行预算内耗尽）。物理到达率 14/15，执行器到达
  信号精度 0.929；该轮已满足首段前缀推进门槛但不代表四段完成验收。
- 本轮保持 R0-A/R0-B/R0-N/R0-C 冻结接口不变，下一轮推进第二段子指令。

## Round 292 — 第二段前缀基线（2026-09-05）

- 在十个固定真实 R2R EP 上将目标前缀推进到第二个子指令。冻结配置下仅 EP126 连续
  完成两段；大量后续候选在门后/物体间投影到可行走网格外，二段连续前缀为 1/10。

## Round 293 — 后续阶段 navmesh 投影修复（2026-09-05）

- 将 v31 的后选择地面/navmesh 修复扩展到非首段，保持 VLM RGB-only；二段连续前缀仍受
  形式安全边界影响，但 EP204/EP219 等后续目标恢复，统计为 4/10，物理点到达率约 0.813。

## Round 294 — 转向/物体间关系预算复测（2026-09-05）

- 将 TURN_LEFT/RIGHT、BETWEEN_OBJECTS、OTHER 的形式级候选范围统一到 8m、32 步。
  二段连续前缀为 4/10，无首段回退；剩余问题转为最小进度门和恢复距离门。

## Round 295 — 转向最小进度与恢复窗口修复（2026-09-05）

- 纯转向最小进度降为 0.30m，未知/转向续接恢复窗口为 8m；十 EP 二段连续前缀仍为
  4/10，但 EP0、EP27 的二段通过，首段结果保持。物理到达/执行器信号均为 1.0。

## Round 296 — 区域/物体间软检测与预算修复（2026-09-05）

- BETWEEN_OBJECTS、ENTER_REGION、TRAVERSE_PORTAL_REGION 使用软语义检测、硬地面可达性，
  区域目标范围 6m、32 步；十 EP 二段连续前缀提升到 5/10（EP0、9、126、204、219），
  五个 episode 已可按序完成两段。该轮仍未进入第三段，下一轮继续全十 EP 前缀测试。

## Round 297 — 第三段前缀基线与转向证据修复（2026-09-05）

- 十 EP 推进到第三个子指令，EP0 连续完成三段；EP9、EP126、EP204、EP219 保持两段，
  其余 episode 在更早前缀停止。第三段失败主要集中在区域目标未真正进入、楼梯/门后转向
  和 TURN_RIGHT 的左右符号歧义。
- 新增形式级转向判定：当同一地标从前一节点右侧移动到当前节点左侧（或反向）、同时有
  有效航向变化/移动时，使用时序扇区变化纠正 VLM 偶发的左右符号误判；楼梯未完成保护仍
  生效。该修复通过 100 个单元测试，下一轮做十 EP 前缀回归。

## Round 298 — 转向扇区规则十 EP 回归（未接纳）（2026-09-05）

- 在十个真实 EP 上复测第三段前缀；由于 DeepSeek VLM 的候选随机性和长恢复预算，
  该轮出现已有前缀被恢复动作带偏，连续第三段完成未提升，部分 EP 物理恢复失败。
- Round298 不冻结转向扇区规则，也不覆盖 Round296 的二段基准；下一轮限制恢复只在
  未到达目标时触发，避免已到达节点被额外探索破坏。

## Round 299 — 保守恢复参数回归（未接纳）（2026-09-05）

- 将恢复上限收回到 6 hops/8 targets，试图避免长恢复破坏已完成前缀；第三段完成仍为
  0/10，说明主要瓶颈仍是长目标的选点/执行连续性，不冻结该轮。

## Round 300 — PASS_LANDMARK 预算修复（2026-09-05）

- PASS_LANDMARK 提供 40 步、ADVANCE_STRAIGHT 提供 32 步形式级预算，解决约 7.7m
  目标在 20 步内必然耗尽的问题。十 EP 第三段前缀恢复到 3/10（EP0、EP9、EP27），
  物理到达判定准确率 1.0；继续保持二段基准。

## Round 301 — v31 转向提交优先回归（未接纳）（2026-09-05）

- 将 v31 纳入已有三视角转向提交优先集合；第三段完成回落到 1/10，未证明稳定收益，
  不冻结该轮。

## Round 302 — ENTER_REGION/带目的地转向回归（未接纳）（2026-09-05）

- 加入跨前后视角的 ENTER_REGION 形式证据和带目的地 TURN_LEFT/RIGHT 的侧向提交，
  100 个单元测试通过；十 EP 第三段仍为 1/10，EP0 出现物理恢复失败。该轮不接纳，
  下一轮先做前缀回归审计，再继续向第四段推进。

## 终点指标口径修正（2026-09-05）

- 旧轮次中的 `r2r_success` 只是几何 goal-radius 命中，不能再作为任务终点成功引用。
  Round296 的 EP3、EP9、EP126、EP204 均未完成完整指令链，因此严格成功统一重判为
  0；其中 EP3 属于错误轨迹的偶然命中，另外三个只运行了截断的两段前缀。
- 后续正式指标改为 `instruction_validated_r2r_success`：必须从原始起点按序完成完整
  子指令链并进入终点半径。单纯几何命中只作为失败诊断，不进入成功数、成功率或
  “到达终点”的文字汇报。
- 子指令完成统计进一步收紧为双门条件：在线导航系统必须在到点新节点/边上判定
  `completed`，且运行后独立语义核验必须确认确实完成。旧轮次只含在线 judge 输出而
  没有逐段标准化核验文件的结果，不能计入严格的 `sub_instructions_completed`。

## 严格重启 Round 001–005 — 第 0 段双门重审（未冻结，2026-09-05）

- 目录：`outputs/r2r_curriculum_10ep_20260905/round_005_stage0_rear_endpoint_postalign/`。
- 固定 EP 为 0、3、6、9、18、27、45、126、204、219；全部从数据集真实起点执行，
  `targets=1`，没有从保存节点续跑，也没有把示范方向、深度或审计标签输入模型。
- 每个 EP 均实际完成一次点导航并新建一个节点/边；物理到点 10/10，在线节点判定
  completed 10/10。加入严格的“当前语义完成且未提前跨过下一段关键边界”重审后，
  独立验收为 8/10：EP3 实际仍在楼梯中段，EP126 的转身边提前穿过了下一段的门。
- 严格交集结果修正为 8/10；每例的 `stage_completion_verification.json` 均已加载，
  只有 EP0、6、9、18、27、45、204、219 的 `jointly_passed_stage_ids=[0]`。缺少语义、
  顺序边界或在线节点门中的任一项都会按 fail-closed 记为未完成。
- EP18 的通用绕行端点依据是目标只出现在 rear/rear-side、没有 reverse transition；
  EP204 在物理到点后仅做 arrived-state 朝向对齐，壁炉 mantel 位于前半球且约 27°，
  满足冻结的 30° 方位标准；EP219 的时序关键帧确认跨到门的另一侧。这些规则均为
  形式级规则，未按 EP 或场景打补丁。
- 116 个单元/契约测试全部通过。第 0 段尚不能冻结；下一轮必须优先修复 EP3 的楼梯
  端点和 EP126 的纯转向越界，并在十 EP 原始起点同步回归。只有本轮实际通过的前缀
  才允许执行第 1 段，两段仍需分别独立核验。

## 严格重启 Round 006 — 第 1 段真实起点基线与边界重审（2026-09-05）

- 目录：`outputs/r2r_curriculum_10ep_20260905/round_006_stage1_from_true_start/`。
- 十个 EP 全部从真实起点重放，`targets=2`。在线系统宣称两段完成 7/10；逐边审计后
  连续两段严格通过仅 4/10（EP0、9、18、204），第 0 段严格通过 8/10，共 20 个被测
  段的严格有序前缀为 12 段。
- 假阳性包括：EP3 在 y=2.41 m 楼梯中段即报到顶；EP126 的 TURN_AROUND 边移动
  2.6 m 并提前穿过下一段门，随后 stage-1 边只沿阁楼栏杆移动；EP219 只上升约
  0.61 m 到中间平台即报楼梯顶部。几何 goal-radius 偶然命中 2 次，均保持无效。
- 其余 stage-1 失败为：EP6 BETWEEN_OBJECTS 的正确语义视角没有可达地面投影；EP27
  多次沿错误支路移动并持续 unknown；EP45 第一次到点后 unknown，恢复点执行失败。
- 新增 `ordered_stage_boundary_verified` 为严格必需字段；当前边提前执行下一段，或后续
  边才补完当前段，均不得跨边补算。该规则已加入 `project_rulle.md` 和评分器测试。
- 下一轮采用形式级修复：复合“转向+上/下楼”以垂直终点为 primary form；完整楼梯
  终点若前半球仍有可靠楼梯检测则保持 unknown；纯 TURN_AROUND 点目标限制在 1.25m，
  垂直 unknown 必须继续真实位移。118 项测试通过后进行十 EP Round007 回归。

## 严格重启 Round 007 — 垂直/顺序边界回归（API 余额中断，2026-09-05）

- 目录：`outputs/r2r_curriculum_10ep_20260905/round_007_stage1_vertical_boundary_fix/`。
- EP0、EP3、EP6 在 DeepSeek 余额耗尽前完成有效运行；EP9 在 completion confirmation
  阶段返回 HTTP 402，EP18、27、45、126、204、219 随后的初始 VLM 调用均返回
  `Insufficient Balance`。后七例不计为算法成败，本轮不能形成十 EP 正式通过率。
- EP3 的新垂直门生效：node_0001（y=2.41m）和 node_0002（y=2.62m）均保持
  unknown，node_0003 到达 y=3.59m 的真实上层落脚区才完成 stage-0。说明“中间平台
  假到顶”问题已在该真实 EP 上消除。
- EP3 stage-1 暴露新的通用问题：上楼后的正确上层路线在水平投影上与来路重叠，旧的
  planar incoming-direction exclusion 将正确栏杆/卧室方向删掉，导致错误侧选点并由
  judge 假阳性完成。现已仅在垂直阶段切换后的第一条边开放完整 landing panorama，
  prompt 明确选择同层命名路线并禁止沿楼梯下降。
- EP6 仍在 BETWEEN_OBJECTS 正确语义视角内出现无可达 anchor；修复为在同一冻结
  Grounded-SAM mask 内将物理 anchor 抽样从 16 加密至 128，不开放未分割像素、不改变
  VLM 视角，也不把深度/navmesh 输入 VLM。
- 上述后续修复与 strict boundary 评分器共 119 项测试通过。需要 DeepSeek 余额恢复后，
  使用相同十 EP/真实起点完整重跑 Round008；不得混用 Round006/007 的旧条目补齐统计。

## 严格重启 Round 008 — 续费后两段基线恢复（当前冻结基线，2026-09-05）

- 目录：`outputs/r2r_curriculum_10ep_20260905/round_008_stage1_postvertical_densegap/`。
- DeepSeek 视觉接口恢复后，固定十 EP 全部从真实起点执行两段；运行后逐边写入独立 RGB、
  action history、节点与示范轨迹投影核验。严格有序通过 15/20（75%），连续两段通过
  6/10：EP0、6、9、18、204、219；EP3/27/45 仅通过第一段，EP126 为 0。
- EP3 已真正走完整楼梯，但第二段进入错误卧室；EP27 的正确侧向走廊视图被旧的
  straight-route 几何修正改回近前方；EP45 的语义点正确但到点剩余 0.768m，略高于
  0.75m 隐藏物理阈值；EP126 的纯转身后方地面被 1.25m 安全上限全部拒绝。
- 该轮是后续修改必须回归对照的当前冻结基线；不因后续轮在线值更高而自动替换。

## 严格重启 Round 009 — 侧向保留/延长到点回归（未接纳，2026-09-05）

- 目录：`outputs/r2r_curriculum_10ep_20260905/round_009_stage1_routepreserve_turn_dense_coast/`。
- 十 EP 同步回归、独立审计后严格通过 13/20（65%），连续两段 5/10，低于 Round008。
  EP6 的 BETWEEN_OBJECTS 首点为近零位移，后续绕回泳池/吧台起点附近却在线报完成；
  EP27 已把楼梯放入后方扇区、EP45 已到床旁右侧门，但两者在线判定仍为 unknown。
- 因存在实际假阳性且严格指标回退，本轮不冻结。提取的共性问题为：关系型指令也必须
  强制产生不同物理节点；八视角长地标可在后方和前方重叠；控制器局部转向符号不能单独
  否决语义端点。

## 严格重启 Round 010 — v21 端点恢复/关系位移回归（未接纳，2026-09-05）

- 目录：`outputs/r2r_curriculum_10ep_20260905/round_010_stage1_progress_pass_endpoint_turncap/`。
- 124 项契约/单元测试前的运行版本通过 123 项；固定十 EP 从真实起点同步执行。在线完成
  14/20，独立审计后严格为 13/20（65%），连续两段 6/10，仍未超过 Round008。
- 改进：EP27 的楼梯/卫生间在 rear-left/rear/rear-right 且时序、关键帧、移动与无回退门
  同时成立，按项目规则完成；EP45 第二次真实到点后完成床旁右侧门端点。
- 回退/假阳性：EP6 的第二段选在偏离示范方向约 45° 的椅子侧，VLM 本身也只给 partial
  且无边界关键帧，被独立审计拒绝；EP18 与 Round008 使用相同首点和到点位置，却因长
  沙发多报一个 front-right 重叠扇区而没有触发后方端点；EP3 楼梯远点执行失败；EP126
  即使 1.50m 上限仍无 Grounded-SAM mask 内安全锚点。
- 后续共性修正：CIRCUMNAVIGATE 采用“后方扇区存在 + 足够真实位移 + 无反向/运动不矛盾”
  而非要求后方扇区独占；BETWEEN_OBJECTS 端点恢复重新强制 instructed 时序和关键帧；
  纯转身只对 Grounded-SAM 底部相连边界做 12px 小范围扩展，新增像素仍须通过 navmesh
  与最短路径门，禁止继续把上限放大进下一段门内。124 项测试通过后进入 Round011。

## 严格重启 Round 011 — 后方重叠/转身 mask 边界回归（未接纳，2026-09-05）

- 目录：`outputs/r2r_curriculum_10ep_20260905/round_011_stage1_rearpresent_betweenkey_maskedge/`。
- 在线宣称 17/20；复核节点朝向并修正初次审计后，严格为 13/20（65%），连续两段
  5/10，低于 Round008。严格前缀为 EP0=2、EP3=0、EP6=1、EP9=2、EP18=2、
  EP27=1、EP45=1、EP126=0、EP204=2、EP219=2。
- 可保留的局部改进：EP126 首次从无可执行点恢复。VLM 后方视图不变，只把 Grounded-SAM
  底部相连 mask 的锚点 `[155,196]` 修到 `[137,213]`；初始/最终 geodesic 为
  1.468/0.057m。但图节点记录的源到终点朝向变化只有约 98°，并未完成要求的 180°
  转身，因此 stage0 必须拒绝，后续跨门也不能进入有序前缀。EP18 恢复了沙发后方与
  厨房台面两段。
- 被剔除的在线完成：EP6 直到 z≈-4.85m、已经执行下一段吧台拐角后才完成 BETWEEN；
  EP45 在 VLM 给出 partial、ambiguous、无关键帧边界时，仅凭约 58° 节点朝向变化被
  TURN endpoint recovery 提升。EP27 则相反：完整 pass 确定性门已成立，但 adapter 的
  旧 front persistence 门仍把长地标前后重叠降为 unknown。
- 下一轮修正仍为形式级：adapter 的 PASS 前方持久门在“rear + instructed + keyframe +
  supports + no reverse”成立时不得否决；TURN 恢复必须有同一实例的 instructed/keyframe
  端点或一致的地标扇区转移；BETWEEN 限制 3.5m 单跳，并允许有关键帧/运动支持的前后
  对置物体作为环视夹缝证据。126 项测试通过后进入 Round012。

## 严格重启 Round 012 — adapter PASS/严格转向/局部 BETWEEN 回归（未接纳，2026-09-05）

- 目录：`outputs/r2r_curriculum_10ep_20260905/round_012_stage1_adapterpass_turnstrict_betweenlocal/`。
- 固定十 EP 从真实起点运行，在线宣称 15/20；六视图、边关键帧、action history 与示范
  轨迹投影逐边独立审计后严格为 14/20（70%），连续两段 6/10。严格前缀为 EP0=1、
  EP3=0、EP6=2、EP9=2、EP18=2、EP27=2、EP45=1、EP126=0、EP204=2、EP219=2。
  仍未超过冻结的 Round008（15/20），因此不接纳为新基线。
- 正向变化：EP27 的 PASS 不再被 adapter 的长地标前方重叠误否决；EP6 的
  BETWEEN_OBJECTS 最终节点位于吧台与座椅的真实夹缝，且在下一段“吧台拐角”之前；
  EP45 的 compound TURN 不再只凭朝向变化误报完成。
- 仍存问题：EP0 的纯 `Turn left` 没有在转向边界闭合，连续探索最终进入浴室才被在线
  宣称完成；EP3 的节点仍在楼梯中段，后续远点执行失败；EP126 虽然真实到达所选地面点，
  但只转约 98°，严格判定保持 unknown。下一轮只做形式级执行修正：纯转向在真实到点后
  按源节点 yaw 执行并记录精确的 90°/180° orientation-only alignment，再采集节点证据；
  带有跨房间、到门口等目的地的 compound turn 不使用该捷径。

## 严格重启 Round 013 — 纯转向到点后精确朝向（接纳，2026-09-05）

- 目录：`outputs/r2r_curriculum_10ep_20260905/round_013_bare_turn_exact_alignment/`。
- 固定十 EP 从真实起点运行并逐边独立审计：在线 16/20，严格交集同为 16/20（80%），
  连续两段 7/10，首次同时超过 Round008 的 15/20、6/10，接纳为新的前两段冻结基线。
  严格前缀为 EP0=2、EP3=0、EP6=2、EP9=2、EP18=2、EP27=2、EP45=1、
  EP126=2、EP204=2、EP219=1。
- 通用修正：只有完整匹配方向型短句（如 `Turn left.`、`Turn around 180 degrees.`）时，
  点执行器真实到点后才相对已验证源节点 yaw 连续渲染并精确对齐 90°/180°；对齐动作写入
  edge action history，之后才采集节点八视图与做完成判断。含有“穿过房间、到门口”等目的地
  的复合转向不会触发该规则。
- 修复证据：EP0 第二段在首个局部点记录 +90° 并完成，不再游走进浴室；EP126 第一段记录
  −180°，随后第二段真实穿门进入 loft。两者的 `exploration.mp4`、图节点 yaw 与 action
  history 一致。EP45 的复合右转仍保持 unknown，未引入朝向捷径假阳性。128 项契约/单元
  测试通过。
- 剩余漏判：EP219 hop5 实际上升 1.418m 到水平上层平台，完成楼梯位于侧后方，独立审计
  为真，但在线被 side-forward stair detector 命中降为 unknown。后续只让“正前方且标签
  是 stair flight（而不是 stair landing）”成为完成 VLM 的硬否决；侧前方可见刚走完的下行
  楼梯不是“还有楼梯在前”。EP3 中段节点的正前方 `stair flight` 仍会被否决。

## 严格重启 Round 014 — 垂直端点 exact-front flight guard（部分保留，2026-09-05）

- 目录：`outputs/r2r_curriculum_10ep_20260905/round_014_vertical_exactfront_flight_guard/`。
- 在线 17/20；独立审计剔除 EP6 的 BETWEEN_OBJECTS 假阳性后严格 16/20（80%），
  连续两段 7/10，与 Round013 严格指标持平。严格前缀为 EP0=2、EP3=0、EP6=1、
  EP9=2、EP18=2、EP27=2、EP45=1、EP126=2、EP204=2、EP219=2。
- 垂直修正有效且保留：EP219 底部与中段节点先保持 unknown，第 4 个节点上升 1.348m
  到水平上层平台后在线完成；EP3 中段节点仍因正前方 `stair flight` 保持 unknown。
- 新发现的假阳性：EP6 第二段本轮在泳池边普通椅排附近便被 BETWEEN endpoint recovery
  升级，尚未到真正的吧台/椅子夹缝；VLM reason 本身也明确写着“不成对夹住、未观察到
  边界”。其源节点中目标物已在全后方，本边又在前方出现，不能证明进入夹缝。后续为所有
  BETWEEN recovery 增加“源节点的成对参考物仍在前/侧前方”前置门；若源节点只有全后方
  证据，后续前后共现按重复实例或折返歧义处理，不允许确定性升级。129 项测试通过。

## 严格重启 Round 015 — BETWEEN 源节点前向时序门（接纳，2026-09-05）

- 目录：`outputs/r2r_curriculum_10ep_20260905/round_015_between_source_forward_gate/`。
- 在线 18/20；独立审计仅剔除 EP45 复合转向假阳性，严格为 17/20（85%），连续两段
  8/10，超过 Round013 的 16/20、7/10，接纳为新的前两段冻结基线。严格前缀为
  EP0=2、EP3=0、EP6=2、EP9=2、EP18=2、EP27=2、EP45=1、EP126=2、
  EP204=2、EP219=2。
- EP6 对照验证：泳池边椅排的早期节点被保持 unknown，系统继续探索；第 6 个节点抵达
  真正由吧台与座椅夹住的通行区域后才完成，且仍在下一段“吧台拐角”之前。EP219 再次在
  第 4 个节点到水平上层平台并在线完成，EP3 中段仍为 unknown，说明垂直 guard 稳定。
- EP45 在线把床边房内节点报为第二段完成；审计图像、选点/示范轨迹投影以及模型 reason
  均表明尚未到“右侧门口”，因此严格前缀只计 1，未把该假阳性带入指标。
- 前两段已达到 8/10 的稳定门槛。下一轮按逐段课程原则把同一固定十 EP 统一扩展到第三个
  子指令；未通过前缀的 EP3/EP45 仍从真实起点运行，但其后续段不得进入严格有序计分。

## 第三段 Round 016 — 固定十 EP stage2 初始基线（待优化，2026-09-05）

- 目录：`outputs/r2r_curriculum_10ep_20260905/round_016_stage2_initial/`。
- 同一十 EP 全部从真实起点运行，目标统一扩展为前三段，总探索跳数由 7 增至 10；物理到点、
  节点构建、在线完成和独立审计标准均未改变。在线 22/30；严格审计后为 18/30（60%），
  严格完成前三段 2/10。严格前缀：EP0=3、EP3=1、EP6=3、EP9=2、EP18=0、
  EP27=2、EP45=1、EP126=2、EP204=2、EP219=2。
- 新增真实进展：EP0 第三段沿直线越过灰色沙发并完成；EP6 在真实吧台/椅子夹缝节点之后
  到吧台拐角完成 STOP_WAIT，两者成为首批严格三段完成。EP3 使用额外预算后在 hop7 到达
  y=3.62m 的水平上层平台，第一段从此前的 0 提升为严格 1，但已无足够有效边继续第二段。
- 主要第三段失败：EP9 对“阳台下入口”多次选点后走进卧室/走廊，属于选点方向漂移；
  EP27、EP126、EP219 的首个第三段点出现无安全点或物理失败且恢复失败；EP204 沿红毯
  前进但未到末端，仍被 BETWEEN endpoint recovery 误报完成。
- 前缀回归：EP18 在首个约 2.9m 边仅凭沙发出现在后方便被 CIRCUMNAVIGATE override
  过早推进，实际还未到沙发后方；EP45 仍在床边误报第二段，后续不得计分。
- 已从上述 failure case 提取两条通用门并通过 131 项测试，尚待下一轮十 EP 回归：
  (1) CIRCUMNAVIGATE 的 rear-present 恢复必须有关键帧时序支持，不能只凭后方共现和位移；
  (2) BETWEEN_OBJECTS 若语义目标明确包含 `end/far end/until/all the way`，partial 状态或
  未观察到终点边界时不得确定性恢复，普通“进入两物体夹缝”不受此终点门影响。

## 第三段 Round 017 — 关键帧/终点边界回归（接纳，2026-09-05）

- 目录：`outputs/r2r_curriculum_10ep_20260905/round_017_stage2_keyframe_terminalguard/`。
- 固定十 EP 全部从真实起点统一运行前三段；54 次点导航尝试中执行器发出 44 次到达，
  42 次满足隐藏的 0.75m 物理核验。在线节点判定完成 21/30；逐节点六视图、edge
  keyframe、action history 与事后示范路径对齐审计后，严格有序前缀为 19/30（63.3%），
  高于 Round016 的 18/30，接纳为新的第三段基线。严格前缀：EP0=3、EP3=1、EP6=3、
  EP9=2、EP18=0、EP27=2、EP45=1、EP126=2、EP204=2、EP219=3。
- 两条待验证修正确实生效：EP18 的四个首段节点全部保持 unknown，未再仅凭后方沙发
  共现报完成；EP204 首两段通过后，在红地毯第三段的其余七次判断均保持 unknown，
  未观察到地毯末端/下一门口时不再被 BETWEEN recovery 升级。
- 新增严格进展：EP219 在拒绝两个楼梯中段节点后到达上层水平平台，随后完成 hard-left
  朝向并面向通往 workout room 的连续走廊，第三段严格通过。EP0、EP6 的前三段基线保持。
- 独立审计拒绝两个新的 ENTER_REGION 假阳性：EP9 第三段实际停在普通室内拱形走廊，
  不在 balcony 下方，且偏离最后参考 waypoint 3.97m；EP27 第三段进入带沙发/壁炉的
  横向房间而不是 cardboard-box room，偏离对应参考 waypoint 4.48m。两者均只计前两段。
- 从这两例提取通用门：v21 的 ENTER_REGION endpoint recovery 不得把 `partial`、
  `boundary_event=not_observed` 升为完成；`room with X`、`entrance under/beneath X`、
  `containing X` 等带限定地标的区域，当前节点必须有 RGB 对齐 open-vocabulary detection
  支持非通用限定 token，不能只凭 doorway/room 转换或 VLM 文本幻觉完成。规则不使用
  深度、navmesh、示范路径或 EP 编号；新增两项定向测试，全量 133 项测试通过。该后续
  修正尚待下一次固定十 EP 同步回归。

## 第三段 Round 018 — ENTER_REGION 限定地标门控回归（判定修正保留，2026-09-05）

- 目录：`outputs/r2r_curriculum_10ep_20260905/round_018_enter_region_qualifier_guard/`。
- 固定十 EP 从真实起点同步执行前三段。65 次点导航中执行器到达 50 次，48 次通过隐藏
  0.75m 物理核验；在线完成 20/30，独立审计后的严格有序前缀仍为 19/30（63.3%）。
  严格前缀：EP0=3、EP3=1、EP6=3、EP9=2、EP18=1、EP27=2、EP45=1、
  EP126=2、EP204=2、EP219=2。与 Round017 总数持平，未替代其更好的 3 段 EP 数，
  但两个完成判定门控通过真实回归，应保留。
- EP9 在 balcony 只位于前方、普通卧室/走廊或低置信度侧后方的 8 个后续事件中全部
  保持 unknown；EP27 在无 cardboard/boxes 检测的 stained-glass 楼梯区保持 unknown。
  Round017 的两个 ENTER_REGION 假阳性均消失。EP204 也继续在未见红毯末端时保持
  unknown，说明不同形式门控未相互旁路。
- EP18 首段回到 Round015 已验证的沙发后开放区；第二段在线在厨房反方向的 counter
  端点报完成，但节点 `z=-17.13`，距离事后参考路线 4.77m，而正确延伸保持在 `z≈-12`，
  因方向错误被审计拒绝。EP219 前两段保持，第三段两次点执行均未到达，未建节点。
- 新暴露的 outer-policy 问题：`infer_unknown_disposition` 把任何移动超过 0.5m 且没有显式
  reverse/stall 的 `partial` 或 `unsatisfied` 都当作 `on_route`，每跳会清空分支状态；因此
  EP9/204 连续 6--8 个 unknown 仍得到 `blocked_yaws_by_verified_node={}`。已通用收紧为仅
  `partial + instructed order + keyframe support + same instance + motion supports + real move`
  才继续；ambiguous/unsatisfied/无关键帧进入“一次额外探索、仍错则回溯并屏蔽”。不使用
  深度、示范路径或 EP 信息；新增定向测试后全量 134 项通过，待 Round019 十 EP 回归。

## 第三段 Round 019 — 激进 unknown 回溯实验（不接纳，2026-09-05）

- 目录：`outputs/r2r_curriculum_10ep_20260905/round_019_unknown_recovery_blocking/`。
  固定十 EP 均从真实起点完成运行并生成视频，但在线完成上限仅 16/30：EP0=3、EP3=1、
  EP6=1、EP9=2、EP18=0、EP27=2、EP45=1、EP126=2、EP204=2、EP219=2；仅
  EP0 在线走完前三段。该在线上限已低于冻结严格基线 19/30，因此无需用事后审计扩大
  指标，也绝不接纳为新基线。
- 回溯机制首次真实启动：共写入 8 个 blocked direction；EP9、EP204 各成功回溯一次，
  证明“一次额外探索—回溯—屏蔽”的控制链能运行。但把缺关键帧、实例歧义或
  `unsatisfied` 的真实移动一律判为 wrong 过于激进：EP6 沿吧台/椅子区域的正常渐进节点和
  EP18 绕沙发的两个节点被提前切断，分别从此前 3/1 段退化为 1/0 段；部分回溯还因实际
  revisit 超出 0.75m 而终止。
- 通用结论：不能依靠提高单节点语义置信门来同时解决“渐进过程”和“无限漂移”。已改为
  分层约束：真实平移且无 reverse/stall 的 partial/unsatisfied 仍可暂记为 on-route，但每个
  子指令至多连续接受 2 次；第 3 个 unknown 成为一次额外探索起点，第 4 个仍 unknown 才
  回溯并屏蔽。成功回溯后保留屏蔽方向并重置试探状态，使新方向获得完整的一次试探，而
  不是立刻再次回溯。完成子指令也会清零连续计数。规则不使用深度、示范路径、EP 编号或
  case 文本；新增状态机契约后全量 135 项测试通过，待 Round020 固定十 EP 同步回归。

## 第三段 Round 020 — 有界 unknown 进展回归（不接纳，2026-09-06）

- 目录：`outputs/r2r_curriculum_10ep_20260905/round_020_bounded_unknown_progress/`。
  固定十 EP 全部从真实起点运行并生成视频，进程成功率 10/10；47 次前向点导航中执行器
  到达 39 次，隐藏 0.75m 物理核验到达 38 次。在线完成仅 17/30：EP0=3、EP3=0、
  EP6=3、EP9=2、EP18=0、EP27=2、EP45=1、EP126=2、EP204=2、EP219=2。
  在线上限已低于冻结严格基线 19/30，故不做可计分语义审计，也不接纳本轮。
- 有界策略部分达到预期：EP6 在第二段连续出现 3 个 on-route unknown 时，前两个继续，
  第三个进入一次额外探索，第 4 个节点完成该段，最终恢复到严格基线中的在线 3 段；不再
  复现 Round019 只完成 1 段的退化。EP9/EP18 在 2+1 unknown 后均实际请求了回溯，说明
  无限清空分支状态的问题已消失。
- 新的主瓶颈是回溯落点而不是状态机触发：EP9 回溯节点最终相距 0.949m，EP18 为
  0.877m，均在视觉相似度 >0.99 时仍因严格 0.75m 几何门失败；EP204 更明显，当前距待回
  节点约 0.90m，但恢复器选择了初始 geodesic 4.94m/5.30m 的远地面点，随后受回溯预算
  限制停在约 0.90m。EP219 第二次恢复同样停在 0.831m。这里不放宽 0.75m 到达标准。
- 已做通用修正：回溯选择器不再取每个视角第一个“深而宽”的可达地面最大值，而是对 12
  个空间分散、可投影且 navmesh 可达的 RGB 地面 anchor，优先选世界投影最接近已存节点
  或历史 breadcrumb 的点；像素、深度投影与传给执行器的 navmesh 目标保持同源。该几何
  只用于已明确目标节点的回溯，不参与 VLM 前向语义选点，也不使用 R2R 示范路径。新增
  “远高分点不得压过近节点 anchor”契约后全量 136 项测试通过，待 Round021 十 EP 回归。

## 第三段 Round 021 — 回溯节点近邻 anchor 回归（修正保留，整轮不接纳，2026-09-06）

- 目录：`outputs/r2r_curriculum_10ep_20260906/round_021_backtrack_node_proximity/`。
  固定十 EP 均从真实起点运行，10/10 视频可解码（960x480，共 121--591 帧）；生成 51 张
  arrival 六视图审计图。51 次前向点导航中执行器到达 46 次，隐藏 0.75m 物理到达 43 次。
  在线完成 20/30；写入十份独立 RGB/action/reference-trajectory 审计并由官方汇总器重算后，
  严格有序前缀为 18/30（60%）：EP0=3、EP3=0、EP6=3、EP9=2、EP18=1、EP27=2、
  EP45=1、EP126=2、EP204=2、EP219=2。低于 Round017 的冻结 19/30，整轮不接纳。
- 回溯修正有效且保留：EP9 的序列回溯成功，写入 2 个 blocked yaw 并继续到新分支；
  EP204 从 Round020 的首次 live recovery 失败变为成功回溯、屏蔽并继续，共运行 8 跳；
  EP126 的物理失败恢复也由失败变成功。EP6 的 3 段冻结路径未受影响。整批 recovery
  success 为 6/11，blocked direction 为 4；仍有 EP18 的 sequence backtrack 和若干后续
  physical recovery 失败，需继续优化，但不能靠放宽 0.75m 阈值。
- 独立审计拒绝两个在线假阳性：(1) EP9 hop6 是普通拱形门厅，位置
  `[19.13, 0.13, -1.27]`，距示范终点约 4.4m，并非 balcony 下入口；(2) EP18 hop2 仅进入
  厨房并到台面起段附近 `[-5.42, 0.15, -12.10]`，尚未沿吧椅/台面走到末端参考区域。
  因此 online 20 不得当作 strict 20。官方结果只计 EP6 为当前三段评估范围内同时满足
  有序完成与 goal-radius 的有效 R2R success；EP0 的几何 goal hit 因未评完整指令仍无效。
- 从 EP18 提取通用原子复合段门：若同一子指令明确要求进入区域后继续 `along/follow` 某
  地标直到 `end/far end/entire length`，仅进入区域是 partial；当前 RGB 对齐的指令相关
  地标仍以 >=0.30 出现在 exact FRONT 时，终点仍在前，必须 unknown。侧面/后方保留地标
  不否决真实角点/终点。该门不使用深度、示范路径或 EP 信息，并写入 completion prompt；
  新增集成契约后全量 137 项测试通过，待 Round022 固定十 EP 回归。

## 第三段 Round 022 — 原子复合段终点门回归（判定修正保留，整轮不接纳，2026-09-06）

- 目录：`outputs/r2r_curriculum_10ep_20260906/round_022_compound_terminal_extent_guard/`。
  固定十 EP 从真实起点全部运行且视频齐全。56 次前向点导航中执行器到达 46 次、隐藏
  0.75m 物理到达也为 46 次；在线完成 18/30：EP0=3、EP3=0、EP6=3、EP9=2、
  EP18=0、EP27=2、EP45=2、EP126=2、EP204=2、EP219=2。在线本身已低于冻结严格
  基线 19/30，故无需进行可计分审计即可判定整轮不接纳。
- 新终点门达到针对的通用目标：EP18 不再在“刚进入厨房、台面仍位于正前方”的节点把
  “进入厨房并沿吧椅/台面走到末端”误报完成；系统保持第 0 段继续探索。代价不是该门
  错杀，而是本轮 CIRCUMNAVIGATE VLM 在相同已验证位置把关键帧/实例关系持续标为 ambiguous，
  首段未闭合。EP45 在线第二段仍需独立审计，不能因 online=2 直接视作修复。
- bounded recovery 的控制链稳定运行：EP9 两次序列回溯均成功并写入 4 个 blocked yaw，
  到 10-hop 上限仍未找到 balcony 下入口；EP18 的序列回溯也成功。本轮全部 recovery
  success 10/15，blocked direction 8，显著证明回溯触发与屏蔽已不再是空逻辑。
- 失败记录暴露下一层通用问题：已知节点恢复仍以目标的欧氏直线 bearing 约束候选视角；
  当实际 navmesh 路径先绕墙角时，正确首段方向会被 100° gate 排除。EP27/204/219 的重试
  均出现“目标节点约 0.9--1.2m，但所选 floor anchor 沿错误直线方向”的模式。下一步仅对
  已存节点回溯使用 navmesh shortest-path 的首个可分辨拐点作为 desired bearing；不改变
  前向 VLM 选点、不使用 R2R 示范路径，也不放宽节点 0.75m 到达门。

## 第三段 Round 023 — navmesh 首拐点回溯回归（回溯修正保留，整轮不接纳，2026-09-06）

- 目录：`outputs/r2r_curriculum_10ep_20260906/round_023_navmesh_corner_recovery/`。
  固定十 EP 全部从真实起点完成运行并生成视频。59 次前向点导航中执行器到达 46 次，
  隐藏 0.75m 物理到达 48 次；在线完成 17/30：EP0=3、EP3=0、EP6=3、EP9=3、
  EP18=0、EP27=2、EP45=2、EP126=2、EP204=2、EP219=0。在线上限低于冻结严格
  基线 19/30，故不做可计分审计，整轮不接纳。
- 回溯子模块的通用修正有效并保留：总 recovery success 从 Round022 的 10/15（66.7%）
  提升到 16/19（84.2%）；EP126 为 5/5、EP204 为 4/4、EP219 为 4/4。三者不再因墙角
  后方的欧氏 bearing 选错恢复视角而终止，blocked direction 总数也增至 10。改动仅使用
  当前 pose 到已存节点的 navmesh 最短路首个可分辨拐点，不使用示范轨迹，节点到达仍是
  严格 0.75m；138 项测试通过。
- 改善回溯后，主要瓶颈转为节点完成判定的跨调用波动：EP219 有 9/10 隐藏物理到达且
  4/4 recovery 成功，最终几何 goal hit，但三段在线均保持 unknown，因此严格上仍是无效；
  EP18 同样完成多次到点/一次回溯后仍未闭合首段。EP9 在线第三段完成与 Round021 相同，
  其普通拱厅语义此前已审计为假阳性，不能计分；EP45 在线第二段也仍需审计，不能当作
  新进展。下一轮应在不泄露示范路径的条件下稳定完成证据/多次判断聚合，而不是继续扩大
  回溯或探索预算。

## 第三段 Round 024 — 出口语义 taxonomy 规范化回归（修正保留，整轮不接纳，2026-09-06）

- 目录：`outputs/r2r_curriculum_10ep_20260906/round_024_exit_semantic_normalization/`。
  固定十 EP 全部从真实起点运行且 10/10 视频可解码（960x480，共 121--608 帧）。在线
  完成 18/30：EP0=3、EP3=0、EP6=3、EP9=2、EP18=0、EP27=2、EP45=2、
  EP126=2、EP204=2、EP219=2。在线总数低于冻结严格最佳 Round017 的 19/30，故
  不做全量语义审计，整轮不接纳；两个几何 goal hit 仍因完整指令未评完而明确记为无效。
- 通用完成判定修正命中了预期 failure：冻结旧拆解把显式“走出房门到 outside”归为
  `OTHER`，此前虽施加了出口防误报门，却只有正式 `EXIT_REGION` 能使用“同一门从前到后、
  有序关键帧和真实平移”的正向完成证据。规范化后，只有明确含 outside/outdoor/exit/leave
  room 的 `OTHER` 可使用同一严格时序门；仅“approach doorway”的普通 `OTHER` 不会被
  提升。EP219 从 Round023 的 0/3 恢复至 2/3，其中第 0 段在首个已由 Round017 独立审计
  验证过的同位置节点闭合，第 1 段由完整楼梯顶端证据闭合。正反契约及全量 140 项测试通过。
- 本轮 59 次前向尝试中执行器到达 52 次、隐藏 0.75m 物理到达 51 次，物理到达判定精度
  98.3%；但 recovery 仅 7/13（53.8%），并造成 5 个 EP 以 physical recovery failed
  结束。失败记录揭示跨视角近端排序缺口：EP27 距目标节点 geodesic 0.88m 时，存在距节点
  0.63m 的可达 anchor，却选了距节点 1.70m 的高 hybrid-score 点；EP204 在 1.17m 时同样
  放弃 0.93m anchor 而选择 2.32m 点，第二次恢复反而走远。
- 已新增通用但尚未做十 EP 运行回归的近端规则：仅当 live recovery 到已知节点的 navmesh
  geodesic <=1.5m 时，跨允许视角按 anchor 到节点的距离优先；远距离仍使用方向/视觉 hybrid
  排序。节点成功标准保持 0.75m，不使用示范轨迹，也不影响前向语义选点。新增近/远正反契约
  后全量 142 项测试通过，待 Round025 固定十 EP 同步回归。

## 第三段 Round 025 — 近端节点 anchor 跨视角排序回归（恢复修正保留，整轮不接纳，2026-09-06）

- 目录：`outputs/r2r_curriculum_10ep_20260906/round_025_short_range_node_proximity/`。
  固定十 EP 全部从真实起点运行，10/10 三格视频可解码（960x480，共 121--603 帧）。在线
  完成 19/30：EP0=3、EP3=0、EP6=3、EP9=2、EP18=0、EP27=2、EP45=2、
  EP126=3、EP204=2、EP219=2。十份独立 RGB/action/reference-path 审计落盘并由官方
  汇总器重算后，严格有序前缀仅 15/30（50%）：EP0=3、EP3=0、EP6=1、EP9=2、
  EP18=0、EP27=2、EP45=1、EP126=2、EP204=2、EP219=2。低于冻结 Round017 的
  19/30，整轮不接纳；3 个几何 goal hit 均因完整指令未评完继续记为无效。
- 恢复子模块改善明确且保留：recovery 从 Round024 的 7/13（53.8%）升至 16/20（80%）。
  EP18 为 4/4、EP219 为 5/5、EP27 为 3/4。EP27 两次首次停在 0.852m/0.755m 的重访，
  经 `short_range_node_proximity_priority=true` 后分别到 0.617m/0.362m，另一次直接到
  0.266m；EP204 首次 1.127m 后修到 0.077m。所有成功仍严格使用 0.75m 节点门。
  EP204 另一处 0.777m 失败时，当前所有可见地面 anchor 本身都在约 0.90m 以外，说明
  剩余近端问题是缺少节点半径内可见 anchor，而非跨视角排序。
- 严格审计拒绝三类在线早闭合：(1) EP6 在 `[-0.63,0.05,2.11]` 即把“walk between the
  bar and chairs”判完，但吧台/椅子仍在前方，且示范进入 gap 的下一 waypoint 约为
  `[0.03,0.10,-0.15]`；因此严格前缀停在 stage0，后续 bar-corner 节点也失去有序边界。
  (2) EP45 stage1 横穿卧室后，未满足“doorway directly ahead、bed to the left”的完整
  当前关系。(3) EP126 stage2 到 `[-2.33,3.62,1.22]`，而示范左转/loft far-side 段经
  `[-0.47,3.62,7.63] -> [0.58,3.62,7.20]`，属于方向相反的错误侧。以上均未计分。
- 新的优先优化对象是完成判定的结构不对称：`BETWEEN_OBJECTS` 的 VLM unknown 路径会
  额外检查对象夹持关系，但某些 endpoint recovery/模型完成路径仍可在目标主要位于前方时
  提前闭合；复合 TURN 指令也需把实际有符号 heading/action 与最终语义关系共同作为硬门。
  下一轮应先构造通用正/负节点契约验证这些门，不能只针对 EP6/45/126 文本做补丁。

## Round 026 — 全子指令端到端诊断（正式严格审计，2026-09-06）

- 目录：`outputs/r2r_curriculum_10ep_20260906/round_026_full_end_to_end_diagnostic/`。
  本轮不传 `--targets`，从真实起点执行十 EP 的全部 37 个冻结子指令；其余模型、阈值、
  10-hop 总预算和恢复预算均与当前正式配置一致。10/10 三格视频可解码（960x480，
  157--603 帧）。
- 在线完成 17/37；十份独立 RGB/action/reference-trajectory 审计加载后，严格有序完成
  14/37（37.84%）：EP0=4/4、EP3=0/3、EP6=1/3、EP9=2/4、EP18=0/3、
  EP27=2/4、EP45=1/4、EP126=2/4、EP204=2/4、EP219=0/4。完整严格序列与正式
  R2R success 都只有 EP0，即 1/10。EP0 四跳完成并停在 rug 区域，final geodesic
  1.27m。几何 goal hit 共 3 条；EP9 与 EP219 因完整有序序列未闭合而保持无效。
- 基础模块整体已相对稳定：66 次前向点导航中隐藏 0.75m 物理到达 58 次（87.88%）；
  点到达判定准确率 98.48%、precision 98.31%、recall 100%；节点恢复 14/17（82.35%）。
  因此本轮主瓶颈不是一般性的“走不到已选点”。底层明确终止只有 EP3、EP204 的
  physical recovery failed，以及 EP27 最后一次 sequence backtrack failed。
- 最大瓶颈是后段语义选点/完成边界：
  1. EP6 在线 3/3 但严格仅 1/3、final geodesic 4.46m；BETWEEN_OBJECTS 在真正穿过
     bar/chair gap 前提前闭合，令后一段也整体前移。
  2. EP9 用满 10 hop 后在线 3/4且进入 goal radius，但第三段节点仍是普通拱门楼梯门厅，
     不是 balcony 下入口；严格仅 2/4，几何命中无效。
  3. EP18 的恢复 2/2，但八视角复选三次都以“没有与首个后续地标一致的地面视角”失败；
     EP45 同样以 no-safe-point 结束。二者是语义选点约束/召回问题。
  4. EP126 恢复 2/2且屏蔽两组方向，仍在第三段用满 10 hop；EP204 第三段走向错误区域，
     final geodesic 从 11.51m 恶化到 14.10m。复合方向段的选点仍不能稳定结合来向。
  5. EP219 最具诊断性：8/10 物理到点、恢复 4/4、最终进入 2.20m goal radius，但完成
     判定在已被此前严格审计验证的第一个出口节点仍返回 unknown，游标始终停在 stage0，
     严格 0/4。这是完成判定跨调用假阴性，而不是导航或回溯失败。
- 控制效率也是次级瓶颈：EP9、EP126、EP219 均在 10-hop 上限结束；第三段反复 unknown/
  错误分支消耗预算，使最终停止段通常根本没有获得执行机会。下一步应优先修复通用的
  BETWEEN/复合 turn 硬边界以及相同节点证据的判定稳定性，再评估是否需要按剩余子指令数
  分配 hop，而不能先靠扩大总预算掩盖错误方向。

## Active-STOP Round 000 — 全链路冻结基线（2026-09-07）

- 测试目录：`outputs/r2r_curriculum_10ep_active_stop_20260907/round_000_baseline_active_stop/`。
- 固定十 EP：`0,3,6,9,18,27,45,126,204,219`，全部从数据集真实起点运行。
- 冻结配置：DeepSeek VLM、V31 选点、V19 完成判定、dense-majority 地面、
  DINO+SAM 指令语义检测、GNM、TAPIR V13、CUDA:0、seed 17、最多 30 hop。
- 严格成功：1/10（仅 EP6）；STOP 3 次，其中 EP3/EP9 在目标半径外；EP219 仅几何
  命中但没有完成序列与主动 STOP，明确不计成功。
- 在线完成前缀：EP0=1/4、EP3=3/3、EP6=3/3、EP9=4/4、EP18=0/3、
  EP27=1/4、EP45=3/4、EP126=0/4、EP204=2/4、EP219=1/4。
- 事后 GT 审计（GT 未输入在线模块）：92 个点中选点方向误差不超过 30 度为
  29/92，真实边位移同时满足误差不超过 30 度且沿参考折线正向推进为 20/92。
- 协议核验：固定十 EP、同一冻结配置、CUDA 推理、视频齐全、主动 STOP 语义、
  到达建点合同、点簇严格落地合同均通过；最终 10/10 门槛失败。
- 通用失败 taxonomy：错误语义闭合后 STOP、纯转向局部落点无解、PASS/FOLLOW
  安全距离上限过短、长时间 unknown 后跨方向漂移、点簇丢失/恢复失败。
- 本轮只冻结为诊断基线，不接纳为成功配置。证据：`protocol_audit.json`、
  `round_report.md`、逐 EP `trajectory.json` 与 `exploration.mp4`。

## Active-STOP Round 001 — 五次通用迭代回归（2026-09-07）

- 测试目录：`outputs/r2r_curriculum_10ep_active_stop_20260907/round_001_five_iteration_unified_route/`；
  固定十 EP 全部从数据集真实起点运行，三个 CUDA:0 worker，同一冻结配置，10/10 视频
  可解码，未向在线模块暴露 GT/reference path。
- 五次改动包括事后 GT 审计、TURN_AROUND 局部点半径、PASS/ADVANCE/FOLLOW 前向点
  半径、V34 统一路线 guard、V23 终点关系判定。160 项回归测试通过。
- 严格 simulator success 仍为 **1/10（仅 EP6）**；EP0、EP3 虽由系统完成游标并主动
  STOP，但 STOP 分别距目标 8.759m、12.433m，明确按失败处理。其余七条没有完成序列；
  EP219 本轮不再出现“仅几何命中”。
- 事后 GT 审计：120 个点中选点方向误差不超过 30 度为 **31/120**，真实执行边同时
  满足 30 度和正向进展为 **26/120**。选点比例从基线 29/92（31.5%）退化到 25.8%，
  执行边比例约持平（21.7%）；V34 不得冻结。
- 通用根因：纯方向 TURN 可先平移约 5.5m 再对齐朝向；多地标 PASS 的 override 只需
  任一地标在后方；CIRCUMNAVIGATE 仅凭终点后视和短距离平移就覆盖模型的 partial；
  两次 on-route unknown 之后仍会再放行 off-sequence hop；最终 pixel ray 没有完整继承
  history-safe incoming/blocked guard。这些共同造成错误游标推进和长时间离轨恢复。
- 所有协议不变量通过，但最终 10/10 门槛失败。本轮仅作为 failure evidence，下一组
  五次迭代必须修正以上形式级/状态机级问题并重新覆盖同一十 EP。

## Active-STOP Round 002 — 严格路线合同五次迭代回归（2026-09-07）

- 测试目录：`outputs/r2r_curriculum_10ep_active_stop_20260907/round_002_five_iteration_strict_route_contract/`；
  固定十 EP 全部从真实起点运行，代码在整轮期间冻结，三个 CUDA:0 worker，10/10 视频
  齐全。GT/reference path 仅由结束后的审计器读取，未进入在线选点、执行或节点判定。
- 五次通用改动为：拒绝退化的 V34 并恢复 V31 骨架、纯 TURN 目标半径限制、V31 最终
  pixel-ray 来路/blocked guard、PASS/CIRCUMNAVIGATE 完成门收紧与有符号转向动作门、以及
  UNKNOWN 后严格一次 lookahead 再回溯。全量 165 项单元/合同测试通过。
- 严格 simulator success 仍为 **1/10（仅 EP6）**。只有 EP3、EP6 主动 STOP；EP3 虽在线
  完成 3/3，但第二段首次真实边相对 GT 偏 86.5 度并进入错误路线，最终在错误 closet
  前闭合第三段，STOP 距目标 13.02m，明确失败。在线完成前缀为 EP0=1/4、EP3=3/3、
  EP6=3/3、EP9=1/4、EP18=1/3、EP27=1/4、EP45=2/4、EP126=2/4、EP204=2/4、
  EP219=1/4。
- 后验方向审计为选点 **27/102（26.5%）**、真实执行边同时满足 30 度且沿 GT 正向推进
  **20/102（19.6%）**。相较 Round001 的 25.8%/21.7%，选点略升但真实推进下降，未构成
  可接纳改进。
- 通用失败分为三组：EP0/45/126/219 的合法候选被路线/历史门耗尽；EP9/27/204 回溯最终
  失败且 EP18 在重复 UNKNOWN/回溯中耗尽 30 hop；EP3 在早期方向偏离后对错误同类地标
  产生局部自洽的完成误判。形式上 ENTER_REGION 仅 3/23 边正确推进，APPROACH_LANDMARK
  为 0/10，PASS_LANDMARK 为 3/15，是下一轮首要聚合对象。
- 固定配置、CUDA 推理、视频、主动 STOP、到达建点和双地面点簇合同全部通过。分片 driver
  的外层 stdout 本轮误重定向到 `/dev/null`，但逐 EP `process.log`、trajectory、视频及根级
  audit/process log 完整；该记录缺口不影响行为结果，下轮恢复 driver log。最终 10/10 门失败，
  本轮不冻结为成功配置。

## Active-STOP Round 003 — 分支原点与语义路线承诺五次迭代回归（2026-09-07）

- 测试目录：`outputs/r2r_curriculum_10ep_active_stop_20260907/round_003_five_iteration_branch_origin_route_commit/`；
  固定十 EP 全部从真实起点运行，三个 CUDA:0 worker，代码与配置整轮冻结，10/10 视频齐全。
  GT/reference path 仅在全部运行结束后由审计器读取，未进入在线选点、执行、节点判定或恢复。
- 五次通用改动为：显式转向只重开 soft incoming cone、未来子句地标降为 tie-break、保留高置信
  且序列一致的 VLM 路线、物理重访继承 UNKNOWN 分支原点与全部 blocked yaw、回溯预算加入一个
  离散控制步和 settle margin。全量 171 项测试通过。
- 严格 simulator success 仍为 **1/10（仅 EP0）**。EP0 从上一轮 1/4 提升为 4/4 并主动
  STOP，距终点 2.59m；但 EP6 从上一轮唯一成功回退为 1/3。EP126 在线 4/4 且主动 STOP，
  但距终点 5.53m，明确失败。EP6、EP45 虽分别进入 1.76m、0.32m goal radius，却未完成完整
  指令和主动 STOP，也明确不计成功。
- 后验方向审计为选点 **20/85（23.5%）**、真实执行边同时满足 30 度且沿 GT 正向推进
  **18/85（21.2%）**。上一轮为 26.5%/19.6%；执行比例小幅增加但选点下降，且成功数未升，
  因此不接纳为通过配置。
- 通用失败聚合为三组：EP6/18/27/45/219 在仍需探索时被 history-safe/final RGB-ray/floor-bearing
  候选耗尽；EP9/204 回溯失败；EP3 在第三段对几乎相同的后向方位连续重选直到 30 hop。
  此外 EP126 的在线完成属于错误路线上的局部语义闭合，暴露“节点判定通过但全局路线错误”。
- 固定十 EP、同一配置、CUDA、本地小模型、视频、主动 STOP、到达建节点与严格双地面点簇合同
  全部通过。最终 10/10 门失败；下一轮必须从十条共同证据修复候选耗尽、方向去重/分支记忆、
  回溯失败与完成判定的路线一致性，不做 episode 文本或坐标补丁。

## Active-STOP Round 004 — 地面召回与失败方向记忆五次迭代（外部 API 中断，不计分，2026-09-07）

- 五次通用改动及 177 项测试已完成：物理失败射线封锁、完整严格地面 mask 安全重采样、
  soft-incoming 切线回退、近端已存节点末端引导、复合转向语义终点硬门。固定十 EP 按三
  CUDA:0 worker 启动，配置与代码已冻结。
- DeepSeek 在运行中开始统一返回 HTTP 402 `Insufficient Balance`。余额耗尽前完整且未受污染的
  EP 只有 0、3、6；EP9、27 在中途调用失败，EP18、45、126、204、219 从首个调用即失败。
  因此本轮没有合法的十 EP 分母，不计算成功率、不运行正式汇总审计，也不与历史轮次比较。
- 三条有效局部证据：EP0 继续保持 4/4、主动 STOP、模拟器成功；EP6 仍为 1/3，物理失败
  射线已正确写入 block，但唯一关系地面点只有 0.57m、低于当前 0.75m 新节点进展门，随后
  无安全点；EP3 从上一轮 30-hop 同向重试缩短到 10-hop 后无安全点，证明失败方向记忆消除了
  无限同射线 churn，但尚未解决正确路线召回。
- 本轮状态固定为 `invalid_external_api_interruption`。恢复 DeepSeek 余额/有效 key 后必须从十个
  数据集起点整轮重跑，不能把上述部分结果拼接成正式 Round004。
- 中断后补充了不计入五次算法迭代的协议修复：缺失密钥以及 DeepSeek HTTP 401/402/403 会记为
  `provider_fatal`，不再被导航状态机吞为普通选点失败；对应 batch shard 会立即以 code 86 停止并
  写独立中断文件，避免继续产生污染轨迹。全量 **181** 项回归测试通过。Round004 的冻结代码哈希
  与无效状态保持不变；恢复服务后仍需从十个起点重跑整轮。

## Active-STOP Round 004b — 充值后完整干净重跑（2026-09-07）

- DeepSeek 多模态健康检查恢复后，在全新目录从固定十个数据集起点完整重跑 Round004 的同一组
  五次通用改动；三个 CUDA:0 worker，未复用 402 中断轮的任何轨迹。十个进程均正常返回，未出现
  provider-fatal。
- 严格 simulator success 为 **1/10（仅 EP0）**；只有 EP0 完成 4/4、主动 STOP 且 STOP pose
  位于 Habitat goal radius。EP6、EP126 虽几何进入目标半径，但未完成完整序列且未主动 STOP，
  明确不计成功。
- 在线完成前缀：EP0=4/4、EP3=2/3、EP6=2/3、EP9=2/4、EP18=1/3、EP27=1/4、
  EP45=1/4、EP126=2/4、EP204=2/4、EP219=1/4。
- 后验 GT 审计（从未输入在线模块）：57 个点中选点方向误差≤30°为 **22/57（38.6%）**；
  真实执行边同时满足≤30°且沿参考轨迹正向推进为 **17/57（29.8%）**。
- 主要通用失败为 safety/history gate 耗尽：EP9/18/27/45/126/204 都终止于最终 RGB 射线没有
  history-safe 地面锚点，EP3 终止于无安全点，EP219 终止于显式方向门内无地面候选；此外 EP6
  的物理失败回溯未恢复。固定十 EP、冻结配置、CUDA、本地小模型、视频、主动 STOP、到达建点、
  严格地面点簇合同通过。
- 补强协议审计后，EP0 暴露为“模拟器 STOP 成功但非逐段验收成功”：4 个系统完成边中只有 2 个
  同时满足≤30°与 GT 正向推进，而且没有逐段独立核验文件。因此新增两项硬协议——成功声明必须
  具有独立逐段核验、所有完成边必须 GT 对齐——本轮二者均失败，最终 instruction-validated +
  aligned success 为 **0/10**。以后 simulator success 与最终验收成功分开报告。

## Active-STOP Round 005 — 切线恢复、路线承诺、同边 STOP 与回溯 v5（2026-09-07）

- 固定十 EP 从数据集起点在三个 CUDA:0 worker 上完整重跑，191 项测试通过，10/10 轨迹与视频
  齐全，无 DeepSeek provider error。在线模块未加载 reference path；GT 仅由结束后的审计器读取。
- 五项通用改动为：blocked-direction 切线地面候选、supported-partial 前向承诺、楼梯连续段不重复
  消费初始转向、相邻 STOP_WAIT 在同一真实到达边上再次语义判定、backtrack stored endpoint v5。
- 模拟器 active-STOP 成功仍为 **1/10（EP0）**；EP3 也在线完成 3/3 并 STOP，但距目标很远，
  不计模拟器成功。独立 RGB/action 段核验与所有完成边 GT 方向/进度门相交后，最终有效成功仍为
  **0/10**。EP0 仅 2/4 完成段通过，EP3 仅 1/3 通过。
- 后验方向审计为选点 **27/93（29.0%）**、真实执行边同时满足≤30°且正向推进
  **24/93（25.8%）**。相比 Round004b 的 38.6%/29.8% 均退化，不能接纳本轮策略。
- 切线恢复把“立刻无锚点”变成更长的错误分支搜索：目标数从 57 增到 93，EP6/9/45/126/204/219
  最终仍耗尽五个阻塞方向，EP27 回溯失败。大量 PASS/BETWEEN/ENTER/TURN 的第一或第二个边本来
  沿 GT 正向，完成判定返回 partial/unknown 后，后续一次 trial 被封锁并开始扫描反向分支；说明
  下一轮应提高关系边界识别和一次 trial 的端点质量，并收紧切线恢复到语义路线，不应继续全景放宽。
- 物理执行器不是主瓶颈：多数 shard 的到点判定准确率为 90%–100%；主要瓶颈为恢复后的选点方向、
  BETWEEN/PASS/ENTER 等完成边界假阴性，以及阻塞集合最终覆盖所有候选。证据位于本轮目录的
  `protocol_audit.json`、`round_report.md`、逐段 `stage_completion_verification.json` 和视频。

## Active-STOP Round 006 — 在线走廊约束与关系边界恢复（2026-09-07）

- 固定十 EP、三个 CUDA:0 worker、194 项测试、10/10 轨迹与视频均通过运行协议；GT 仍仅用于后验。
- 模拟器 active-STOP 成功 **1/10（EP0）**，逐段独立验证且完成边全部 GT 对齐的有效成功仍为
  **0/10**。EP9 在线前缀由 2/4 提升为 3/4，其余未形成新的完整序列。
- 方向审计提升至选点 **20/46（43.5%）**、执行边≤30°且正向 **19/46（41.3%）**；相比
  Round005 的 29.0%/25.8% 显著恢复，同时目标尝试从 93 降至 46，证明在线 supported-partial
  走廊约束有效抑制反向扫描，应保留。
- 新瓶颈是正确走廊内的可执行点与完成边界：EP6 的 3/3 点和边全部方向正确，却在 BETWEEN
  completion 保持 unknown 后由 far-lane anchor 触发 no-safe-point；EP3/18/204 在走廊内没有
  strict-ground ray，EP45/126 被 final RGB-ray history gate 拒绝，EP27 回溯失败。
- 下一轮应在同一走廊内由“最远像素”改为“最远但可达的已验证锚点”，修复双物体 token 提取和
  ENTER/STOP 边界，并为走廊内地面缺口提供局部重采样；不得恢复全景切线扫描。

## Active-STOP Round 007 — 关系恢复与局部走廊补视角（2026-09-07）

- 固定十 EP 全部完成，197 项测试通过；模拟器 active-STOP 为 **2/10（EP0、EP6）**，但独立
  逐段复核和 GT 完成边门相交后仍为 **0/10**。EP6 的同边 STOP 将“bar corner”误判为完成，
  所以该新增模拟器命中明确无效。
- 方向后验为选点 **15/43（34.9%）**、执行边 **15/43（34.9%）**，较 Round006 退化。
  BETWEEN/CIRCUMNAVIGATE 的通用关系恢复减少了一些漏判，但 supported-partial 的 ±75° 走廊
  会让后续选点漂移，且宽松 STOP 面积增长规则不能证明 corner/end 等精确关系。

## Active-STOP Round 008 — 路线精度与精确 STOP 保护（2026-09-07）

- 固定十 EP、201 项测试、视频和协议产物齐全；模拟器 active-STOP **1/10**，严格成功
  **0/10**。EP6 的 bar-corner 假 STOP 被消除，即使进入目标半径也因关系未证实而不发 STOP。
- 方向提升为选点 **17/39（43.6%）**、执行边 **15/39（38.5%）**。±30° 支持走廊抑制了
  EP3 的选点漂移，但 6 条 EP 因走廊内无严格地面射线终止，表明曲线路线与历史安全恢复尚未分离。
- 后续子句不再无条件覆盖当前物体方向；最终 history-safe 阶段增加同走廊局部 RGB 补采；纯转向
  先约束到可用 ±90° 视图。但纯转向结果随后仍被 post-selection 可达地面修复换到相邻视图。

## Active-STOP Round 009 — 关系共识（基础设施无效，不计分，2026-09-07）

- v23 下禁止 BETWEEN/CIRCUMNAVIGATE/精确 ENTER 的检测规则覆盖 VLM partial/ambiguous，并把
  支持走廊调到一个八视角扇区（±45°）。运行中 EP9 暴露局部补采把失败视图也加入候选，第三次
  补采超过 eight-plus-one 可视化 schema；该 EP 无轨迹结果，整轮作废，剩余 worker 主动停止。
- 修复为失败补采只保留诊断，只有首个通过严格地面/方向检查的视图才追加；新增三探针回归测试。

## Active-STOP Round 010 — 保守关系共识干净重跑（2026-09-07）

- 203 项测试后从十个数据集起点干净重跑，10/10 returncode=0、视频齐全，无补视角 schema 或
  provider 错误。模拟器 active-STOP **1/10（EP0）**，严格成功仍为 **0/10**。
- 方向继续提升到选点 **21/40（52.5%）**、执行边≤30°且正向 **17/40（42.5%）**；物理到达
  判定 39/40 一致，因此当前首要瓶颈不是执行器。
- EP6 BETWEEN、EP18 CIRCUMNAVIGATE 的 partial/ambiguous 不再假完成；大量 GT 正确方向的
  FOLLOW/BETWEEN/CIRCUMNAVIGATE/TURN/PASS 边仍被判 unknown，随后在一次受限续走/回溯后耗尽
  地面或历史安全候选。EP204/219 的首段选点仍在错误半球。
- EP0 的 nominal ±90° 视图锁定有效，但 post-selection 可达地面修复把 view2 换成 view3，
  左转边仍为 55.7°；下一轮须把这个锁传递到隐藏投影修复，并用多次独立共识稳定逐段复核，不能
  通过重新放宽确定性关系规则换取表面完成数。

## Active-STOP Round 011 — 因果路线与跨边证据（无效尝试，2026-09-07）

- `round_011_five_iteration_causal_route_carryover` 因失败的局部 RGB 探针也被追加到正式候选，导致
  eight-plus-one 可视化超过 9 视图，按实现缺陷作废；修复后只有最终被采用的探针才进入候选集。
- `round_011b_five_iteration_causal_route_carryover_rerun` 运行中 GPU0 被外部训练进程占用，三个
  worker OOM，按外部资源污染作废；此后本轮改为 GPU0 单 worker 串行。
- `round_011c_five_iteration_causal_route_carryover_serial` 的早期续跑混入 provider 代理失败，隔离后
  又暴露 TURN_TO 路线点上限在基础上限为空时的 `None` 比较错误，按 provider/实现失败作废。
- 上述目录及其 manifest 只保留为基础设施诊断，任何局部成功、几何命中或已生成视频均不进入
  正式成功率分母。

## Active-STOP Round 011d — 因果路线与跨边证据干净重跑（2026-09-07）

- 正式目录：`round_011d_five_iteration_causal_route_carryover_clean`。固定十 EP 从真实起点、同一
  冻结配置、GPU0 串行完成；10/10 子进程返回 0，轨迹、视频、独立逐段复核与事后 GT 审计齐全，
  206 项回归测试通过。在线模块未加载 reference path。
- 五项通用算法改动为：纯方向 TURN 的最终修复不再改变 nominal ±90° 视图；TURN_TO_LANDMARK
  使用下一子句作短距离路线建立后再在到达态对齐地标；首个 supported-partial 真实边的结构化证据
  传入唯一续走边；完成 prompt 联合评价前后两条边；直线恢复走廊无地面时一次扩至前方 180°。
- 模拟器 active-STOP **0/10**，最终严格成功 **0/10**；没有 EP 发出 STOP。EP6 虽进入 3m
  goal radius，但未完成全部子指令，按协议不计成功。在线完成前缀为 EP0=1/4、EP3=1/3、
  EP6=1/3、EP9=2/4、EP18=1/3、EP27=2/4、EP45=1/4、EP126=2/4、EP204=0/4、
  EP219=0/4；独立复核前缀更低，分别为 1、0、1、1、0、0、1、1、0、0 段。
- 后验方向审计：44 个点中选点≤30°为 **21/44（47.7%）**，真实执行边同时≤30°且正进度为
  **16/44（36.4%）**，均较 Round010 的 52.5%/42.5% 退化。44 次中大多数点执行能到点，
  瓶颈仍是语义方向与完成边界，不是点簇物理到达。
- 主要失败：正确的首边/次边在 FOLLOW、BETWEEN、PASS、CIRCUMNAVIGATE 上持续被判 UNKNOWN；
  后续恢复扩到 ±90° 后产生明显反向边；EP0 的纯转向 nominal 视图没有可达地面点，EP204 的
  TURN_TO 下一路线点 1.5m 上限过紧，EP219 首段被未来 stairs 方位覆盖当前 exit 判别。终止原因
  全部是 no-safe-ground/history/corridor exhaustion。协议不变量全部通过，但本轮算法配置不接纳。

## Active-STOP Round 012 — 图证据、路线坐标与独立核验修正（2026-09-07）

- 正式目录：`outputs/r2r_curriculum_10ep_active_stop_20260907/round_012_five_iteration_graph_evidence_and_route_frame_clean/`。
  固定十 EP、真实起点、同一冻结配置和 GPU0 三 worker 完成。
- 原始 simulator active-STOP 为 **1/10（EP219）**。独立核验器最初把当前节点侧/后方已走过的楼梯
  误认为前方尚未完成；加入 PREV/CURRENT FRONT/REAR 时间标签和垂直/水平位移后重新核验，EP219
  四段均通过。事后协议复审得到严格成功 **1/10**；其他随机或几何命中仍不计。

## Active-STOP Round 013 — 非线性路线重获取与终端地面（2026-09-08）

- 正式目录：`outputs/r2r_curriculum_10ep_active_stop_20260908/round_013_five_iteration_nonlinear_reacquire_and_terminal_ground/`。
  10/10 进程、视频和审计产物齐全，219 项测试通过。
- 原始 simulator active-STOP 为 **1/10（EP0）**，但独立逐段核验和 GT 方向门相交后严格成功
  **0/10**。选点≤30°为 **33/61**，执行边≤30°且正向为 **30/61**。
- 关键物理失败：EP45 在距冻结落地点 1.49m 时因全部点簇消失误报到达；EP126 在最终距落地点
  0.496m 时因步数耗尽漏报到达。两者都会让上层错误 block 正确方向，促成下一轮执行器修订。

## Active-STOP Round 014/014b — 端点校验执行器、多跳核验与终点姿态（2026-09-08）

- Round014 首次启动因 RGB 共识门引用未初始化的 `current_sector_set` 抛出异常，立即停止全部 worker，
  整轮作废且不统计。修复后 EP6 smoke 完整通过，再在全新目录
  `outputs/r2r_curriculum_10ep_active_stop_20260908/round_014b_five_iteration_endpoint_guard_multiedge_stop_pose_clean/`
  从十个真实起点干净重跑；10/10 returncode=0，视频齐全，224 项测试通过。
- 五项通用变更：点簇全失但距冻结选点>0.75m时继续同端点引导；步数边界内且停止点至少半失时补发
  到达；独立核验覆盖同一子指令的全部连续真实边；物体相对 STOP 使用 1.5m 近侧短跳；`in front`
  作为物体相对空间位置而非最终相机朝向，并仅允许完整 RGB 共识补偿 DINO 漏检。
- 点执行器在 67 个目标上为 **TP=62、FP=0、FN=0、TN=5，准确率 100%**；所有到达都建立节点，
  所有初始/导航/停止点簇保持在综合地面 mask 上。说明物理到达已经不再是本轮主要瓶颈。
- 原始 simulator active-STOP 为 **1/10（EP219）**，另一个 STOP（EP3）在 7.16m 外，明确不计。
  EP219 在线 4/4 按序完成、独立核验 4/4、完成边 4/4 满足≤30°且正向，最终 STOP 距目标
  2.297m，因此严格 instruction-validated aligned success 首次在本轮成立：**1/10**。
- 全轮选点≤30°为 **27/67（40.3%）**，执行边≤30°且正向为 **26/67（38.8%）**。其余在线前缀：
  EP0=3/4、EP3=3/3（但错误终点）、EP6=2/3、EP9=3/4、EP18=1/3、EP27=1/4、
  EP45=2/4、EP126=2/4、EP204=1/4。主要剩余瓶颈为正确边被判 UNKNOWN 后的一次恢复转成反向，
  以及 PASS/TURN/STOP 精确语义边界和局部严格地面候选耗尽。

## Active-STOP Round 015 — 多参考判定与局部门槛负向校准（2026-09-08）

- 正式目录：`outputs/r2r_curriculum_10ep_active_stop_20260908/round_015_five_iteration_multireference_local_portal_clean/`。
  固定十 EP、真实数据集起点、V31 选点、V24 完成判定、V28 执行器、GPU0 三 worker；10/10
  returncode=0，视频与轨迹齐全，独立逐段核验和最终协议审计均已执行，229 项测试通过。
- 本组五项通用迭代包含：协调 PASS 多地标分别核验；撤销会把 `walk towards` 错判完成的宽松恢复；
  拒绝 DINO `boxes` 全景假阳性的 V35 硬门；显式转向的真实来路参考系；带 `just inside/beyond`
  语义的局部门槛距离上限。前四项保留，最后一项的 2.0m 参数不接纳并进入下一组校准。
- 原始 simulator active-STOP **0/10**，主动 STOP **0/10**，严格成功 **0/10**；30 个目标中
  选点≤30°为 **15/30（50.0%）**，真实执行边同时≤30°且正进度为 **14/30（46.7%）**。
  独立核验仅确认 EP3 在线前两段、EP45 第一段；EP6 在线第二段未通过独立核验。
- 失败机制是通用且可复现的：2.0m cap 使 EP0、EP9、EP27、EP219 在首段已选语义视角中找不到
  同时满足最小进度与上限的地面锚点而零步退出；EP126 的 TURN_AROUND 借用下一门段语义时也被
  错误继承 2.0m cap。Round014b 对应可达锚点分别约 2.04m、5.71m、2.60m、2.60m、2.50m，
  证明固定 2.0m 不是可执行的通用门槛策略。本轮不计为改进，不覆盖 Round014b 权威基线。
- 下一组第 1 次迭代把词法门槛 cap 限于原本无 form cap 的 `TRAVERSE_PORTAL_REGION` 并校准为
  3.0m；EXIT/SELECT_PORTAL 继续使用既有 3.0m form cap，ENTER_REGION 不按当前位置截断远处入口。

## Active-STOP Round 016 — 成段路线完整性与地标中心复采（2026-09-08）

- 正式目录：`outputs/r2r_curriculum_10ep_active_stop_20260908/round_016_five_iteration_stage_integrity_landmark_center_clean/`。
  固定十 EP 从真实起点、V31/V24/V28、DINO+SAM 与综合地面、GPU0 三 worker；10/10 returncode=0，
  轨迹和视频齐全，232 项测试及 18 个子测试通过，独立逐段复核和隐藏 GT 后验审计已完成。
- 五项通用迭代为：把局部门槛 cap 校准到仅 TRAVERSE 3.0m；远侧 CIRCUMNAVIGATE 要求单边或
  跨边累计 4m 且路线弯折不超过 90°；转向门口的终态左右侧与真实转向动作解耦；全景检测别名下
  TURN_TO 必须以当前地标图像方位为主并对偏心地标实际补采 22.5° RGB；BETWEEN 浅边先经
  2.25m 累计路线完整性门控。所有决策仍不读取深度或 GT。
- 原始 simulator active-STOP **2/10（EP6、EP219）**，最终严格成功 **1/10（仅 EP219）**。
  EP219 在线 4/4、独立 4/4、完成边全部≤30°且正向，主动 STOP，终点测地距离 0.626m。
  EP6 虽 STOP 且 2.373m，但独立只确认 1/3，严格无效；EP9 的 STOP 距目标 7.458m，也无效。
- 57 个目标中选点≤30° **30/57（52.6%）**，真实执行边≤30°且正向 **29/57（50.9%）**，较
  Round015 的 50.0%/46.7% 小幅提升。独立前缀：EP0=1、EP3=2、EP6=1、EP9=1、EP18=0、
  EP27=1、EP45=1、EP126=2、EP204=0、EP219=4。
- 协议检查发现一个必须先修复的适配层越权：V24 已将 EP6 的 1.76m BETWEEN 浅边因路线完整性
  不足降为 UNKNOWN，但 `instruction_completion_judge` 的旧“下一地标新出现”恢复随后又把它提升
  为 COMPLETED。该段及其后续 STOP 不计有效；下一迭代必须让所有适配层 BETWEEN 恢复共同服从
  路线完整性门，且补测试防止任何后处理绕过底层否决。

## Active-STOP Round 017 — 路线消歧、当前节点地标对齐与裸转向审计（2026-09-08）

- 正式目录：`outputs/r2r_curriculum_10ep_active_stop_20260908/round_017_five_iteration_rgb_alignment_bare_turn_clean/`。
  固定十 EP 从真实起点运行，V31/V24/V28、DINO+SAM、综合地面、seed=17 等冻结协议未变；最终
  10/10 returncode=0，十条轨迹/视频、独立逐段复核和后验 GT 审计齐全。一次四 worker 试跑在
  首点前因 GPU0 OOM 作废并移出正式目录；正式数据最多三 worker，不混入基础设施失败。
- 通用迭代包括：BETWEEN 适配层不能绕过路线完整性否决；全景且词法多义的 TURN_TO 用“当前
  地标候选可见 + 完整有序路线”消歧，不把 detector query alias 当成语义归一化；真实到点后用
  当前节点八张纯 RGB 居中同一地标；该专用 RGB 对齐在真实到点、精确转向且非倒退时可解决广域
  detector 假阳性；裸左右转的可达锚点在完整同侧三视图内修复且局部上限校准为 4m；把选择器已
  实际执行/视频可见的预转写入 action history，避免审计遗漏真实转向。
- 定向回归均满足严格链条：EP204 首段一次完成，选点误差 5.95°、执行边 5.63°、GT 正向推进
  2.84m，并获独立语义复核；EP0 前两段均在线和独立通过，第二段裸左转选点/执行误差
  0.95°/0.98°。197 项测试及 18 个子测试通过。
- 正式全轮原始 simulator active-STOP **1/10（EP219）**，但 EP219 后两条在线完成边分别超过
  30°（stage2 约 44.5°、stage3 约 33.7°），因此最终严格成功仍为 **0/10**；随机/几何到达没有
  计入。60 个目标中选点≤30° **32/60（53.3%）**，执行边≤30°且正向 **29/60（48.3%）**。
- 独立严格前缀为 EP0=2、EP3=2、EP6=1、EP9=1、EP18=0、EP27=0（stage1 单段虽通过但前段
  未通过）、EP45=1、EP126=2、EP204=1、EP219=2。主要共性问题已转移为：新 `straight/pass`
  子指令丢失上一条真实路线方位而选到反向/侧向；正确 UNKNOWN 后可达地面候选耗尽；长
  CIRCUM/BETWEEN 的跨边完成证据仍不足；部分 compound TURN 在线过早完成。下一组先让带明确
  straight/forward 词的线性阶段继承上一条真实语义路线，再分别复测 EP0/EP9。

## Active-STOP Round 018 — 线性短边、区域通过与环绕连续证据（2026-09-08）

- 正式目录：`outputs/r2r_curriculum_10ep_active_stop_20260908/round_018_five_iteration_linear_pass_between_circ_clean/`。
  固定十 EP 从真实起点在 GPU0 三 worker 干净运行，10/10 returncode=0；轨迹、视频、独立逐段
  RGB/action 核验和隐藏 GT 后验审计齐全，243 项测试通过。在线模块未读取 reference path，选点
  VLM 仍为 RGB-only。
- 五项通用迭代为：显式 straight/pass 的已承诺路线使用 2.5m 短边且保留 metadata；PASS 房间按
  拓扑上下文变化判定；BETWEEN 的“到间隙”与“穿过间隙”使用不同完整性距离；CIRCUMNAVIGATE
  的 UNKNOWN 只在完整性不足且运动一致时保留一次同障碍路线续走；无指定侧的环绕用 67.5° 首边
  和 22.5° 续边补视角。独立核验器同时修复多边阶段遗漏中间节点全景的问题，并按起点、每边
  keyframe、中间到达节点、终点的严格时间顺序审计。
- 原始 simulator active-STOP 成功 **3/10（EP0、EP6、EP219）**；EP3、EP126 也主动 STOP，但
  分别距目标 3.673m、5.287m，明确不计。独立逐段核验与所有完成边方向/进度门相交后，严格
  instruction-validated aligned success 仍为 **0/10**。
- 52 个点中选点方向误差≤30°为 **30/52（57.7%）**，真实执行边同时≤30°且正向推进为
  **28/52（53.8%）**，较 Round017 的 53.3%/48.3% 提升。点执行器 52 次尝试中 50 次到达，
  隐藏到达标签与执行器信号仍为 TP=50、FP=0、FN=0、TN=2；执行器不是主要瓶颈。
- 独立通过段：EP0=0,1,2；EP3=0,1,2；EP6=0,1；EP9=0；EP18=0；EP27 仅 stage1（因此有序
  前缀为0）；EP45=0；EP126=0；EP204=0；EP219=0,1。EP0 最终 STOP_WAIT 侧偏 63.7°，
  EP219 后两段约 40°/37°，EP126 后两段反向；EP9/18/27/45 的正确 UNKNOWN 后续走转成反向，
  EP204 在首段后无安全地面点。下一组优先让已到达节点直接审计相邻 STOP_WAIT，并让
  ENTER/TURN 的 UNKNOWN 沿已证实走廊续走，禁止通过无证据放宽 completion 获得假成功。

## Active-STOP Round 019 — 同节点 STOP、长边连续路线与像素射线走廊（2026-09-08）

- 正式目录：`outputs/r2r_curriculum_10ep_active_stop_20260908/round_019_five_iteration_chained_stop_extent_corridor_qualified_enter_clean_v2/`。
  固定十 EP、真实数据集起点、V31/V24/V28、DINO+SAM 与综合地面、GPU0 三 worker；修复一次
  `stage_progress: null` 协议回归后在全新目录 10/10 干净重跑。十条视频、逐段独立核验和隐藏 GT
  审计齐全，249 项回归测试通过。此前混入空值异常的同名非 v2 目录只作诊断，不计结果。
- 五次通用尝试为：普通 near/beside STOP_WAIT 可在前一段到达节点以视觉/动作证据链式审计；显式
  `along/follow ... to the end` 使用跨两条真实边的累计行程和侧/后方 RGB 终端证据；该类复合长段
  UNKNOWN 后继承一次 60° 已承诺路线走廊；post-selection 地面可达性修复按实际像素射线而非相机
  中心服从绝对走廊；带限定物的 ENTER_REGION 尝试用至少双视角 RGB 共识补偿检测漏检。最后一项
  在 EP27 的 65.7° 错误边上仍产生过早完成，独立核验拒绝，判定为失败尝试并在下一轮前撤销。
- 模拟器 active-STOP **3/10（EP0、EP9、EP219）**；严格成功 **1/10（仅 EP0）**。EP0 在线
  4/4、独立 4/4，三条移动边选点/执行误差分别约 9.3/10.3°、1.0/1.0°、5.2/5.6°，均正向，
  最终主动 STOP 距目标 0.564m。EP9、EP219 的后段方向和独立语义门未通过，几何成功无效。
- 48 个目标中选点≤30° **31/48（64.6%）**，实际执行边同时≤30°且正向 **28/48（58.3%）**；
  比 Round018 的 57.7%/53.8% 提升。所有 42 次物理到达信号仍与隐藏 0.75m 标签一致，无假到达或
  漏到达，且到达节点、双点簇地面约束和视频协议全部通过，执行器仍不是主要瓶颈。
- 在线完成前缀为 EP0=4/4、EP3=3/3、EP6=2/3、EP9=4/4、EP18=1/3、EP27=4/4、EP45=3/4、
  EP126=2/4、EP204=1/4、EP219=4/4；独立核验实际只通过 EP0 全部，EP3/6/9/219 前两段，
  EP18/45/126/204 第一段，EP27 仅第二段。主要失败是 APPROACH/TURN/ENTER 的 UNKNOWN 恢复
  选到反向支路，以及复合长段/最终 STOP 在严格地面走廊内找不到可达点。下一轮先撤销不安全的
  限定房间 RGB 覆盖，再让恢复候选服从已验证节点的来路坐标与实际像素射线，不能因在线完成数增加
  而放松独立证据或隐藏 GT 验收门。

## Active-STOP Round 020 — 门槛事件、复合转向续走与局部 STOP（2026-09-08）

- 正式目录：`outputs/r2r_curriculum_10ep_active_stop_20260908/round_020_five_iteration_portal_boundary_turn_continuation_stop_clean/`。
  固定十 EP 从数据集真实起点串行运行；V31/V24/V28、DINO+SAM、综合地面、seed=17 等冻结配置
  不变。GPU0 同时有外部训练长期占用约 33GB，本系统单进程约 11GB，因此本轮不用多 worker，
  10/10 returncode=0；十条视频、逐段独立 RGB/action 核验和隐藏 GT 审计齐全，249 项测试和
  18 个子测试通过。
- 五项保留的通用迭代是：显式局部 `TRAVERSE_PORTAL_REGION` 的可执行上限由 3m 校准为 4m；
  TURN_TO_LANDMARK 的独立 verifier 先读烧录的 CURRENT FRONT 判断最终朝向；复合左右转的门槛
  恢复必须有 `boundary_event=observed/endpoint_inferred`，不能以终点两侧同时看见门框覆盖模型的
  partial；首条真实转向边若为有效 partial，下一跳沿其冻结 RGB 选点射线优先续走，不重复执行
  `turn`，主走廊没有严格地面时仅允许一次过滤来向/屏蔽方向后的相邻 RGB 弯折；局部物体 STOP
  的可执行上限由 1.5m 保守放宽到 2.0m，仍要求综合地面与 navmesh 可达。
- 两项失败探针没有保留：EP6 的同节点 `bar corner` 链式 STOP 虽进入目标半径，但独立核验确认
  仍是沿吧台而非明确拐角；把所有来路走廊从 20° 放到 22.5° 对 EP204 无改善。限定 ENTER 的
  双视角检测覆盖继续保持 Round019 已撤销状态，不混入本轮正式代码。
- 模拟器 active-STOP 为 **4/10（EP0、EP45、EP204、EP219）**；严格 instruction-validated、
  所有完成边≤30°且正向的成功为 **2/10（EP0、EP45）**。EP45 是本轮新增严格成功：在线与
  独立均为 4/4，完成边全部通过方向/进度门，主动 STOP 距目标 0.503m。EP204 的四条完成边均
  几何合格且 STOP 距目标 0.728m，但最终 FRONT 是关门、床在 FRONT_LEFT，未满足“面向大床”；
  EP219 后两条完成边被隐藏 GT 几何门拒绝，二者都不计严格成功。
- 59 个目标中选点≤30°为 **35/59（59.3%）**，实际执行边同时≤30°且正向为 **31/59
  （52.5%）**。执行器隐藏 0.75m 标签为 TP=51、FP=3、FN=0、TN=5，准确率 94.9%、precision
  94.4%、recall 100%；三个 FP 分别是 EP3 的 0 步近场捷径（0.853m）以及 EP9/EP126 的停止
  点簇消失但冻结端点仍为 1.442m/0.882m。下一轮首先让冻结端点 guard 覆盖所有点簇消失型到达，
  包括导航点仍可见的情况，并禁止超出 0.75m 的零步近场到达。
- 在线完成数：EP0=4/4、EP3=2/3、EP6=2/3、EP9=2/4、EP18=1/3、EP27=2/4、EP45=4/4、
  EP126=2/4、EP204=4/4、EP219=4/4。独立语义通过段分别为 EP0=0,1,2,3；EP3=0,1；EP6=0；
  EP9=0,1；EP18 无；EP27 仅 1；EP45=0,1,2,3；EP126=0；EP204=0,1,2；EP219=0,1。与
  Round019 相比，ENTER/复合 TURN 假完成显著减少，但主要瓶颈转为正确 UNKNOWN 后的严格地面
  候选耗尽、恢复边漂离已承诺路线，以及最终面向目标的姿态没有在普通非 TURN_TO 终段统一校正。

## Active-STOP Round 021 — 统一到达端点、终态朝向与方向承诺（2026-09-08）

- 正式目录：`outputs/r2r_curriculum_10ep_active_stop_20260908/round_021_five_iteration_arrival_facing_between_turn_route_commitment/`。
  固定十 EP 从真实数据集起点串行运行，冻结配置仍为 V31/V24/V28、DeepSeek、DINO+SAM 综合地面、
  seed=17、GPU0。10/10 returncode=0，十条完整视频、导航图、独立逐段 RGB/action 核验、隐藏 GT
  后验审计与协议报告齐全；255 项测试和 18 个子测试通过。
- 本轮保留的通用改进为：冻结端点距离 guard 覆盖全部图像到达 proposal（包括导航点仍可见和零步
  near-field）；非 TURN 阶段只有原始指令明确要求 `face/facing/look` 时才在真实到点后执行当前节点
  八视图朝向校正；`reach/walk between A and B` 完成需当前 RGB 检测呈两参照物左右夹持；纯 180°
  转向路点上限由 3m 校准为 2.5m，防止吞掉紧随其后的门槛段；复合左右转保存八视图三方向硬门控
  的真实预转证据，并对主候选与最强相邻同侧竞争视图采集中点 RGB；`turn ... and walk towards X`
  以正确转向、真实持续移动及下一段地标同/邻视角作为路线承诺边界，而不要求已抵达 X。
- 失败探针未保留：U-turn 2m 会让水平相机没有可见地面锚点；把 portal 135°语义视图强制归一到
  75°使 EP126 第二段误差升到约74°，已撤销；复合浅左转采前/左 22.5°中点仍吸附到原走廊，改为
  VLM 主候选与最强相邻同侧候选的67.5°中点。独立审计同步统一语义定义，但隐藏 GT ≤30°和正向
  进展门从未放宽。
- 模拟器 active-STOP **5/10（EP0、EP45、EP126、EP204、EP219）**；五条全部通过在线有序完成、
  独立逐段语义核验和所有完成边 GT 对齐，因此严格 success 同为 **5/10**。EP126 是新增完整成功：
  在线/独立4/4，三条移动边选点误差3.25°/10.40°/16.45°、执行误差2.89°/9.80°/17.65°，均正向，
  主动 STOP 距目标1.859m。EP204 经当前节点朝向校正后独立4/4、STOP 0.728m；EP219 本轮后两段
  也通过隐藏几何和独立核验，STOP 2.895m。
- 57 个目标中选点≤30° **38/57（66.7%）**，真实执行边同时≤30°且正向 **36/57（63.2%）**，
  较 Round020 的59.3%/52.5%明显提升。物理到达混淆矩阵为 TP=55、FP=0、FN=0、TN=2，准确率、
  precision、recall 均100%；Round020 的三类假到达已消除，点导航/执行器目前不是主要瓶颈。
- 在线完成数为 EP0=4/4、EP3=2/3、EP6=2/3、EP9=3/4、EP18=1/3、EP27=2/4、EP45=4/4、
  EP126=4/4、EP204=4/4、EP219=4/4。独立通过段分别为 EP0全部；EP3=0,1；EP6=0,1；
  EP9=0,1；EP18=0；EP27仅1；其余四个成功 EP 全部。失败五条中 EP3/6/9/18 都最终耗尽严格地面
  候选，EP27 在长错误探索后耗尽已 block 方向；其共同瓶颈已从物理到达转为“正确节点之后的
  语义终点地面候选缺失”和“UNKNOWN 后恢复方向漂离已验证走廊”。最终10/10 gate仍 FAIL，下一轮
  优先对这五条的第一个失效段做跨 EP 通用候选/恢复分析。

## Rk-B 候选（未编号，尚未十 EP 回归）— 前进停滞早停 + 每目标步数预算 40（2026-09-11）

- 触发问题：`outputs/e2e_eval/20260911_001436_e2e_opennav/` 中 34 个 `end_reason=max_steps` 的 hop
  同时包含两类相反情形——约 15 个是选点方向正确、自由行走 3.5–4.4 m 后被 20 步预算截断；约 13 个
  是起步即撞墙（≥6 次前进隐藏位移 <2 cm，整段只走 0–0.8 m）却把 20 步全部耗尽；另约 6 个是贴墙
  斜擦微滑（每步 2–10 cm）。单纯加大预算会放大第二类浪费，单纯缩小预算会加重第一类误判。
- 离线标定（evaluation-only，25 条带隐藏几何 EP、192 hop、2004 次 `move_forward`，脚本
  `scripts/analyze_forward_stall_calibration.py`）：已有的逐帧灰度差 `rgb_motion_score` 在隐藏位移
  <2 cm 时中位数为 0（确定性渲染，帧完全相同）、微滑 6–14 mm 时稳定在 3–8；自由前进（≥20 cm）
  5 分位 12.3、中位 24.1。规则「连续 K 次前进 score < T（转向不重置）」在 T=8/K=3 触发 16 次、
  0 次误停（误停定义：该 hop 后来正常到达且触发点之后仍走 >0.3 m），可省 153 步；T=10/K=3 触发
  18 次但有 1 次误停；T=5/K=2 0 误停但漏掉 score≈8 的微滑段。到达 hop 的步数中位 10、90 分位
  15.3、最大 18，加大预算不改变正常到达。
- 改动（仅走路层，RGB-only 合规：只读相邻 RGB 差分与自己的 action history，`project_rulle.md` §0
  明确允许「根据相邻 RGB 的变化计算纯视觉运动量」，不读 collision）：
  `TRACKING_CLUSTER_PROFILES["rgb_only_dense_stop_v1"]` 新增 `stall_motion_threshold=8.0`、
  `stall_forward_frames=3`（0 = 关闭）；`_execute_rgb_only` 维护 `stall_forward_streak`，达到阈值即以
  非到达 `end_reason="rgb_forward_stall"` 结束 hop（记录 `terminal_stall_forward_streak`、
  `terminal_stall_motion_scores`，每步 action_history 带 `stall_forward_streak`），外层策略沿用既有
  「回溯到上个已验证节点 + block 该方向 + 重新选点」分支，执行器内不做转向脱困。
  `evaluate_point_navigation.py` 与 `habitat_point_navigation.py` 的 `--max-steps-per-target` 默认值
  统一为 40（原 20 / 32 不一致）。
- 已实际运行并通过：`tests/test_rgb_only_executor_stall.py`（6 例：帧不变 K 步即停、帧有变化跑满
  预算得 `max_steps`、先走后停、K=0 关闭、转向不重置）、`tests/test_rgb_only_instruction_sequence.py`
  新增停滞 hop 走回溯/拉黑分支用例；全量 363 个单元/契约测试通过；EP0 heuristic 后端单 hop GPU 冒烟
  正常到达（12 步，streak 全 0，`config.max_steps_per_target=40`）。
- 尚未验证：固定十 EP 同轮回归（`bash run_e2e_eval.sh 0,3,6,9,18,27,45,126,204,219 --workers 2`），
  按 §16 在此之前本候选不得视为冻结配置。已知不覆盖：贴墙斜擦微滑与自由前进的 score 重叠，本规则
  捕获不到（停簇像素位移同样无法区分：微滑中位 4 px vs 自由 6 px），留作后续时序规则。

## 端到端评测记录（非十 EP 轮次）— OpenNav100 对齐起始朝向 + action-reversal 全量（2026-09-11/12）

- 轮次目录 `outputs/e2e_eval/20260911_235744_opennav100_aligned_actrev/`，commit `400fbfd`，数据集
  `data/datasets/opennav100_start_aligned/val_unseen_opennav100ids_start_aligned.json.gz`（sha256 `cb20c192…`），
  `--workers 2 --backtrack-method action-reversal`，85 分钟，云端 VLM 1387 次 / 约 842 万 token。
- 结果：100/100 跑完、0 崩溃；`simulator_reported_success` 5/100（id 11、166、187、721、1117）；主动 STOP 14 次
  （9 次在 3 m 圈外）；结束时在圈内未 STOP 14 条；结束方式 `no_floor_bearing_candidate` 83、序列完成 14、
  回溯被 VLM 拒绝 3。与 `20260911_001436` 轮不可直接比成功数（数据集与四处改动不同），可比「走到多深」：
  0 句完成 21 条（前 42）、结束时圈内 19（前 7）、平均沿路净进展 1.60 m（前 0.35 m）。
- 事后审计：独立复核 `verify_round_stage_completions.py --include-unknown --always-call-vlm`（519 次调用）与
  `analyze_judge_round.py` 均已实际运行；300 次不确定中 235 次确认、65 次推翻，167 次完成中 14 次推翻。
- 失败归因（互斥五组，95 条）：转弯句方向门清空 30、最后一句 STOP_WAIT 停不下来 17（9 条曾在终点圈内）、
  中途句候选耗尽 36、停错地方 9、回溯被拒 3。详见
  `docs/e2e_eval_reports/20260911_235744_opennav100_aligned_actrev/failure_analysis_95_plain_zh.md`，
  含按优先级 / 有效性排序的改进措施（M1–M8，全部尚未实施、尚未验证）。
- 本轮不构成任何候选的冻结依据；上一批改动（`272c223`、`2a3dbd9`、停滞早停、步数预算 40、action-reversal）
  仍未做固定十 EP 同轮回归。
