# 端到端评测报告：OpenNav100 全量 100 条，双 worker 并行（2026-09-11）

- 轮次目录：`outputs/e2e_eval/20260911_001436_e2e_opennav/`（tmux 会话 `e2e_opennav100_0911`，终端全文另存 `outputs/e2e_eval/tmux_e2e_opennav100_0911.log`）
- 代码版本：`8fe1404`（工作区干净，与上一轮 OpenNav10 报告相比只多了「按 episode_id 命名目录/视频」一个提交，**没有修上一轮报告第 5 节第 1 条要求先修的未捕获异常**）
- 配置：与上一轮完全一致——`--vlm-backend deepseek`（DMXAPI `deepseek-v4-flash-vision-exp`，thinking 关闭），其余为 `evaluate_point_navigation.py` 默认值：GNM 策略、dense-majority 地面分割、DINO+SAM 检测、8 视图、每跳最多 20 步、每 episode 最多 30 跳、同一节点最多封锁 5 个方向、选点 prompt `v10_approach_relation_router`、判定 prompt `v13_structured_node_edge_binary`、seed 17、`cuda:0`
- 数据：官方 R2R VLN-CE v1-3 `val_unseen.json.gz`（sha256 前缀 `1767a407e2c8`），OpenNav100 的 100 个 episode id，起点位姿取官方值（不使用 OpenNav 文件的错误 `start_rotation`）
- 口径：本报告只使用运行时自带的 `simulator_reported_success`（在线判定最后子指令 completed 后主动 STOP，且 STOP 点在 goal radius 3 m 内）。`verify_round_stage_completions.py` **未运行**（需额外 VLM 费用，留给人工审核决定），因此 `instruction_validated_r2r_success` 与「独立核验的子指令完成数」按 fail-closed 记 0，不代表真实值。`audit_active_stop_round.py` 对本轮**不适用**（见第 2 节）。
- 「已实际运行并通过 / 仅静态检查 / 尚未验证」的区分：第 1–5 节全部数字来自本轮实际产物（`summary.json`、各 episode 的 `trajectory.json`、`navigation_graph.json`、`vlm_calls.json`、`process.log`、`evaluation_only/evaluation_geometry.json`）；第 6 节的根因分析基于产物证据加源码静态阅读，**尚未用修复后的代码验证**。

## 1. 总览

| 指标 | 值 |
|---|---|
| 运行 episode 数 | 100 / 100（无 not_run、无 unscored；两个分片返回码均为 0） |
| 进程崩溃（记为失败） | **75**（shard_0 38 条，shard_1 37 条） |
| 正常结束 | 25 |
| `simulator_reported_success` | **6 / 100**（id 11、308、531、670、698、810） |
| 主动 STOP 次数 | 8（6 次落在 goal radius 内；id 469、546 两次 STOP 位置错误） |
| 终止原因 | `episode_process_failed` 75；`all_candidate_directions_blocked_at_verified_node` 17；`instruction_sequence_complete` 8 |
| 总控制步数 | 正常结束 25 条 2221 步；崩溃 75 条在崩溃前共走 3490 步（只计入图中的边） |
| 总 VLM 调用 | 1257 次（分解 100、选点 641、选扇区 18、判定 498），全部 HTTP 200，0 次重试；平均 2.01 s，中位 1.89 s，p90 2.75 s，最长 12.3 s |
| 总 token | 约 792 万 |
| 耗时 | 00:14:37 启动，shard_1 01:33:21 结束，shard_0 01:35:41 结束，**共 81 分钟**；单条 21 s–3 min 08 s，中位 90 s |
| 显存 | 两个 worker 稳定共占 22.5 GB / 24.5 GB，无 OOM |
| 磁盘 | 1.6 GB |

一句话结论：**这一轮的主要产出不是成功率，而是把上一轮在 10 条上看到的崩溃模式在 100 条上定量确认了**——75% 的 episode 死于同一个未捕获异常，其中 43 条在崩溃前已经完成了至少一条子指令却没有留下 `trajectory.json`。在修复它之前，6/100 这个数字不能当作系统的真实成功率来引用。

## 2. 基础设施结论

- **启动器与双 worker 并行稳定**：两个分片各 50 条全部执行完毕，返回码 0；合并后的 `summary.json`/`manifest.json`/`process.log` 齐全。每 worker 约 11 GB 显存，24 GB 单卡上 2 个 worker 是上限。
- **耗时低于预估**：上一轮预估 1.5–2 小时，实际 81 分钟。原因是崩溃的 episode 平均只跑 ~1 分钟；如果 75 条都正常跑满恢复流程，预计 2–2.5 小时。
- **DMXAPI 中转站全程可用**：1257 次调用 0 次非 200、0 次重试，平均 2 s，比上一轮（2.69 s）还快；`thinking: disabled` 有效。
- **`audit_active_stop_round.py` 不能对本轮目录运行**：它只认固定十 EP 的 episode 索引（`FIXED_EPISODES`），在本轮目录上找到 0 条 episode，并且会**直接覆盖轮次根目录的 `summary.json`**（`audit_active_stop_round.py:556`）、向 `process.log` 追加一行、新建 `failure_cases.json/.md`。本次分析时误跑了一次，已用 `run_end_to_end_eval.merge_shards()` 从两个完好的 `shard_N/summary.json` 重新生成了一份逐字节等价的 `summary.json`，并删除了审计脚本生成的 4 个文件、去掉了 `process.log` 里多出的那一行（与 tmux 日志 diff 一致）。**启动器结束时打印的「post-run audits (manual)」提示里不应再列这个脚本**，或者该脚本改为写独立文件名。

## 3. 正常结束的 25 条

| id | 场景 | 子指令 | 步数 | 跳数（到达） | 系统判完成 | 封锁/回溯 | 终止原因 | STOP | 成功 | 起点→终点 geodesic (m) |
|---:|---|---:|---:|---|---:|---|---|:-:|:-:|---|
| 11 | 2azQ1b | 2 | 36 | 3 (3) | 2 | 0 / 0 | sequence_complete | ✓ | **✓** | 7.11 → 0.81 |
| 13 | zsNo4H | 4 | 79 | 8 (8) | 2 | 5 / 5 | directions_blocked | ✗ | ✗ | 12.04 → 5.43 |
| 94 | QUCTc6 | 4 | 69 | 6 (6) | 0 | 5 / 5 | directions_blocked | ✗ | ✗ | 13.93 → 11.68 |
| 140 | 2azQ1b | 3 | 137 | 9 (6) | 1 | 6 / 3 | directions_blocked | ✗ | ✗ | 13.37 → 21.05 |
| 166 | 2azQ1b | 2 | 87 | 6 (4) | 0 | 5 / 3 | directions_blocked | ✗ | ✗ | 6.01 → 5.67 |
| 171 | TbHJru | 6 | 66 | 6 (5) | 0 | 5 / 4 | directions_blocked | ✗ | ✗ | 11.21 → 7.80 |
| 308 | TbHJru | 5 | 135 | 10 (8) | 5 | 3 / 1 | sequence_complete | ✓ | **✓** | 8.07 → 1.39 |
| 469 | Z6MFQC | 4 | 63 | 6 (6) | 4 | 1 / 1 | sequence_complete | ✓ | ✗ | 16.25 → 16.33 |
| 526 | EU6Fwq | 2 | 56 | 6 (5) | 0 | 5 / 4 | directions_blocked | ✗ | ✗ | 5.68 → 6.36 |
| 531 | TbHJru | 4 | 50 | 7 (7) | 4 | 1 / 1 | sequence_complete | ✓ | **✓** | 5.27 → 2.31 |
| 546 | zsNo4H | 3 | 164 | 11 (6) | 3 | 6 / 1 | sequence_complete | ✓ | ✗ | 4.61 → 5.09 |
| 568 | QUCTc6 | 3 | 48 | 6 (6) | 0 | 5 / 5 | directions_blocked | ✗ | ✗ | 10.37 → 10.38 |
| 576 | Z6MFQC | 5 | 129 | 7 (3) | 0 | 6 / 2 | directions_blocked | ✗ | ✗ | 10.26 → 14.21 |
| 609 | QUCTc6 | 3 | 106 | 9 (8) | 1 | 6 / 5 | directions_blocked | ✗ | ✗ | 9.10 → 10.22 |
| 620 | EU6Fwq | 3 | 71 | 7 (6) | 0 | 6 / 5 | directions_blocked | ✗ | ✗ | 5.84 → 4.38 |
| 670 | zsNo4H | 4 | 118 | 12 (11) | 4 | 6 / 5 | sequence_complete | ✓ | **✓** | 8.82 → 0.12 |
| 698 | TbHJru | 5 | 82 | 10 (10) | 5 | 3 / 3 | sequence_complete | ✓ | **✓** | 6.12 → 1.91 |
| 781 | EU6Fwq | 1 | 59 | 6 (6) | 0 | 5 / 5 | directions_blocked | ✗ | ✗ | 5.61 → 4.04 |
| 804 | 2azQ1b | 6 | 109 | 9 (7) | 1 | 6 / 4 | directions_blocked | ✗ | ✗ | 12.82 → 14.05 |
| 810 | 2azQ1b | 3 | 91 | 8 (5) | 3 | 3 / 0 | sequence_complete | ✓ | **✓** | 7.01 → 1.19 |
| 824 | Z6MFQC | 5 | 87 | 7 (6) | 1 | 5 / 4 | directions_blocked | ✗ | ✗ | 6.54 → 2.42 |
| 842 | QUCTc6 | 6 | 62 | 6 (6) | 0 | 5 / 5 | directions_blocked | ✗ | ✗ | 11.80 → 17.27 |
| 1071 | zsNo4H | 6 | 183 | 15 (10) | 2 | 10 / 5 | directions_blocked | ✗ | ✗ | 13.37 → 12.22 |
| 1117 | x8F5xy | 1 | 68 | 6 (5) | 0 | 5 / 4 | directions_blocked | ✗ | ✗ | 8.57 → 5.50 |
| 1139 | QUCTc6 | 4 | 66 | 6 (5) | 0 | 5 / 4 | directions_blocked | ✗ | ✗ | 7.59 → 10.43 |

「封锁/回溯」= 同一已验证节点上累计封锁的方向数 / 序列恢复回溯次数；25 条里 118 次回溯物理上全部成功回到节点（`sequence_recovery_success_rate=1.0`）。所有 25 条的 `rgb_only_contract_audit_passed=true`。

### 3.1 六条成功

- **id 11**（"Walk across the floor and wait at the archway"，2 段）：第 1 跳判 unknown（"还没到对面"）、继续探索一跳后判 completed，第 3 跳 STOP，终点 0.81 m。与上一轮同一条 episode 的结果（2 跳、0.73 m）略有差异，说明 seed 固定但云端 VLM 输出不完全确定。
- **id 670**（4 段，12 跳、5 次回溯、6 个方向封锁）：终点距目标 **0.12 m**，是本轮最精确的一条，也是「unknown → 回溯 → 换方向」恢复链路在长流程里成功的例子。
- **id 308 / 698**（各 5 段，10 跳）：5 条子指令全部在线判 completed 后 STOP，终点 1.39 m / 1.91 m。
- **id 531**（4 段，7 跳）终点 2.31 m、**id 810**（3 段，8 跳）终点 1.19 m。
- 六条的共同点：子指令 2–5 段、起点距目标 5–9 m（中短程）、场景为 2azQ1b / TbHJru / zsNo4H 三个；没有一条起点距目标 >10 m 的 episode 成功。

### 3.2 两条「STOP 位置错误」

- **id 469**（"Turn right and walk along the red carpet to your right. Walk past the doorway wait by the glass info panes."）：6 跳全部到达，4 条子指令全部在线判 completed（置信 0.7–0.85），主动 STOP——但终点距目标 **16.33 m，比起点（16.25 m）还远**。也就是说判定层把 4 个阶段逐一「确认完成」，而实际上一步都没走对。这是判定层假阳性链的最清晰样本，值得作为 Rk-C 的首个回归 state。其中第 1、2 跳对 "Turn right" 都判 unknown，理由均是「action history 里有 12/17 次前进、**0 次右转命令**」——见第 6.2 节，这是一个系统性的信息缺失，不是模型幻觉。
- **id 546**（3 段，11 跳，其中 5 跳 `max_steps` 未到达）：终点 5.09 m，在半径外。前 5 跳都耗在第 1 段 "Turn right" 上（同样被判「没有右转命令」），最后两段的 completed 判定把 STOP 发在了错误房间。

### 3.3 十七条「方向全部封锁」

共同模式与上一轮相同：在某个已验证节点反复尝试新方向 → 到达 → 判 unknown → 封锁该方向 → 回溯；累计 5–6 个（id 1071 为 10 个）封锁后终止。17 条里 **11 条一条子指令都没完成**（系统判完成 = 0），其余 6 条完成 1–2 条。平均起点距目标 9.65 m，终点 9.59 m，**几乎零进展**；有 8 条终点比起点更远（最差 id 140：13.37 → 21.05 m）。

## 4. 崩溃的 75 条

### 4.1 异常分类（全部来自 `process.log` 最后一行）

| 异常 | 条数 | 抛出位置 |
|---|---:|---|
| `RuntimeError: No floor-bearing candidate can be sent to the VLM` | 50 | `vlm_harness.py:2677` |
| `RuntimeError: No floor-bearing candidate remains inside the explicit right direction gate` | 9 | `vlm_harness.py:2929` |
| `… explicit left direction gate` | 9 | 同上 |
| `… explicit rear direction gate` | 6 | 同上 |
| `RuntimeError: VLM harness exhausted retries for select_ground_target: ["'view_index'", …]` | 1（id 116） | `vlm_harness` 重试耗尽 |

75 条的调用栈完全一致：`habitat_point_navigation._run_habitat_episode` → `rgb_only_instruction_sequence.RGBOnlyInstructionSequenceStrategy.run`（`rgb_only_instruction_sequence.py:275` 直接调 `self.point_selector.select(...)`，**没有 try/except**）→ `point_selectors.select/choose_view` → `vlm_harness.select_ground_target` raise。不是 GPU、不是网络、不是 Habitat。

### 4.2 崩溃发生在什么阶段

- **崩溃时所在子指令序号**：第 0 条 31 个、第 1 条 26 个、第 2 条 15 个、第 3 条 3 个。即 **41% 的崩溃发生在第一条子指令还没完成的时候**。
- **崩溃前的进展**：44 / 75 条在崩溃前至少有一次 completed 判定（43 条至少完成一条子指令并开启了下一条），这些进度全部没有写进 `trajectory.json`，评分只能记 0。
- **6 条零步崩溃**（id 52、150、207、643、765、1087）：第一次选点就抛异常，一步都没走。其中 5 条的第一条子指令是 TURN_RIGHT / TURN_AROUND / TURN_LEFT（显式方向门里找不到地面），1 条（id 207 "Walk out of the room keeping the fireplace on your left"）是 EXIT_REGION。
- **「No floor-bearing candidate」（50 条）崩溃前的 unknown 连击**：3 次 12 条、4 次 20 条、5 次 17 条（另 2 条为 0 次）。也就是说，它们本来离状态机的「5 个方向封锁 → 正常终止」只差一两步，但选点器的硬排除（已封锁方向 + 来路方向 + 门控）先一步把候选清空了。这 50 条如果不崩，大概率会变成第 3.3 节那种 `directions_blocked` 的正常失败，**不会变成成功**。
- **「direction gate」（24 条）崩溃前的 unknown 连击**：0 次 7 条、1 次 13 条、2 次 3 条。门控崩溃来得更早：一个 TURN_* 子指令只要在对应扇区（左/右/后各 2–3 个 45° 视图）里封锁 1–2 个方向，就无候选可选。

### 4.3 崩溃时的子指令 form

| 崩溃类型 | 当时活跃的 form（条数） |
|---|---|
| direction gate（24） | TURN_RIGHT 9、TURN_LEFT 9、TURN_AROUND 6——**100% 是转向类** |
| no candidate（50） | STOP_WAIT 11、TURN_LEFT 9、EXIT_REGION 8、TRAVERSE_PORTAL_REGION 4、TURN_RIGHT 4、ADVANCE_STRAIGHT 3、BETWEEN_OBJECTS 2、FOLLOW_PATH_BOUNDARY 2、OTHER 2、其余各 1 |

两点值得注意：(a) 75 条里 **38 条崩在转向类子指令上**（TURN_LEFT 18、TURN_RIGHT 13、TURN_AROUND 6、TURN_TO_LANDMARK 1），与第 6.2 节的判定缺陷直接相关；(b) 11 条崩在 **最后一段 STOP_WAIT**，即已经走到最后阶段、判定反复 unknown、把可选方向耗尽——这些是「差一个 completed 就能 STOP」的近失案例（id 187、259、321、411、479、513、586、1106、1142、1301、1307）。

### 4.4 逐条清单

| id | 场景 | 子指令 | 崩溃时子指令序号 / form | 节点 | 步数 | 判定 completed/unknown | 崩溃类型 |
|---:|---|---:|---|---:|---:|---|---|
| 7 | x8F5xy | 3 | #1 BETWEEN_OBJECTS | 8 | 90 | 1 / 6 | no_candidate |
| 40 | zsNo4H | 7 | #2 ENTER_REGION | 7 | 73 | 2 / 4 | no_candidate |
| 42 | zsNo4H | 6 | #2 TURN_RIGHT | 4 | 32 | 2 / 1 | gate_right |
| 52 | zsNo4H | 3 | #0 TURN_RIGHT | 1 | 0 | 0 / 0 | gate_right |
| 70 | 2azQ1b | 6 | #0 TRAVERSE_PORTAL_REGION | 5 | 39 | 0 / 4 | no_candidate |
| 116 | 2azQ1b | 5 | #2 TURN_RIGHT | 8 | 67 | 2 / 5 | vlm_parse |
| 150 | QUCTc6 | 5 | #0 TURN_AROUND | 1 | 0 | 0 / 0 | gate_rear |
| 156 | zsNo4H | 7 | #2 TRAVERSE_PORTAL_REGION | 9 | 70 | 2 / 6 | no_candidate |
| 176 | QUCTc6 | 8 | #0 ADVANCE_STRAIGHT | 5 | 50 | 0 / 4 | no_candidate |
| 181 | TbHJru | 8 | #1 TURN_RIGHT | 8 | 46 | 1 / 6 | no_candidate |
| 187 | x8F5xy | 3 | #2 STOP_WAIT | 10 | 76 | 2 / 7 | no_candidate |
| 190 | QUCTc6 | 4 | #1 OTHER | 7 | 70 | 1 / 5 | no_candidate |
| 191 | QUCTc6 | 6 | #1 TURN_AROUND | 3 | 30 | 1 / 1 | gate_rear |
| 207 | Z6MFQC | 3 | #0 EXIT_REGION | 1 | 0 | 0 / 0 | no_candidate |
| 218 | X7HyMh | 5 | #1 TURN_LEFT | 7 | 45 | 1 / 5 | no_candidate |
| 226 | X7HyMh | 6 | #0 EXIT_REGION | 5 | 16 | 0 / 4 | no_candidate |
| 232 | oLBMNv | 6 | #1 VERTICAL_UP | 6 | 61 | 1 / 4 | no_candidate |
| 244 | X7HyMh | 4 | #1 TURN_LEFT | 3 | 25 | 1 / 1 | gate_left |
| 247 | zsNo4H | 6 | #0 ADVANCE_STRAIGHT | 5 | 59 | 0 / 4 | no_candidate |
| 259 | TbHJru | 2 | #1 STOP_WAIT | 9 | 46 | 1 / 7 | no_candidate |
| 265 | 2azQ1b | 5 | #3 TURN_LEFT | 9 | 104 | 3 / 5 | no_candidate |
| 275 | TbHJru | 4 | #2 TURN_LEFT | 6 | 47 | 2 / 3 | gate_left |
| 312 | x8F5xy | 3 | #0 EXIT_REGION | 5 | 30 | 0 / 4 | no_candidate |
| 321 | EU6Fwq | 3 | #2 STOP_WAIT | 10 | 84 | 2 / 7 | no_candidate |
| 330 | 2azQ1b | 3 | #0 VERTICAL_DOWN | 4 | 26 | 0 / 3 | no_candidate |
| 338 | TbHJru | 4 | #1 TURN_LEFT | 7 | 67 | 1 / 5 | no_candidate |
| 348 | EU6Fwq | 5 | #1 TURN_LEFT | 3 | 16 | 1 / 1 | gate_left |
| 362 | zsNo4H | 3 | #1 TURN_LEFT | 9 | 110 | 1 / 7 | gate_left |
| 371 | X7HyMh | 3 | #0 TRAVERSE_PORTAL_REGION | 5 | 29 | 0 / 4 | no_candidate |
| 377 | TbHJru | 4 | #1 TURN_RIGHT | 3 | 25 | 1 / 1 | gate_right |
| 387 | zsNo4H | 8 | #2 TURN_RIGHT | 9 | 73 | 2 / 6 | no_candidate |
| 403 | 2azQ1b | 3 | #0 TURN_RIGHT | 2 | 11 | 0 / 1 | gate_right |
| 411 | TbHJru | 2 | #1 STOP_WAIT | 7 | 54 | 1 / 5 | no_candidate |
| 423 | QUCTc6 | 5 | #1 CROSS_SPACE | 8 | 72 | 1 / 6 | no_candidate |
| 432 | Z6MFQC | 3 | #1 FOLLOW_PATH_BOUNDARY | 8 | 92 | 1 / 6 | no_candidate |
| 439 | zsNo4H | 7 | #0 TURN_AROUND | 2 | 3 | 0 / 1 | gate_rear |
| 447 | zsNo4H | 2 | #0 BETWEEN_OBJECTS | 6 | 43 | 0 / 5 | no_candidate |
| 454 | QUCTc6 | 6 | #0 TURN_LEFT | 6 | 49 | 0 / 5 | no_candidate |
| 461 | EU6Fwq | 6 | #0 TURN_AROUND | 2 | 14 | 0 / 1 | gate_rear |
| 479 | oLBMNv | 4 | #3 STOP_WAIT | 13 | 105 | 3 / 9 | no_candidate |
| 513 | 2azQ1b | 2 | #1 STOP_WAIT | 9 | 85 | 1 / 7 | no_candidate |
| 516 | zsNo4H | 6 | #2 TURN_LEFT | 3 | 24 | 2 / 0 | gate_left |
| 550 | Z6MFQC | 4 | #0 OTHER | 6 | 69 | 0 / 5 | no_candidate |
| 559 | Z6MFQC | 4 | #0 EXIT_REGION | 4 | 36 | 0 / 3 | no_candidate |
| 586 | Z6MFQC | 2 | #1 STOP_WAIT | 7 | 62 | 1 / 5 | no_candidate |
| 602 | X7HyMh | 5 | #0 EXIT_REGION | 5 | 32 | 0 / 4 | no_candidate |
| 643 | 2azQ1b | 3 | #0 TURN_LEFT | 1 | 0 | 0 / 0 | gate_left |
| 655 | zsNo4H | 4 | #0 EXIT_REGION | 4 | 34 | 0 / 3 | no_candidate |
| 677 | QUCTc6 | 4 | #1 TURN_LEFT | 7 | 62 | 1 / 5 | no_candidate |
| 705 | Z6MFQC | 1 | #0 FOLLOW_PATH_BOUNDARY | 5 | 57 | 0 / 4 | no_candidate |
| 715 | zsNo4H | 5 | #1 TURN_RIGHT | 3 | 23 | 1 / 1 | gate_right |
| 721 | 2azQ1b | 4 | #0 TURN_AROUND | 2 | 15 | 0 / 1 | gate_rear |
| 739 | QUCTc6 | 5 | #2 TURN_RIGHT | 5 | 53 | 2 / 2 | gate_right |
| 743 | zsNo4H | 6 | #0 EXIT_REGION | 4 | 33 | 0 / 3 | no_candidate |
| 748 | X7HyMh | 4 | #1 TRAVERSE_PORTAL_REGION | 5 | 35 | 1 / 3 | no_candidate |
| 755 | EU6Fwq | 4 | #1 TURN_LEFT | 5 | 42 | 1 / 3 | no_candidate |
| 765 | EU6Fwq | 3 | #0 TURN_RIGHT | 1 | 0 | 0 / 0 | no_candidate |
| 787 | oLBMNv | 5 | #1 TURN_LEFT | 10 | 94 | 1 / 8 | no_candidate |
| 821 | zsNo4H | 5 | #0 EXIT_REGION | 5 | 22 | 0 / 4 | no_candidate |
| 1051 | x8F5xy | 5 | #0 TURN_AROUND | 2 | 5 | 0 / 1 | gate_rear |
| 1056 | x8F5xy | 3 | #1 TURN_RIGHT | 6 | 56 | 1 / 4 | gate_right |
| 1061 | QUCTc6 | 4 | #2 TURN_LEFT | 5 | 55 | 2 / 2 | gate_left |
| 1077 | TbHJru | 5 | #0 TURN_RIGHT | 3 | 25 | 0 / 2 | gate_right |
| 1084 | zsNo4H | 6 | #2 TURN_LEFT | 7 | 51 | 2 / 4 | gate_left |
| 1085 | zsNo4H | 6 | #1 TURN_LEFT | 4 | 29 | 1 / 2 | gate_left |
| 1087 | 2azQ1b | 5 | #0 TURN_RIGHT | 1 | 0 | 0 / 0 | gate_right |
| 1092 | X7HyMh | 4 | #0 TURN_TO_LANDMARK | 4 | 29 | 0 / 3 | no_candidate |
| 1106 | QUCTc6 | 3 | #2 STOP_WAIT | 7 | 59 | 2 / 4 | no_candidate |
| 1133 | Z6MFQC | 3 | #0 ADVANCE_STRAIGHT | 4 | 40 | 0 / 3 | no_candidate |
| 1142 | QUCTc6 | 4 | #3 STOP_WAIT | 8 | 58 | 3 / 4 | no_candidate |
| 1148 | 8194nk | 5 | #1 CIRCUMNAVIGATE | 6 | 76 | 1 / 4 | no_candidate |
| 1284 | TbHJru | 5 | #1 TURN_LEFT | 8 | 68 | 1 / 6 | no_candidate |
| 1301 | Z6MFQC | 3 | #2 STOP_WAIT | 9 | 81 | 2 / 6 | no_candidate |
| 1307 | TbHJru | 3 | #2 STOP_WAIT | 9 | 77 | 2 / 6 | no_candidate |
| 1406 | TbHJru | 4 | #0 TURN_LEFT | 5 | 54 | 0 / 4 | no_candidate |

「节点」= `navigation_graph.json` 里的节点数（含起点），「步数」= 图中所有边的 `control_step_count` 之和。崩溃 episode 没有 `evaluation_only/evaluation_geometry.json`，因此无法给出隐藏几何的终点距离。id 116 的 `vlm_parse` 是 DeepSeek 连续 3 次返回缺少 `view_index` 字段的 JSON（`vlm_calls_attempts.json` 只记了 `error: 'view_index'`，没有保留原始响应，无法进一步判断）。

## 5. 模块级观察

### 5.1 走路层（执行器 + TAPIR + GNM）——189 次点导航（仅正常结束的 25 条有隐藏几何）

- 执行器报「到达」155 次、`max_steps` 等未到达 34 次，到达信号率 82%。
- 报到达时距隐藏参考点：中位 0.94 m，≤0.75 m 48 次（31%），≤1.0 m 89 次（57%），≤1.5 m 140 次（90%），>2 m 11 次，>4 m 5 次。与上一轮（中位 0.93 m，≤0.75 m 26%）一致：**走路层说「到了」时九成在 1.5 m 内，但按 0.75 m 严格口径只有三成**。对应 `summary.json` 的 `arrival_signal_precision=0.30`、`recall=0.92`。
- 34 次未到达里 5 次隐藏距离已 ≤1.2 m（差几步被 20 步上限截断），9 次 >4 m（选点太远，走路层无法负责）。上一轮建议的「把批量入口 `--max-steps-per-target` 与单 episode 入口的 32 对齐」仍然成立，本轮未做。

### 5.2 判定层——498 次边判定（100 条全部）

- 总体：**completed 103、unknown 395（79%）**。completed 置信集中在 0.7–0.85（均值 0.73），unknown 集中在 0.6（均值 0.62），置信度基本没有信息量。
- 按 form 拆：转向类（TURN_LEFT/RIGHT/AROUND）**completed 13、unknown 90**；非转向类 completed 90、unknown 305。转向类的 unknown 率 87%，明显高于其它 form 的 77%。
- 90 个转向类 unknown 里 **78 个的理由明确写着「action history 中只有前进、0 次转向命令」**（正则统计，含 "zero right-turn commands"、"only forward"、"consecutive forward moves" 等）。这不是模型看错，见第 6.2 节——它确实没看到转向命令。
- 判定假阳性：id 469 四次 completed 后 STOP 在距目标 16 m 处；id 546 两次 completed 后 STOP 在半径外。没有独立核验时无法统计整体假阳性率，但 8 次 STOP 里 2 次位置错误（25%）已经足以说明 completed 也不可靠。

### 5.3 选点层——641 次 `select_ground_target` + 18 次 `select_ground_sector`

- 除 id 116 的 3 次 `view_index` 缺失外全部返回有效 JSON。
- 75 次崩溃全部在选点层的候选过滤阶段（VLM 调用之前）抛出，即「过滤后无候选」没有被当作一种正常的选点结果返回给策略层。
- 显式方向门（`direction_gate`）在 TURN_* 子指令上只允许 2–3 个视图，配合「来路方向排除」和「已封锁方向硬排除」后极易清空。

### 5.4 分解层——100 条全部分解成功

- 子指令数分布：1 段 3 条、2 段 8 条、3 段 24 条、4 段 23 条、5 段 20 条、6 段 16 条、7 段 3 条、8 段 3 条。
- form 频次：STOP_WAIT 92、TURN_LEFT 51、TURN_RIGHT 45、ENTER_REGION 42、EXIT_REGION 40、TRAVERSE_PORTAL_REGION 31、ADVANCE_STRAIGHT 24、PASS_LANDMARK 23、APPROACH_LANDMARK 18、OTHER 15。**转向类子指令共 104 条（TURN_LEFT 51、TURN_RIGHT 45、TURN_AROUND 8）、几乎每条指令都有一段**，因此第 6.2 节的缺陷影响面是全局的。
- 上一轮指出的可疑分解（id 94 "take a left around the corner" → TURN_AROUND）本轮复现，id 94 再次 0 完成、6 次 unknown、终点 11.68 m（与上一轮完全一样）。

### 5.5 按场景

| 场景 | 条数 | 崩溃 | 成功 |
|---|---:|---:|---:|
| zsNo4HB9uLZ | 20 | 16 | 1（670） |
| QUCTc6BB5sX | 16 | 11 | 0 |
| 2azQ1b91cZZ | 14 | 9 | 2（11、810） |
| TbHJrupSAjP | 14 | 10 | 3（308、531、698） |
| Z6MFQCViBuw | 11 | 8 | 0 |
| EU6Fwq7SyZv | 8 | 5 | 0 |
| X7HyMhZNoso | 7 | 7 | 0 |
| x8F5xyUWy9e | 6 | 5 | 0 |
| oLBMNvg9in8 | 3 | 3 | 0 |
| 8194nk5LbLH | 1 | 1 | 0 |

X7HyMhZNoso、oLBMNvg9in8、8194nk5LbLH 全部崩溃，没有任何可评分数据。

## 6. 根因分析（产物证据 + 源码静态阅读，尚未用修复验证）

### 6.1 未捕获异常（直接原因，75 条）

`rgb_only_instruction_sequence.py:275` 调 `select()` 没有 try/except；`vlm_harness.select_ground_target` 在候选被过滤为空时 raise `RuntimeError`（`vlm_harness.py:2677` 和 `:2929` 两处）。旧的非 RGB-only 策略会把同一异常转成 `end_reason="vlm_selection_failed: …"` 正常收尾，RGB-only 重写时丢了这层保护。上一轮报告已指出并建议先修再跑全量；本轮未修，结果与预估一致（上一轮按 40% 比例预估约 40 条白跑，实际 75 条）。

修复后这 75 条**绝大多数会变成 `directions_blocked` 类的正常失败**而不是成功（第 4.2 节），但会保住 43 条已完成子指令的记录、补齐 `trajectory.json`/隐藏几何，让 100 条的走路层与判定层统计有完整样本。

### 6.2 转向子指令的「转向命令丢失」（上游原因，影响约 100 条转向子指令）

证据链：

1. id 469 第 1 跳：`selected_action_heading_rad=-90°`、`direction_gate.sector=right`、`allowed_views=[5,6,7]`——agent 确实面向右侧视图后出发；但该跳 `action_history` 的 12 条记录全是 `move_forward`，判定理由为「12 次前进、0 次右转命令，未完成 Turn right」→ unknown。第 2 跳同样（17 次前进），再 unknown → 封锁 → 回溯。第 3 跳 GNM 自己发了 2 次 `turn_right` 才被判 completed。
2. 源码：转向到选定视图的动作由 `point_selectors.continuous_turn()`（`point_selectors.py:808`）执行，它在 `rgb_only_v1` 契约下确实通过 `policy_sim.step("turn_left"/"turn_right")` 发出离散转向动作，并记录到 **`motion_log`**；而执行器 `PointNavigationExecutor.execute()`（`point_navigation_executor.py:1020-1022`）的 `action_history` 从空列表开始，只记录 GNM 阶段的动作。边的 `action_history` 与 `edge_keyframes` 都来自执行器，**因此判定层拿到的动作历史里永远没有「转到选定视图」这一段旋转**。
3. 统计：90 个转向类 unknown 里 78 个明确引用「0 次转向命令」；转向类 completed 只有 13 次，且多发生在 GNM 途中恰好自己转了几步的跳上。

后果：几乎每条指令都有一段 TURN_*；这一段几乎必被判 unknown → 封锁方向 → 回溯 → 在 2–3 个视图的方向门里很快无候选 → 要么崩溃（24 条 gate 崩溃 + 13 条 no_candidate 转向崩溃），要么 `directions_blocked`。这是一个**instruction-form 级**的系统缺陷，符合 16.4 节「可抽象」的修复范畴：把 `continuous_turn` 产生的转向动作并入该边的 `action_history`（以及 keyframe 的起始帧应取转向前的帧），使判定看到的动作历史与真实动作一致。它同时属于 Rk-B（执行器/边记录）与 Rk-C（判定输入），按规则应先在 id 469、546 这种真实 state 上做单点模块测试，再十 EP 回归。

### 6.3 判定层整体偏保守且置信无信息（79% unknown）

即使排除转向类，非转向 form 的 unknown 率也有 77%。STOP_WAIT 作为最后一段在 11 条崩溃 + 多条 `directions_blocked` 里反复 unknown，是「已走到终点附近却发不出 STOP」的主要原因。这需要独立核验数据（`verify_round_stage_completions.py`）才能判断是 prompt 偏保守还是到达点确实不对；本轮未运行。

## 7. 结论与建议（按优先级）

1. **必须修，零风险（Rk-A/策略层）**：`rgb_only_instruction_sequence.py:275` 的 `select()` 外补 `except RuntimeError → end_reason`，与旧策略一致；`evaluate_point_navigation.termination_category` 里已有的 `no_floor_bearing_candidate` 分类即可用上。不修则任何全量数字都没有意义。
   - **已于 2026-09-11 修复**（`rgb_only_instruction_sequence.py` 的 `run()` 捕获 `RuntimeError` 并以 `vlm_selection_failed: …` 结束；新增 `tests/test_rgb_only_instruction_sequence.py`，全量 311 个单元测试通过）。真实 state 诊断 `outputs/e2e_eval/20260911_104247_no_candidate_fix_smoke`（id 52、7，单 worker）：两条 `returncode=0`、无 traceback；id 52 复现右侧方向门失败并被评为 `termination_category=no_floor_bearing_candidate`，`trajectory.json`/`evaluation_geometry.json`/mp4 齐全；id 7 本次走了 8 跳后以 `all_candidate_directions_blocked_at_verified_node` 正常终止，保留 1 条已完成子指令与 10.45 → 3.33 m 的进度记录。仅为诊断运行，未做十 EP 回归。
2. **必须查，影响面最大（Rk-B + Rk-C）**：把 `continuous_turn` 的转向动作并入边的 `action_history`/keyframes（第 6.2 节）。先用 id 469 第 1 跳的 state 做单点复现（预期：并入后判定理由不再出现「0 次转向命令」），再十 EP 回归；不要改 prompt 来「提示模型忽略缺失的转向命令」——那是补丁式优化。
3. **配置对齐（Rk-B）**：批量入口 `--max-steps-per-target 20` 与单 episode 入口 32 不一致，本轮 5 次「≤1.2 m 被截断」再次出现。
4. **判定核验（Rk-C）**：对本轮 8 条 STOP episode（尤其 469、546）跑 `verify_round_stage_completions.py`，量化 completed 的假阳性率；在此之前不要调判定 prompt。
5. **工具修正**：`audit_active_stop_round.py` 不要再出现在 `run_end_to_end_eval.py` 结束时的提示里，或者改成不覆盖 `summary.json`。
6. **下一轮全量前**：先完成第 1、2 条并在固定十 EP 上回归，再跑 100 条；预计耗时 2–2.5 小时（崩溃消失后单条会变长）、token 约 1200 万、磁盘约 3 GB。

## 8. 复现与查看

```bash
bash run_e2e_eval.sh --all --workers 2                  # 本轮命令（在 tmux 会话 e2e_opennav100_0911 中执行）
# 单条产物：outputs/e2e_eval/20260911_001436_e2e_opennav/shard_N/episode_XXXX/
#   正常结束：trajectory.json, vlm_calls.json, episode_XXXX.mp4, navigation_graph/, evaluation_only/evaluation_geometry.json
#   崩溃：process.log（末尾 traceback）, vlm_calls.json, navigation_graph/, episode_XXXX.tmp.mp4（未封装）
python scripts/verify_round_stage_completions.py outputs/e2e_eval/20260911_001436_e2e_opennav \
  --dataset /workspace/3d_wm_vln/StreamVLN/data/datasets/r2r/val_unseen/val_unseen.json.gz   # 独立核验，花 VLM 费用，本轮未跑
# 不要对本轮目录运行 audit_active_stop_round.py（会覆盖 summary.json）
```
