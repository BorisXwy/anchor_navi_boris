# 端到端评测报告：OpenNav100 前 10 条，双 worker 并行（2026-09-10）

- 轮次目录：`outputs/e2e_eval/20260910_224927_opennav10_w2/`
- 代码版本：`a6f29aa`（工作区干净）
- 配置：`--vlm-backend deepseek`（DMXAPI `deepseek-v4-flash-vision-exp`，thinking 关闭），其余全部为 `evaluate_point_navigation.py` 默认值（GNM 策略、dense-majority 地面分割、DINO+SAM 检测、8 视图、每跳最多 20 步、每 episode 最多 30 跳、选点 prompt `v10_approach_relation_router`、判定 prompt `v13_structured_node_edge_binary`、seed 17、`cuda:0`）
- 数据：官方 R2R VLN-CE v1-3 `val_unseen.json.gz`，episode id `7,11,13,40,42,52,70,94,116,140`（OpenNav100 前 10 个 id，位姿取官方值）
- 口径：本报告只使用运行时自带的 `simulator_reported_success`（在线判定最后子指令 completed 后主动 STOP 且 STOP 点在 goal radius 内）。`verify_round_stage_completions.py` / `audit_active_stop_round.py` **均未运行**，因此 `instruction_validated_r2r_success` 和"独立核验的子指令完成数"按 fail-closed 记为 0，不代表真实值。

## 1. 总览

| 指标 | 值 |
|---|---|
| 运行 episode 数 | 10 / 10（无 not_run、无 unscored） |
| 进程崩溃（记为失败） | 4（id 7、42、52、116） |
| 正常结束 | 6 |
| `simulator_reported_success` | **2 / 10**（id 11、13） |
| 主动 STOP 次数 | 2（全部落在 goal radius 3 m 内） |
| 终止原因 | `instruction_sequence_complete` 2；`all_candidate_directions_blocked_at_verified_node` 4；`episode_process_failed` 4 |
| 总控制步数 | 802 |
| 总 VLM 调用 | 140 次，全部 HTTP 200，无重试；平均 2.69 s，中位 2.58 s，p90 3.65 s |
| 总 token | 约 93 万 |
| 磁盘 | 208 MB（单条 2–49 MB） |

## 2. 基础设施结论（本轮首要目的）

- **双 worker 并行可行**：两个进程各占 11.7–11.8 GB，加上一个 384 MiB 的无关进程，总量稳定在 24.0 / 24.5 GB，全程无 OOM。余量只有约 0.5 GB，这是极限，不能再加 worker，也不能在有其它 GPU 进程时跑双 worker。
- **耗时**：启动 22:49:29，结束 23:00:05，**共 10 分 36 秒**。单条 20 s–4 min（崩溃的短，正常结束的 47 s–4 min 03 s），远快于事前估计的 1 小时。原因是每跳实际只走 3–20 步、DeepSeek 单次仅 2.7 s。按此推算 100 条全量双 worker 约 1.5–2 小时。
- 两个分片返回码都是 0，合并的 `summary.json`、`manifest.json`、`process.log` 齐全；分片布局 `shard_N/episode_XXXX/` 可直接交给两个审计脚本。

## 3. 逐 episode 结果

| index | id | 场景 | 参考路径 | 子指令 | 步数 | 跳数（到达） | 系统判完成 | 回溯 | 终止原因 | STOP | 成功 | 起点→终点 geodesic (m) |
|---:|---:|---|---:|---:|---:|---|---:|---:|---|---|---|---|
| 10 | 11 | 2azQ1b | 7.1 m | 2 | 29 | 2 (2) | 2 | 0 | sequence_complete | ✓ | **✓** | 7.11 → 0.73 |
| 12 | 13 | zsNo4H | 12.5 m | 4 | 128 | 9 (8) | 4 | 3 | sequence_complete | ✓ | **✓** | 12.04 → 1.86 |
| 39 | 40 | zsNo4H | 15.4 m | 7 | 238 | 15 (7) | 2 | 2（+5 方向封锁） | directions_blocked | ✗ | ✗ | 14.96 → 12.78 |
| 69 | 70 | 2azQ1b | 16.0 m | 6 | 229 | 17 (12) | 3 | 5 | directions_blocked | ✗ | ✗ | 15.64 → 9.07 |
| 93 | 94 | QUCTc6 | 14.9 m | 4 | 69 | 6 (6) | 0 | 5 | directions_blocked | ✗ | ✗ | 13.93 → 11.68 |
| 139 | 140 | 2azQ1b | 13.6 m | 3 | 109 | 9 (7) | 2 | 4 | directions_blocked | ✗ | ✗ | 13.37 → 19.42 |
| 6 | 7 | x8F5xy | 10.5 m | 3 | — | 4 跳后崩溃 | 0 | — | process_failed | — | ✗ | — |
| 41 | 42 | zsNo4H | 15.4 m | 6 | — | 3 跳后崩溃 | 2 | — | process_failed | — | ✗ | — |
| 51 | 52 | zsNo4H | 16.0 m | 3 | — | 第 1 跳选点即崩溃 | 0 | — | process_failed | — | ✗ | — |
| 115 | 116 | 2azQ1b | 7.1 m | 5 | — | 5 跳后崩溃 | 2 | — | process_failed | — | ✗ | — |

### 3.1 两条成功案例

- **id 11**（"Walk across the floor and wait at the archway"）：2 条子指令 2 跳直达，两次到达点距隐藏参考点 0.39 m / 0.71 m，判定两次 completed（置信 0.7），终点距目标 0.73 m。教科书式路径。
- **id 13**（餐桌→厨房→经过灶台→水槽旁停）：9 跳、3 次回溯。前 4 跳完成了 3 条子指令；第 4–7 跳判定连续 4 次 unknown（其中两次到达点离参考点 1.19 / 1.34 m，属于走偏），触发回溯，第 8 跳判定 completed 并 STOP，终点距目标 1.86 m（半径 3 m 内）。说明"unknown → 回溯 → 换方向"这条恢复链路在真实场景里是能工作的。

### 3.2 四条"方向全部封锁"失败

共同模式：系统在某个已验证节点上反复尝试新方向，每次到达后判定 unknown，把该方向封锁；同一节点累计 5 个封锁方向就终止（`max_blocked_directions_per_node=5`，`instruction_sequence_exploration.py:902`）。四条里 35 次恢复回溯全部"物理成功"（回到了节点），但都没换来 completed。

- **id 40**（7 条子指令，含 TURN_RIGHT / TURN_LEFT）：前两跳走满 20 步未到达（隐藏距离 0.72 m 和 3.66 m——第一跳其实差一点就到了）；之后在 node_0005 封锁 5 个方向。7 条子指令只完成 2 条，终点离目标 12.8 m。指令越长（7 段）失败面越大。
- **id 70**（"through living room / door on the right / den / dining / outdoor foyer"，5 个 TRAVERSE_PORTAL_REGION 连击）：17 跳、5 次回溯，完成 3 条；3 次 max_steps 未到达时隐藏距离 6.9 m、6.5 m，说明选点选到了很远的点，20 步走不完。
- **id 94**（"walk forward, take a left around the corner, end of hallway"）：6 跳全部到达（隐藏距离 0.8–1.0 m，走路层很准），但 **6 次判定全是 unknown，一条子指令都没完成**，在 node_0001 封锁 5 个方向后终止。第 2 段被分解成 TURN_AROUND（"take a left around the corner" 应是拐弯而非掉头），这是分解/判定层的问题，不是走路的问题。
- **id 140**（三个门洞）：前两段完成（到达点距参考点 0.17 m、0.70 m），第 3 段"楼梯左边的门"连续 5 次 unknown 后终止；终点离目标 19.4 m——比起点还远 6 m，说明后半程走反了方向。

### 3.3 四条进程崩溃

四条崩溃是**同一个未捕获异常**，不是 GPU、不是网络、不是启动器：

- 抛出点：`vlm_harness.select_ground_target`——`RuntimeError("No floor-bearing candidate can be sent to the VLM")`（id 7，第 5 跳）或 `RuntimeError("No floor-bearing candidate remains inside the explicit right direction gate")`（id 42 / 52 / 116，均在 TURN_RIGHT 子指令处：候选地面点里没有一个落在"右侧扇区"里）。
- 未捕获点：`rgb_only_instruction_sequence.py:275` 直接调用 `self.point_selector.select(...)`，没有 try/except。旧的（非 RGB-only）策略 `instruction_sequence_exploration.py:2103-2106` 会把同一异常转成 `end_reason="vlm_selection_failed: ..."` 让 episode 正常收尾；RGB-only 重写时丢了这层保护，`evaluate_point_navigation.termination_category` 里对应的 `no_floor_bearing_candidate` 分类因此永远用不上。
- 后果：没有 `trajectory.json`、没有 `evaluation_only/`、没有 STOP 记录，评分脚本按 `episode_process_failed` 记 0 分。其中 id 42 和 116 崩溃前已经各完成 2 条子指令，这些进度全部丢失。
- id 52 最典型：第一条子指令就是 "Take a right at the large clock"，起点视图右侧没有可走地面，**一次选点 VLM 调用都没发出**就崩了（仅 1 次分解调用，20 秒）。

## 4. 模块级观察（6 条正常结束的 episode，58 跳）

**走路层（执行器 + TAPIR + GNM）**
- 执行器报"到达" 42 / 58（72%）；15 次 max_steps 未到达。
- 到达时距隐藏参考点：中位 0.93 m，≤0.75 m 的 11 次，≤1.0 m 的 26 次，≤1.5 m 的 36 次，>2 m 的 3 次。即执行器说"到了"时，绝大多数在 1.5 m 内，但按 0.75 m 的严格口径只有 26%。
- 15 次 max_steps 里有 4 次隐藏距离已经 ≤1.2 m（0.3、0.7、0.9、1.2 m）——**差几步就到了却被 20 步上限截断**。`evaluate_point_navigation.py` 默认 `--max-steps-per-target 20`，而单 episode 入口默认 32，两处默认值不一致，值得对齐后再看。
- 另外 6 次 max_steps 隐藏距离 >4 m（最远 6.9 m），是选点选得太远，走路层无法负责。

**判定层**
- 42 次到达后判定：completed 13、unknown 29。completed 的置信度固定 0.7–0.75，unknown 固定 0.6，说明模型给的置信度信息量很低。
- 没有独立核验，无法判断 13 个 completed 里有多少是真的；但 id 94 六次到达都很准却六次 unknown，提示判定偏保守（或分解把 "take a left around the corner" 标成 TURN_AROUND 导致判定标准错位）。

**选点层**
- 73 次 `select_ground_target` + 3 次 `select_ground_sector`，全部返回有效 JSON。
- 4 次崩溃全在选点层（见 3.3），且 3 次集中在 TURN_RIGHT 形式的"显式方向门"上——这是一个可抽象的 instruction-form 级问题，不是单 episode 问题。

**分解层**
- 10 条全部成功分解（2–7 段）。可疑分解：id 94 的 "take a left around the corner" → TURN_AROUND；id 40 的 "Walk by the sink and oven" → OTHER；id 70 的 "to the outdoor foyer" → OTHER。

## 5. 结论与建议（按优先级）

1. **必须修（Rk-A，选点/策略）**：在 `rgb_only_instruction_sequence.py` 的 `select(...)` 外补回 `except RuntimeError → end_reason="vlm_selection_failed: ..."`，与旧策略一致。这是零风险改动，能把 4 条"进程崩溃"变成可评分的正常失败并保住已完成的子指令记录。之后再看 TURN_RIGHT 方向门为什么在这三个场景里找不到右侧地面（是否需要像左右扇区已有的 forward fallback 那样有一条被审计的兜底）。
2. **值得试（Rk-B，走路）**：把批量入口的 `--max-steps-per-target` 与单 episode 入口对齐到 32，4 次"差几步就到"的 max_steps 可能直接变成到达。这是配置对齐而非调参，但按规则要作为独立候选并在十 EP 上回归。
3. **需要数据（Rk-C，判定）**：先对本轮跑 `verify_round_stage_completions.py`（会产生额外 VLM 费用）拿到 13 个 completed 的独立核验结果，再决定判定 prompt 是否偏保守；`audit_active_stop_round.py` 在本轮不适用（它只认固定十 EP 的索引）。
4. **基础设施**：双 worker 配置可以作为默认；全量 100 条预计 1.5–2 小时、约 2 GB 磁盘、约 900 万 token。跑之前先修第 1 条，否则按本轮比例约 40 条会白跑。

## 6. 复现与查看

```bash
bash run_e2e_eval.sh 7,11,13,40,42,52,70,94,116,140 --workers 2 --run-tag opennav10_w2
python scripts/verify_round_stage_completions.py outputs/e2e_eval/20260910_224927_opennav10_w2 \
  --dataset /workspace/3d_wm_vln/StreamVLN/data/datasets/r2r/val_unseen/val_unseen.json.gz   # 独立核验，花 VLM 费用
# 单条产物：shard_N/episode_XXXX/{trajectory.json, vlm_calls.json, exploration.mp4, navigation_graph/, evaluation_only/}
```
