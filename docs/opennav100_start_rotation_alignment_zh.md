# OpenNav100 起始朝向对齐数据集（2026-09-11）

规则正文见 `project_rulle.md` 第 17 节；本文记录动机、构建结果和使用方法。

## 1. 问题

官方 R2R VLN-CE `val_unseen.json.gz` 的 `start_rotation` 与指令开局转向词、GT
`reference_path` 初始方向经常不一致。对 100 条 OpenNav id 的量化（左为正，
Δ = GT 初始方向 − 官方 yaw）：

| 开局 form | 条数 | 官方 yaw 相对 GT 的表现 |
|---|---|---|
| `TURN_LEFT` | 6 | Δ 在 +43°~+149°，与"左转后面向路径"一致 |
| `TURN_RIGHT` | 10 | 7 条 Δ 在 −60°~−166°；546 / 1307 / 52 却是 +106°~+116°（路径在左侧） |
| `TURN_AROUND` | 6 | \|Δ\| 在 81°~180° |
| 非转向开局 | 78 | 大量背对或侧对路径：1133 "Walk straight…" Δ=−179°、842 "Go straight…" Δ=−94°、207 "Walk out of the room…" Δ=−179° |

这类起点会让"按指令开局"的选点在第一步就没有可用的地面候选（见
`docs/e2e_eval_reports/20260911_001436_e2e_opennav/failure_analysis_94_plain_zh.md` 病根 B），
与选点/导航/判定模块的能力无关。

## 2. 构建规则（form 级，不允许逐条手改）

- 分类：`instruction_taxonomy.decompose_by_definition(text)[0]["form"]`（确定性 regex，不用 VLM）。
- GT 初始方向：起点 → `reference_path` 上第一个沿折线累计距离 ≥ 1.0 m 的 waypoint；
  100 条中首段长度 0.57~7.22 m（中位 1.78 m），仅少数 0.6~0.9 m 的短首段会往后取一个点。
- 转向角：`TURN_LEFT=+90°`、`TURN_RIGHT=−90°`、`TURN_AROUND=180°`，其余 form 0°；
  `start_yaw = GT 初始方向 − 转向角`。
- `TURN_TO_LANDMARK`、`OTHER` 无法定义角度 → 保留官方值（`kept_official`）。
- 四元数 `[x, y, z, w]` 绕 +y，`yaw = 2·atan2(y, w)`；已与 `habitat_point_navigation.yaw_from_coeffs`
  和 `habitat_sim.quat_from_angle_axis` 交叉验证（37 个角度误差 < 1e‑7 rad）。

实现：`scripts/start_rotation_alignment.py`（纯逻辑）+
`scripts/build_opennav100_start_aligned_dataset.py`（构建与自检）。

## 3. 构建结果（rule_version `opennav100_start_aligned_v1`）

产物目录 `data/datasets/opennav100_start_aligned/`：

| 文件 | 内容 |
|---|---|
| `val_unseen_opennav100ids_start_aligned.json.gz` | 100 条，按 `data/opennav100_episode_ids.json` 顺序；除 `start_rotation` 外逐字节等于官方值 |
| `build_manifest.json` | 源文件 / 输出 sha256、规则参数、逐 form 计数、`rule_version` |
| `start_rotation_alignment_audit.jsonl` / `.md` | 逐条：开局 form、转向角、GT 锚点、GT 方位、官方 yaw、对齐 yaw、Δ、status |

统计：`rewritten` 96 / `kept_official` 4（116、550、1087 为 `OTHER`，1092 为 `TURN_TO_LANDMARK`）。
96 条改写中 40 条 |Δ| ≥ 90°（其中 `EXIT_REGION` 15 条），16 条 |Δ| < 15°（官方值本来就基本正对）。

抽查（来自审计表）：

| id | form | GT 方位 | 官方 yaw | 对齐 yaw | Δ | 说明 |
|---|---|---|---|---|---|---|
| 423 | TURN_LEFT | 86.8° | 0.0° | −3.2° | −3.2° | 官方已基本正确，左转 90° 后正对路径 |
| 461 | TURN_AROUND | 180.0° | 0.0° | 0.0° | 0.0° | 官方已正确 |
| 1133 | ADVANCE_STRAIGHT | −88.6° | 90.0° | −88.6° | −178.6° | 官方背对，改为正对 |
| 207 | EXIT_REGION | 91.0° | −90.0° | 91.0° | −179.0° | 同上 |
| 52 | TURN_RIGHT | −33.7° | −150.0° | 56.3° | −153.7° | regex 把"到大钟处再右转"判成开局右转，属已知局限 |
| 1092 | TURN_TO_LANDMARK | −1.3° | −150.0° | −150.0° | 0 | 保留官方值 |

## 4. 使用

```bash
bash run_e2e_eval.sh 7,11,13                      # 默认 --start-pose-source aligned
bash run_e2e_eval.sh 7 --start-pose-source official   # 官方 val_unseen 起始朝向
bash run_e2e_eval.sh --dry-run 7                  # 配置快照里有 "start pose src" 一行
bash run_e2e_eval.sh --start-pose-source official --episode-indices 0,3,6,9,18,27,45,126,204,219
```

- `--episode-indices` 指官方全量 split 的行号；与 `aligned` 同用时 preflight 拒绝（对齐文件只有 100 行，
  index 语义不同）。固定十 EP 协议不受影响。
- 显式 `--r2r-data <file>` 优先于开关，manifest 记为 `start_pose_source="explicit"`。
- 每轮 `manifest.json` 记录 `start_pose_source`、`benchmark`、`dataset_path`、`dataset_sha256`；
  `--resume` 沿用 manifest 里的来源并校验 sha256。
- 事后审计（`audit_active_stop_round.py` / `verify_round_stage_completions.py`）的 `--dataset`
  必须传该轮 manifest 的 `dataset_path`（启动器结束时打印的命令已经是正确文件）。
- 报告 OpenNav100 结果时必须标注 `aligned` / `official`，两者不得混表比较。
  2026-09-11 之前的 e2e 轮次全部是 `official` 起始朝向。

## 5. 重建与冻结

```bash
source local_env.sh
python scripts/build_opennav100_start_aligned_dataset.py   # 输出到 data/datasets/opennav100_start_aligned/
python -m unittest tests.test_start_rotation_alignment tests.test_opennav100_aligned_dataset -v
```

gzip 以 `mtime=0` 写出，同一规则下重建得到相同 sha256。`tests/test_opennav100_aligned_dataset.py`
在官方数据可读时逐条重算并比对提交的 `start_rotation` 与审计行。改规则须重建、更新
`project_rulle.md` 第 17 节并升级 `rule_version`。
