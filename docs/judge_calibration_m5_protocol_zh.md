# M5：RGB-only 完成判定双向校准——标签口径与验收门槛（2026-09-12，实验前冻结）

本文件在任何 v3 判定 prompt 回放之前写定；写定后只允许追加「结果」一节，不得修改口径与门槛（`project_rulle.md` §0「规则变更必须在实验开始前写入」、§16.3「人工完成标签必须在模型调用前冻结且不输入模型」）。

## 1. 要解决的问题

来源：`docs/e2e_eval_reports/20260911_235744_opennav100_aligned_actrev/failure_analysis_95_plain_zh.md` 病根 D / E。

- **D 保守**：几何上「在路上」的 149 跳里 68 跳（46%）被判 unknown；独立复核推翻 65 次 unknown。判词里出现「动作记录以前进结尾 / 未静止」「距离未知」这类系统结构性、与完成无关的理由。
- **E 假阳性**：167 次 completed 里 66 次落在偏离参考路线的跳上；9 次错误 STOP 全是 STOP_WAIT 假阳性，且独立复核 VLM 对 8/9 也说 completed——「画面里看得见同类地标」就点头，与站位无关。

M1（`v2_form_aware_stop_relation`）只处理了 STOP_WAIT 的运动状态措辞；剩余的圈内 unknown（810「still inside bedroom」、259 地毯只在底边、1106 冰箱不在视野）与全部假阳性归本轮。

## 2. 数据与标签（冻结）

- 数据：上述轮次 467 条在线判定（在线 prompt 版本 `v1_baseline`），每条带独立复核（`stage_completion_verification.json`，复核器看不到在线结论）与隐藏几何（`judge_audit/judge_audit.csv`）。
- 标签脚本：`scripts/build_judge_calibration_labels.py` → `data/judge_calibration_labels_20260911_235744_v1.json`（确定性，可重建）。
- 口径：
  - 在路上 = `executed_heading_error_to_gt_deg ≤ 30 ∧ gt_path_progress_delta_m > 0.5 ∧ distance_to_reference_path_m < 1.5`
  - 偏离 = `executed_heading_error_to_gt_deg > 60 ∨ gt_path_progress_delta_m ≤ 0 ∨ distance_to_reference_path_m > 2.5`
  - **POS**（应判 completed）= 最后一句且终点 3 m 圈内；或非最后一句、在路上、独立复核 True。
  - **NEG**（应判 unknown）= 最后一句且离终点 > 3 m；或非最后一句、偏离、独立复核 False。
  - **AMB** = 其余，回放但不计分。
- 计数：POS 134 / NEG 188 / AMB 145。dev / holdout 按 episode_id 升序排名奇偶切分，各 48 个 episode（dev POS 67 / NEG 89 / AMB 65；holdout POS 67 / NEG 99 / AMB 80）。
- 在线 v1 基线（全 322 条计分样本）：TP 81 / FN 53 / FP 18 / TN 170 → recall 0.60、precision 0.82、F1 0.70。
- 标签只用于事后评分；不进入任何 prompt、候选排序、控制器或判定。

## 3. 回放方式

`scripts/replay_rgb_only_judge_round.py`：从 `vlm_calls.json` 里存储的 prompt 解析出五个 JSON 块，用指定版本重新渲染（`NavigationVLMHarness.render_rgb_only_completion_prompt`），把当时的三张 contact sheet 原样再发给 DMXAPI（`deepseek-v4-flash-vision-exp`，thinking 关闭）。除 prompt 版本/schema 外没有任何输入变化；不跑 Habitat/GPU。判定按 `(episode_id, target_index, sub_instruction_id)` 与标签连接。

步骤（顺序固定）：

1. v2（当前默认）回放 dev + holdout 全部 467 跳 → v2 基线（在线跑的是 v1，必须补齐）。
2. v3 草稿回放 **dev**；看混淆矩阵与 reason 文本，最多再出一个修订草稿（v3b）并只在 dev 上比较。所有在 dev 上评过的草稿都保留在 `RGB_ONLY_COMPLETION_PROMPT_VERSIONS` 注册表里，不删除。
3. 最终选定的一个 v3 回放 **holdout** 一次。holdout 结果出来后不得再改该版本。

## 4. 验收门槛（预注册）

在 **holdout**（POS 67 / NEG 99）上，与 v2 回放比较，三条同时满足：

1. F1 ≥ v2 F1 + 0.05；
2. FP 数（NEG 判 completed）≤ v2 FP 数；
3. 最后一句 NEG 上的 FP（= 会导致错误 STOP 的假阳性）≤ v2。

满足 → 三个评测入口的 `--rgb-only-completion-prompt-version` 默认切到 v3；不满足 → v3 保留在注册表，默认仍为 v2，如实记录。无论结果如何都追加到 `docs/curriculum_10ep_round_log.md`。固定十 EP 同轮回归不在本文件范围内，另行安排；在此之前 v3 不构成冻结配置。

## 5. 结果（2026-09-12 追加，口径与门槛未改）

回放目录：`outputs/replay/m5_v2_all`（v2，467 跳）、`m5_v3_dev`（v3 草稿一，221 跳）、`m5_v3b_dev`（v3b 草稿二，221 跳）、`m5_v3b_holdout`（v3b，246 跳）；共 1168 次 DMXAPI 调用、约 1100 万 token（每次约 9.5k，三张 contact sheet），4 线程约 5–8 分钟一轮。三张表里 "v1" 是本轮在线判定的原始结果（`old_status`），其余是回放。

**同一份 322 条计分样本上，三个版本的表现（TP/FN/FP/TN，P/R/F1；「末句 NEG FP」= 会触发错误 STOP 的假阳性 / 末句 NEG 总数）：**

| 版本 | 范围 | TP/FN/FP/TN | P / R / F1 | 末句 NEG FP | AMB 判 completed |
|---|---|---|---|---|---|
| v1_baseline（在线） | 全 467 | 81/53/18/170 | 0.818 / 0.604 / 0.695 | 9/63 | 68/145 |
| v2_form_aware_stop_relation（回放） | 全 467 | 83/51/28/160 | 0.748 / 0.619 / 0.678 | 14/63 | 75/145 |
| v1（在线） | dev 221 | 39/28/10/79 | 0.796 / 0.582 / 0.672 | 4/31 | 27/65 |
| v2（回放） | dev | 41/26/17/72 | 0.707 / 0.612 / 0.656 | 10/31 | 33/65 |
| v3_sector_relation_evidence（草稿一） | dev | 42/25/15/74 | 0.737 / 0.627 / 0.677 | 8/31 | 25/65 |
| v3b_relation_rules_relaxed（草稿二） | dev | 46/21/12/77 | 0.793 / 0.687 / **0.736** | 7/31 | 29/65 |
| v1（在线） | holdout 246 | 42/25/8/91 | 0.840 / 0.627 / 0.718 | 5/32 | 41/80 |
| v2（回放） | holdout | 42/25/11/88 | 0.792 / 0.627 / 0.700 | 4/32 | 42/80 |
| **v3b（最终候选）** | **holdout** | 44/23/12/87 | 0.786 / 0.657 / **0.715** | **7/32** | 44/80 |

**门槛判定（holdout，与 v2 比）**：(1) F1 0.715 < 0.700 + 0.05 = 0.750 → 不满足；(2) FP 12 > 11 → 不满足；(3) 末句 NEG FP 7 > 4 → 不满足。**三条全部不满足，v3b 不能成为默认**；`v3_sector_relation_evidence` / `v3b_relation_rules_relaxed` 保留在注册表可选，三个入口默认仍为 `v2_form_aware_stop_relation`。

**读数**：
- dev 上 v3b 的增益几乎全部来自 STOP_WAIT（0/20/4/27 → 8/12/7/24）：810（3 跳，0.1–1.2 m）、1106（2 跳）、7（2 跳）、259（1 跳）这些人在圈内的末句改判 completed——正是 ② 组的 4 个 episode。但 holdout 的 STOP_WAIT 没有同样的收益（v1 3/8/5/31 → v3b 3/8/6/30）：362（open door on left，0.7–0.8 m，地标就在左侧近处）仍被以「keyframes 显示继续前进 / 不确定是不是那扇门」拒绝；321（bathroom）、821（sink 检测很弱）、1301（doors 不可辨认）是地标辨认问题；586 的 landmark 被拆解成 "wait"（无地标，属拆解问题）。**dev 上的 STOP_WAIT 改善是 dev 特有的，没有泛化。**
- 假阳性方向没有改善：holdout 末句 NEG FP 从 v2 的 4 涨到 7，新增的是 748（dining room table，离终点 3.6–4.9 m，桌子确实在近处——同类不同实例/边界距离）。所有版本共同的 FP（454 sink、469 glass panes、546 closet doors、755 staircase、403、1056、1139）都是「同类地标在近处但不是那一个」，prompt 里的 FALSE-POSITIVE GUARD 不能让 VLM 分辨实例，这是 M6（选点/地标身份）而不是判定 prompt 能解决的。
- 一个本协议之外但必须记录的事实：**在同一份标签上，当前默认 v2 的假阳性比 v1 多**（全 467：FP 18 → 28，末句 NEG FP 9 → 14；holdout：FP 8 → 11，末句 NEG FP 5 → 4）。M1 只用 ②/④ 组 58 跳做过回放；本轮 467 跳的全量回放显示 v2 在把 STOP_WAIT 运动状态措辞去掉的同时放松了整体口径。v2 是否保留为默认不在本协议门槛内，留给固定十 EP 回归和规则维护者决定。
- 逐 episode 表见各回放目录的 `scoring.md`（§16.3 要求的逐 EP 结果）。

**结论**：M5 的 prompt 迭代在 holdout 上未通过预注册门槛，判定层的剩余误差主要是地标实例身份与站位（M6），不再在判定 prompt 上追。
