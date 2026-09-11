# 裁判（完成判定）独立复核报告（2026-09-11）

- 复核对象：在线完成判定 `RGBOnlyNodeTransitionInstructionCompletionJudge`（`trajectory.json` 里记的 prompt 版本 `v13_structured_node_edge_binary`，`vlm_calls.json` 里的任务名 `judge_edge_instruction_completion_rgb_only`），即 `remaining_issues_priority_zh.md` 的**第 3 位**问题。
- 数据：主轮 `outputs/e2e_eval/20260911_001436_e2e_opennav/`（OpenNav100 全量 100 条，代码 `8fe1404`，**转向动作记录修复之前**）+ 修复后的 2 条 smoke `outputs/e2e_eval/20260911_113312_turn_history_smoke/`（id 469、546，代码 `2a3dbd9`）。
- 工具（本次新增/修改，均已提交）：`scripts/verify_round_stage_completions.py --include-unknown --always-call-vlm`（独立 VLM 复核，见第 2 节）、`scripts/analyze_judge_round.py`（确定性统计，不调 VLM）、`scripts/postrun_hidden_geometry.py`（事后读隐藏几何）。产物在各轮的 `judge_audit/`（`judge_audit.json/.csv`、`judge_audit_raw_calls.csv`、`judge_audit_summary.md`）和每个 episode 目录的 `stage_completion_verification.json`。
- **实际运行 / 静态解析的区分**：
  - 第 3 节的 498 次线上判定统计来自解析 100 条 episode 的 `vlm_calls.json`（纯静态，含 75 条崩溃 episode）。
  - 第 4–7 节的独立复核结果来自**实际运行**的云端 VLM 调用：主轮 177 次（158 个判定 + 19 个多跳 stage 候选，15:38–15:46，约 8 分钟，DMXAPI 全部 200）、smoke 轮 11 次。
  - 隐藏几何来自 `evaluation_only/evaluation_geometry.json` 与官方 `val_unseen.json.gz` 的 `reference_path`，只用于事后打分，从未进入任何模型输入。
- 单元测试：全量 329 个通过（含本次新增 15 个）。

---

## 1. 一句话结论

**裁判不是「乱判」，而是「看不见就不敢点头」，同时它和独立复核的第二个 VLM 都会被相似的场景骗过去。** 具体说：

1. 线上 38 次「完成」判定，独立复核 **38/38 也说完成**——但其中 24 次隐藏几何门不通过，10 次判定时离参考路线超过 2 米；id 469 那种在 16 米外说「到了」的错误，第二个 VLM **同样没看出来**。所以「completed 是否可信」靠再问一个 RGB 模型是查不出来的，只能靠几何。
2. 线上 120 次「不确定」判定，独立复核认为 **41 次（34%）其实已经完成**，其中 17 次连隐藏几何门也通过（走对方向、有实际前进）。这 41 次里 25 次直接触发了「封方向 + 回溯」，是若干局零进展的直接原因（id 94、166、842 等）。
3. 「不确定」的理由里 **72/395 是「动作记录里没有转向命令」**，这是第 1 位已经修掉的输入缺失（H3），不是裁判本身的判断问题。修复后的 smoke 里 469/546 的第一句 "Turn right" 都是**第一次判定就 completed**。
4. 剩下的大头（约 200 次「还没到 / 地标还在前面」，约 50 次「地标不可见 / 不在该在的房间」）反映的是**选点+走路把 agent 送到了哪**：这些 unknown 判定的落点中位数离参考路线 1.8 米、沿路线净前进 0 米，也就是说**裁判说「没到」的时候，大多数情况下确实没到**。这不是 prompt 太严，是上游没把人送到位。

对应文档里的三个假设：H1（prompt 过严）**占约三分之一的 unknown**（41/120）且集中在 ADVANCE_STRAIGHT / TRAVERSE_PORTAL / APPROACH 这类「过程型」子指令；H3（输入不足）占 72/395 且已修；H2（到达点确实不对）占剩余的大多数。**结论：不要现在改 prompt 去「放松」裁判**——那会把 H2 那部分变成假阳性；应该先修选点/走路层，只针对 H1 的「过程型子指令」考虑判定规则。

---

## 2. 复核方法（用大白话说）

在线裁判每次判定拿到的材料是：上一个节点的 8 张环视图、这条边的 5 张关键帧、当前节点的 8 张环视图、动作记录摘要、检测器给的物体列表，然后回答 completed / unknown。

**独立复核**做的事：把同一条边的材料（起点全景、关键帧、终点全景，外加动作摘要）交给**另一个 prompt**（`POST_RUN_INDEPENDENT_STAGE_BOUNDARY_AUDIT`），让它从零判断这一段有没有完成该子指令。它**不知道**在线裁判说过什么（status、置信度、理由一律不给）。这就像让第二位老师盲改同一份卷子。

然后再加一道**隐藏几何门**（只有考官能看）：这条边选的点方向、实际走的方向是否都在参考路线前方 30° 以内，且沿参考路线净前进 > 5 cm。三者都满足才算「几何通过」。

所以每个判定最后有三个标签：线上裁判怎么说、第二位老师怎么说、几何怎么说。

本次为了能跑起来做的三处兼容修改（`verify_round_stage_completions.py` 原来只认旧的 RGB-D 契约）：候选门不再要求 rgb-only 轨迹里不存在的 `navigation_physical_arrival` 字段；关键帧改从 `navigation_graph.json` 的边元数据读取；几何从 `evaluation_only/evaluation_geometry.json` 注入而不是从轨迹里读被契约禁止的 pose。另修了一个会误导复核器的 bug：rgb-only 动作记录没有 `moved_m`，原脚本把它汇总成「走了 0 米、14 次前进全部被挡」，第二位老师据此把真实走过的边判成「原地没动」——现在这些字段在没测量值时报 `null`，位移改从隐藏几何取。

**复核范围的硬限制**：100 条里只有 25 条正常结束、有 `trajectory.json` 和隐藏几何，可复核的判定 158 次（38 completed + 120 unknown）。另外 75 条崩溃 episode 的 340 次判定只能做文本统计（第 3 节），不能复核。

---

## 3. 全部 498 次线上判定长什么样（静态解析，100 条全覆盖）

### 3.1 按子指令类型

| 类型 | 判定数 | completed | unknown | unknown 率 | 该类型里「没有对应方向转向命令」的判定 |
|---|---:|---:|---:|---:|---:|
| EXIT_REGION 出房间 | 80 | 23 | 57 | 71% | — |
| STOP_WAIT 在某处停 | 65 | 8 | 57 | 88% | — |
| TURN_LEFT 左转 | 62 | 5 | 57 | 92% | 见 3.3 |
| ADVANCE_STRAIGHT 直走 | 43 | 5 | 38 | 88% | — |
| TURN_RIGHT 右转 | 36 | 7 | 29 | 81% | 见 3.3 |
| TRAVERSE_PORTAL_REGION 穿门/穿区域 | 36 | 6 | 30 | 83% | — |
| PASS_LANDMARK 经过某物 | 35 | 11 | 24 | 69% | — |
| ENTER_REGION 进房间 | 30 | 15 | 15 | 50% | — |
| APPROACH_LANDMARK 走向某物 | 14 | 9 | 5 | 36% | — |
| 其余（OTHER/FOLLOW/CROSS/BETWEEN/VERTICAL/…） | 97 | 14 | 83 | 86% | — |
| **合计** | **498** | **103** | **395** | **79%** | |

置信度确实没有信息量：completed 里 72/103 是 0.7；unknown 里 358/395 是 0.6。

### 3.2 「不确定」的理由分类（对 395 条理由做关键词归类，是启发式，5% 归不进去）

| 理由类别 | 转向类 | 非转向类 | 合计 | 说明 |
|---|---:|---:|---:|---|
| 还没到 / 地标还在前面 / 还在房间里 | 15 | 182 | **197** | 「current panorama still shows the bedroom interior」「the doorway has not been crossed」 |
| 动作记录里没有转向命令 | 61 | 11 | **72** | 「12 forward moves and 0 right-turn commands」——第 1 位已修的输入缺失 |
| 地标不可见 / 不在该在的地方 | 5 | 43 | 48 | 「no railing or balcony is visible」「moving through a living-room-like area rather than the kitchen」 |
| 证据不足 / 关系不能确认 | 6 | 40 | 46 | 「the spatial relation is not clearly satisfied」 |
| 原地没动 / 倒退 | 1 | 9 | 10 | |
| 已经越过到下一句 | 0 | 1 | 1 | |
| 未归类 | 2 | 19 | 21 | |

### 3.3 转向类：裁判到底看没看到转弯（结构性检查，从 prompt 里嵌的动作摘要直接数）

| TURN_* 判定 | 数量 | completed | unknown |
|---|---:|---:|---:|
| 这条边的动作记录里**没有**指令方向的转向命令 | 65 | 1 | **64** |
| 至少有一次指令方向的转向命令 | 38 | 12 | 26 |

也就是说：裁判看不到转向命令时 98% 说不确定；看到了也只有 32% 点头（因为它还要求视觉上「进入了转向后的那条路」）。前者是输入问题（H3），已由 `2a3dbd9` 修掉；后者是 H1/H2 混合。

### 3.4 一个 STOP_WAIT 的小陷阱

57 条 STOP_WAIT unknown 里有 4 条理由写的是「动作记录以前进结束、**没有显式的 stop 命令**」（如 id 670 第 10 跳：「the second sink clearly and at close range, but ... no explicit stop command」）。但系统设计里 STOP 是在裁判说 completed **之后**才发的，边的动作记录里永远不可能有 stop。这是 prompt 层面一个明确的、形式级的矛盾，属于 Rk-C 可以单独修的点（数量小，但每一条都是「人已到终点、发不出 STOP」）。

---

## 4. 158 次可复核判定：三方对照（实际运行）

### 4.1 总表

| 线上裁判 | 独立复核说「完成」 | 几何门 | 次数 | 怎么理解 |
|---|---|---|---:|---|
| completed | 是 | 通过 | 14 | 三方一致，可信的完成 |
| completed | 是 | **不通过** | 24 | 两个 VLM 都点头，几何说方向不对或没前进——含 469/546 的假阳性 |
| completed | 否 | — | **0** | 第二个 VLM 一次都没否决线上的 completed |
| unknown | 是 | 通过 | **17** | 最硬的「裁判太保守」证据（H1） |
| unknown | 是 | 不通过 | 24 | 第二个 VLM 觉得完成了但几何不支持，两个 VLM 至少有一个被骗 |
| unknown | 否 | 通过 | 4 | 走对了方向但两位老师都觉得还没到（多半是真没到，路走了一半） |
| unknown | 否 | 不通过 | **75** | 三方一致：确实没到 |

四个类别的隐藏几何（中位数）：

| 类别 | 次数 | 沿参考路线净前进 | 落点离参考路线 | 实际走的方向与参考方向夹角 | 走路层离选点的距离 |
|---|---:|---:|---:|---:|---:|
| completed 且复核同意 | 38 | 0.55 m | 1.01 m | 39° | 0.96 m |
| unknown 但复核说完成 | 41 | 0.65 m | 1.50 m | 34° | 0.79 m |
| unknown 且复核同意 | 79 | **0.00 m** | **1.96 m** | **108°** | 0.97 m |

第三行是关键：**裁判说「不确定」而第二位老师也同意的 79 次，落点几乎没有沿路线前进（中位数 0）、方向偏了 108°、离路线快 2 米**——这些边根本就没走对，裁判说 unknown 是对的。这部分是 H2（选点/走路层把人送错了），占 unknown 的 66%。

### 4.2 「completed」为什么不能靠第二个 VLM 查

38 次 completed，第二个 VLM 全部同意，可是：

- 24 次几何门不通过。拆开看：15 次是「选点方向 > 30° 且实际方向 > 30° 且没前进」，6 次是方向都偏但有前进，2 次只是选点偏，1 次只是没前进。
- 10 次判定时落点离参考路线 > 2 米。
- id 469 的 t3/t4/t5（FOLLOW → PASS → STOP）沿路线净前进 −1.5 / −1.7 / +0.5 米、离路线 2.1 / 1.9 / 2.6 米，最后 STOP 在离终点 16.3 米处；第二个 VLM 对这三次全部给了 completed，理由是「红地毯 / 门口 / 玻璃展板清晰可见」。

原因不难理解：这个博物馆场景（Z6MFQC）里红地毯、展柜、玻璃展板到处都是，指令说「walk past the doorway, wait by the glass info panes」，agent 走到了*某个*门口和*某块*玻璃展板旁——**从纯 RGB 看这确实「完成」了**，只是不是数据集标注的那一处。这种「相似实例」错误不是裁判的推理错，是任务本身在 RGB-only 下的歧义；再多问一个 RGB 模型也查不出来，只有隐藏几何能。

注意：「沿参考路线净前进 = 0」在终点附近会失真（投影已经到了路线末端，进度不再增加），所以 308 t9、810 t6/t7 这类在终点 1 米内的 completed 也显示 0，它们不是错的。真正可疑的是「前进为负」或「离路线 > 2 米」的那些。

### 4.3 「unknown 但其实完成了」的 41 次长什么样

按类型：ADVANCE_STRAIGHT 11、TRAVERSE_PORTAL_REGION 5、TURN_RIGHT 4、TURN_LEFT 3、EXIT_REGION 3、PASS_LANDMARK 3、STOP_WAIT 3、APPROACH_LANDMARK 3、其余 6。

按线上裁判给的理由：「还没到 / 地标还在前面」22、「证据不足」7、「没有转向命令」5、「地标不可见」4、其它 3。

其中几何门也通过的 17 次（最硬的 H1 证据），逐条：

| id | 跳 | 类型 | 沿路线前进 | 离终点 | 线上裁判理由（缩写） | 后果 |
|---:|---:|---|---:|---:|---|---|
| 94 | 0 | ADVANCE_STRAIGHT "Walk forward" | 3.22 m | 11.6 m | 要等「左边的拐角出现」 | 再探一跳 |
| 94 | 1 | 同上 | 2.38 m | 9.3 m | 同上 | **封方向+回溯** |
| 94 | 2 | 同上 | 2.31 m | 7.0 m | 同上 | **封方向+回溯** |
| 166 | 0 | ADVANCE_STRAIGHT "Go forward by the windows" | 1.39 m | 4.5 m | 窗户还没到「身侧或略后方」 | 再探一跳 |
| 11 | 0 | CROSS_SPACE | 3.53 m | 3.6 m | 还没到「地面区域的对面边界」 | 再探一跳（下一跳判完成，最终成功） |
| 308 | 1 | EXIT_REGION | 0.59 m | 5.3 m | 浴室检测框还占大半画面 | **封方向+回溯**（后来另一方向完成） |
| 526 | 2 | PASS_LANDMARK | 0.57 m | 5.2 m | 地毯「还在前视图里」 | **封方向+回溯** |
| 469 | 1 | TURN_RIGHT | 2.99 m | 11.9 m | 17 次前进、0 次右转（H3） | **封方向+回溯** |
| 546 | 8 | ENTER_REGION | 0.38 m | 3.1 m | 还没「越过门框」 | 再探一跳 |
| 781 | 2 | BETWEEN_OBJECTS | 1.31 m | 4.2 m | 「没有清楚地显示穿过吧台和桌子之间」 | **封方向+回溯** |
| 781 | 5 | 同上 | 1.58 m | 4.0 m | 同上 | **封方向+回溯**（该局 0 完成） |
| 1071 | 1 | TURN_LEFT | 1.48 m | 7.8 m | 2 次右转、0 次左转（H3，但复核认为净向左） | 再探一跳 |
| 670 | 3 | TURN_LEFT | 3.05 m | 2.8 m | 只有 1 次 15° 左转 | **封方向+回溯**（后来成功） |
| 670 | 10 | STOP_WAIT | 0.91 m | **0.6 m** | 「没有显式 stop 命令」（3.4 节陷阱） | **封方向+回溯**（下一跳才完成，STOP 在 0.12 m） |
| 824 | 1 | OTHER "along the red rug" | 2.64 m | **0.4 m** | 地毯「还在往前延伸」 | 再探一跳（之后 4 次 unknown，封满放弃，终点 2.42 m 却没发 STOP） |
| 1117 | 0 | APPROACH_LANDMARK | 1.92 m | 6.6 m | 吧台还没「正对前方」 | 再探一跳 |
| 1117 | 1 | 同上 | 2.05 m | 4.7 m | 同上 | **封方向+回溯**（该局 0 完成） |

这些有共同点：都是**「过程型」子指令**（直走、穿过、走向、沿着），裁判把「completion_cue」里的措辞（"until the corner becomes visible"、"windows alongside or slightly behind"、"opposite boundary"）当成硬条件，而这些条件在 RGB 里要么本来就模糊，要么要走很远才满足；一次 unknown 之后策略只允许再探一跳，第二次 unknown 就封方向回溯，于是 id 94 三跳沿着正确走廊前进了 8 米却被判三次 unknown、封了 5 个方向后放弃，最终 0 完成。

**41 次「其实完成了」里 25 次直接导致封方向+回溯，16 次是「再探一跳」。** 这是第 3 位说的「成功率天花板」在数据上的样子。

### 4.4 「unknown 且复核同意」的 79 次：不是裁判的错

这 79 次的几何：沿路线前进中位数 0，方向偏 108°，离路线 1.96 米。也就是说选点选偏了、或者走路走偏了，人根本不在该在的地方，两位老师都说没到。典型：id 568「出卧室」6 跳里 5 跳两位老师一致说「门框还在前面、还在卧室里」，几何上这 6 跳方向偏 94°–164°、净前进 −1～+1 米——agent 在卧室里打转，没出门；id 1139 同样 5 跳里 4 跳一致，净前进全部为 0；id 171 5 跳全部一致，方向偏 19°–118°。

还有一小类值得单独记：id 842「go straight past the piano」6 跳里 4 跳线上说「没检测到钢琴」、第二位老师却说「钢琴样的柜子已经在身后」，而几何显示其中两跳净前进 0、一跳 −10 米。也就是说两个 VLM 对「那是不是钢琴」意见相反，几何站在线上裁判这边。这类「第二位老师也不可靠」的情况提醒我们 4.1 表里「unknown 但复核说完成、几何不通过」的 24 次不能全算成裁判的错。

要修这部分，动的是选点（第 2 位方向门、候选点选择）和走路，不是裁判。

---

## 5. 最后一句（STOP_WAIT）与「差一个点头」

25 条正常结束的 episode 里，最后一段共有 25 次判定：8 次 completed（全部发了 STOP，6 次在 3 米内），17 次 unknown。17 次 unknown 里离终点 3 米内的（`stop_blocking_unknown`）有 7 次：

| id | 跳 | 离终点 | 线上裁判 | 复核 | 后来怎样 |
|---:|---:|---:|---|---|---|
| 670 | 7 | 0.53 m | unknown「还没到安全偏移」 | 完成 | 又判了 3 次 unknown（8、9、10 跳，0.9/1.6/0.6 m），第 11 跳才 completed，STOP 在 0.12 m |
| 670 | 8 | 0.91 m | unknown | 完成 | |
| 670 | 9 | 1.57 m | unknown | 不完成 | |
| 670 | 10 | 0.64 m | unknown「没有 stop 命令」 | 完成 | |
| 698 | 7 | 2.52 m | unknown「检测器没检到床」 | 不完成 | 第 9 跳 completed，STOP 在 1.91 m |
| 698 | 8 | 2.72 m | unknown | 不完成 | |
| 1117 | 2 | 2.77 m | unknown「吧台没正对前方」 | 完成 | 之后 3 次 unknown，封满放弃 |

id 670 是教科书案例：agent 在第 7 跳就已经站在离终点 0.5 米的第二个水槽旁，裁判连续 4 次说「不确定」（其中一次是要 stop 命令），靠着回溯没走丢、第 5 次才点头。这一条最后成功了，但多花了 4 跳；换一个运气差点的场景就是「封满 5 个方向放弃」——报告里 11 条崩在最后一句 STOP_WAIT 的 episode 很可能就是这种。

id 824 更可惜：第 1 跳时离终点 0.4 米（子指令是 "along the red rug"，不是最后一句），裁判说地毯还在延伸，之后 4 次 unknown，封满放弃，终点 2.42 米却因为没发 STOP 记 0 分。

---

## 6. 两次假阳性 STOP（469、546）逐跳复盘 + 修复前后对比

### id 469（Z6MFQC 博物馆，"Turn right and walk along the red carpet to your right. Walk past the doorway wait by the glass info panes."，起点离终点 16.25 m）

| 轮 | 跳 | 子指令 | 动作记录 L/R/F | 线上 | 复核 | 几何门 | 沿路线前进 | 离终点 |
|---|---:|---|---|---|---|---|---:|---:|
| 修复前 | 0 | TURN_RIGHT | 0/0/12 | unknown（0 次右转） | 完成 | ✗ | +1.9 | 15.7 |
| 修复前 | 1 | TURN_RIGHT | 0/0/17 | unknown（0 次右转） | 完成 | ✓ | +3.0 | 11.9 |
| 修复前 | 2 | TURN_RIGHT | 0/2/3 | completed | 完成 | ✗ | −0.3 | 12.2 |
| 修复前 | 3 | FOLLOW_PATH_BOUNDARY | 0/0/15 | completed | 完成 | ✗ | −1.5 | 15.1 |
| 修复前 | 4 | PASS_LANDMARK | 0/1/8 | completed | 完成 | ✗ | −1.7 | 16.1 |
| 修复前 | 5 | STOP_WAIT | 1/0/4 | completed → STOP | 完成 | ✗ | +0.5 | **16.3** |
| **修复后** | 0 | TURN_RIGHT | 0/**6**/12 | **completed**（第一次） | 完成 | ✗ | +1.9 | 15.7 |
| 修复后 | 1 | FOLLOW_PATH_BOUNDARY | 0/6/17 | completed | 完成 | ✓ | +3.0 | 11.9 |
| 修复后 | 2 | PASS_LANDMARK | 6/0/3 | unknown | 不完成 | ✗ | 0.0 | 12.0 |
| 修复后 | 3 | PASS_LANDMARK | 6/0/14 | completed | 完成 | ✗ | −2.1 | 15.0 |
| 修复后 | 4 | STOP_WAIT | 1/0/11 | completed → STOP | 完成 | ✗ | −1.8 | **17.4** |

读法：修复后第 0 句一次过，省了 2 跳和 1 次封方向（H3 修好了）；第 1 句走对了（几何门通过，前进 3 米，离路线 0.1 米）；**从第 2 句 "walk past the doorway" 开始走反**（净前进 −2.1 m），裁判和第二位老师都对「走过了一个门口」点头，最后在 17.4 米外的另一块玻璃展板前 STOP。两位 RGB 老师一致、几何全部反对——**这一条的病根在选点层选了错的门口，裁判只是没能力分辨两个长得一样的门口。**

### id 546（zsNo4H，3 段，起点离终点 4.61 m）

| 轮 | 跳 | 子指令 | L/R/F | 线上 | 复核 | 几何门 | 前进 | 离终点 |
|---|---:|---|---|---|---|---|---:|---:|
| 修复前 | 0 | TURN_RIGHT | 1/0/9 | unknown | 完成 | ✗ | 0.0 | 3.0 |
| 修复前 | 3 | TURN_RIGHT | 0/0/7 | unknown | 完成 | ✗ | +0.7 | 3.3 |
| 修复前 | 4 | TURN_RIGHT | 0/1/16 | completed | 完成 | ✗ | −2.4 | 2.3 |
| 修复前 | 8 | ENTER_REGION | 0/2/9 | unknown | 完成 | ✓ | +0.4 | 3.1 |
| 修复前 | 9 | ENTER_REGION | 0/1/13 | completed | 完成 | ✗ | +1.5 | 3.9 |
| 修复前 | 10 | STOP_WAIT | 0/0/5 | completed → STOP | 完成 | ✗ | 0.0 | **4.5**（geodesic 5.09） |
| **修复后** | 0 | TURN_RIGHT | 1/**6**/9 | **completed**（第一次） | 完成 | ✗ | 0.0 | 3.0 |
| 修复后 | 1 | ENTER_REGION | 3/0/3 | unknown | 不完成 | ✗ | 0.0 | 2.9 |
| 修复后 | 5 | ENTER_REGION | 0/8/8 | completed | 完成 | ✓ | +0.6 | 2.9 |
| 修复后 | 6 | STOP_WAIT | 3/0/10 | completed → STOP | 完成 | ✗ | +1.5 | **3.4**（geodesic 3.44） |

读法：修复后 11 跳变 7 跳、封方向从 6 次变 1 次；第 2 句进对了房间（几何门通过）；最后一句 "wait by the mirror sliding closet doors" 走到了镜面衣柜门前（两位老师都同意），但 geodesic 3.44 m 比 3.0 m 半径多了 0.44 m。这一条是**「走路层停早 + 终点判定站位偏」**的组合（第 4 位问题），裁判层面没有明显错。

---

## 7. 修复后的 smoke（2 条 9 次判定）单独看

线上 7 completed / 2 unknown；独立复核 7 次 completed 全同意，2 次 unknown 全同意；几何门通过 2/9。转向类 3 次判定全部 completed（修复前同两条的转向类 5 次判定 1 completed / 4 unknown）。样本太小不能下统计结论，但方向与主轮 3.3 节一致：**转向命令进了记录，裁判就不再卡转向句**。

---

## 8. 必须说清楚的局限

1. **主轮的转向数据本身残缺。** 主轮跑在 `2a3dbd9` 之前，边记录里没有「转到选定视图」那段旋转。独立复核喂的是同一份残缺记录 + 关键帧（关键帧从转向后开始），所以它对转向句的意见也是在缺信息下给的（复核器多次靠关键帧里的景物变化「猜」出转了弯）。能证明 H3 的只有 smoke 的 2 条。
2. **独立复核的 VLM 与线上裁判是同一个模型（`deepseek-v4-flash-vision-exp`）、看的是同一批图**，两者不独立。它 38/38 同意 completed、对 469 的假阳性一路点头，说明它作为「反驳者」的价值有限；它的真正价值是在 unknown 里找出「其实完成了」的 41 条。**「独立复核通过」不是 ground truth。**
3. 几何门是相对**一条**示范路线的 30° 判据；R2R 允许多条合理路线，所以「几何门不通过」不等于「走错」，只是「和示范不一样」。判定时的落点距离用的是到终点的 XZ 平面欧氏距离（不是测地距离），近处可靠、跨墙时会偏小；沿路线净前进在终点附近会饱和为 0（4.2 节说明）。
4. 158/498 覆盖率：75 条崩溃 episode 的 340 次判定没有几何、没有复核，只进了第 3 节的文本统计；崩溃修复（`272c223`）后重跑会把它们变成正常结束的 `directions_blocked`，届时可复核。
5. 理由分类是关键词启发式（`analyze_judge_round.py` 里的 `REASON_CATEGORIES`），5% 未归类，边界类别（「还没到」vs「地标不可见」）会有串类，只用来看大势。

---

## 9. 结论与建议（对应第 3 位「怎么修」）

**裁判的病分三块，每块的药不一样：**

| 成分 | 占比（主轮可复核的 120 次 unknown） | 证据 | 该动哪 |
|---|---|---|---|
| H3 输入不足（看不到转向） | 72/395 全量 unknown（18%）；可复核集里 5/120 | 3.3 节结构检查；smoke 修复后转向句一次过 | **已修**（`2a3dbd9`）；十 EP 回归待做 |
| H2 到达点确实不对 | 79/120（66%） | 4.4 节：净前进 0、偏 108°、离路线 2 m | 选点（第 2 位方向门、候选点）+ 走路（第 4 位），**不是裁判** |
| H1 裁判过严 | 41/120（34%），其中几何也支持的 17 次 | 4.3 节；集中在 ADVANCE / TRAVERSE / APPROACH / BETWEEN 这些过程型 form | Rk-C 形式级规则，见下 |

**建议（按 `project_rulle.md` 16.4，全部按 instruction form 抽象，不按 episode 写）：**

1. **先不要整体放松 prompt。** 66% 的 unknown 是对的，整体放松会把它们变成 469 那种假阳性，而假阳性的代价是「错误 STOP、整局归零」，比 unknown 的代价（多探一跳）大得多。
2. **STOP_WAIT 的「没有 stop 命令」矛盾**（3.4 节，4 条 + 670 t10）：判定 prompt 应明确「stop 动作由系统在 completed 之后发出，动作记录里不会有 stop」。零风险、形式级、可以单独立项立即做。
3. **过程型 form（ADVANCE_STRAIGHT / CROSS_SPACE / FOLLOW_PATH_BOUNDARY / TRAVERSE_PORTAL_REGION / APPROACH_LANDMARK）的 completion_cue 处理**：这些 cue（"until the corner becomes visible"、"windows alongside"）是拆解器写的措辞，裁判把它当成硬门槛。候选规则：对过程型 form，若边的方向与子指令一致、有持续前进、且没有越过下一句的地标，允许 completed（这正是独立复核 prompt 里 TURN+walk 那条规则的写法，可以迁移）。需要在 id 94/166/1117 这类真实 state 上单点验证，再十 EP 回归。
4. **外层策略的容忍度**：过程型 form 一次 unknown 就只剩一跳、两次就封方向，是 41 次里 25 次回溯的直接触发器。可以考虑按 form 给「on_route unknown」更多次数（现有代码已对 VERTICAL/EXIT 这样做），这是 Rk-A/策略层的改动，先记 failure taxonomy。
5. **假阳性只能靠几何兜底检查，不能靠再问一次 VLM**（4.2 节）。评测口径上，`instruction_validated_r2r_success` 继续以「几何门 + 独立复核」双重 AND 为准是对的；不要把独立复核单独当成功依据。
6. 下一次全量（修复崩溃 + 转向记录后）跑完直接用本次工具复核：`verify_round_stage_completions.py <round> --dataset ... --include-unknown --always-call-vlm`（先 `--dry-run` 看调用数）→ `analyze_judge_round.py <round> --dataset ...`，届时 75 条崩溃 episode 会进入可复核集合，STOP_WAIT 近失败的统计才完整。

---

## 附：复现命令

```bash
source local_env.sh
DATASET=/workspace/3d_wm_vln/StreamVLN/data/datasets/r2r/val_unseen/val_unseen.json.gz
ROUND=outputs/e2e_eval/20260911_001436_e2e_opennav
SMOKE=outputs/e2e_eval/20260911_113312_turn_history_smoke
python scripts/verify_round_stage_completions.py $ROUND --dataset $DATASET --include-unknown --always-call-vlm --dry-run   # 177 次
python scripts/verify_round_stage_completions.py $ROUND --dataset $DATASET --include-unknown --always-call-vlm             # 实际复核（约 8 分钟）
python scripts/verify_round_stage_completions.py $SMOKE --dataset $DATASET --include-unknown --always-call-vlm             # 11 次
python scripts/analyze_judge_round.py $ROUND --dataset $DATASET
python scripts/analyze_judge_round.py $SMOKE --dataset $DATASET
```
