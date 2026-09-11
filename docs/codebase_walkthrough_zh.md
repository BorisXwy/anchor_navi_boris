# 代码仓库算法与逻辑通俗讲解（会话整理稿）

> 来源：Claude Code 会话 `9b44572a-2bac-486b-be87-bd62b365e107`（2026-09-10 ～ 2026-09-11）的三段分析原文，逐字保存，便于随时查阅。
> 分析基于当时的代码阅读（五个子任务分模块读源码并互相核对关键数字），**没有实际运行评测**；文中引用的行号是当时快照的行号，之后的提交可能使前后有几行偏移，以文件名 + 函数名为准。
>
> 目录：
> 1. 整个仓库在干什么、主线流程、模块 0～8 逐个拆解、文件速查（主体）
> 2. 附录 A：判定模块 `_contact_sheet` 崩溃 bug 与「Round 021 的 5/10 是怎么来的」（已于 2026-09-10 修复，见 `docs/e2e_eval_reports/`）
> 3. 附录 B：地面 mask 的行范围裁剪到底裁了什么、节点为什么太近、"一个点都打不了"的真正原因（`No floor-bearing candidate` 崩溃已于 2026-09-11 在 272c223 改为正常结束 episode）
>
> **2026-09-11 补充**：模块 1～8 各自下面新增了 `N.1 扩充细节` 小节（1.1、2.1 … 8.1），按统一体例逐步讲解算法、常数、各种情况的处理、实测例子、下游用途和已知坑。会话原文一字未改；凡扩充小节与原文有出入的地方（例如 3.1 的锚点采样函数归属、4.1 的 640×480 例子、5.1 的"每停一次建节点"、6.1 的判定闸门归属、7.1 的回溯次数与 on_route 预算），均在扩充小节开头明确标注"以扩充小节为准、原文保留不改"。每个扩充小节开头的引用块都注明了哪些结论是本机实际运行核对的、哪些只是静态阅读。

---

确认了一件事：那条用 navmesh（导航网格）修复选点的分支被 `if (not rgb_only and ...)` 包着（`scripts/point_selectors.py:1586`），只在旧的 RGB-D 契约下运行；正式的 `rgb-only-v1` 配置走不到它。下面是整理后的完整讲解。

---

# 这个仓库在干什么？

**一句话**：让一个"只有一只普通彩色摄像头、没有 GPS、没有测距仪"的虚拟机器人，听懂一段英文口语指令（比如"走过厨房的桌子，在门口停下"），在一栋逼真的 3D 虚拟房子里走到指定终点，并且**自己喊"停"**。

用一个比喻：你把一个人蒙上眼睛带进陌生的房子，然后只允许他透过一根管子看照片（不能摸墙、不能数步子、没有地图），你在旁边念指令，他要自己走到终点并说"我到了"。这个仓库就是训练/搭建这样一个"人"的全部代码，以及给他打分的考场。

几个关键词解释：

- **R2R（Room-to-Room）**：一个公开的考题集。每道题 = 一栋真实扫描的房子（MP3D 场景）+ 一段人写的英文指令 + 起点 + 终点。终点 3 米以内停下算对。
- **Habitat**：一个 3D 仿真器，相当于"虚拟房子的游戏引擎"，负责渲染机器人眼睛看到的画面、执行"前进/左转/右转"。
- **VLM**：视觉语言大模型（这里用 DeepSeek 的多模态模型，通过 DMXAPI 中转站调用）。你给它图 + 文字，它用文字回答。
- **纯 RGB（RGB-only）**：这是本项目最重要的硬规则。机器人只能用彩色图片、指令原文、以及"自己发过的动作记录"来做决定。仿真器里明明有坐标、深度、地图、标准答案路线，但**统统不许看**，看了就算作弊。

---

# 主线流程（用一个例子从头走到尾）

假设指令是：**"Walk past the kitchen table and stop at the doorway."**（走过厨房桌子，在门口停下）

```
① 拆指令      "walk past the kitchen table" → 子指令1（类型：PASS_LANDMARK 经过地标）
              "stop at the doorway"        → 子指令2（类型：STOP_WAIT 到达停下）

② 环顾四周    机器人原地拍 6 张照片（每 60° 一张）+ 8 张照片（每 45° 一张，只给"判卷"用）

③ 看地面      三个分割模型投票：哪些像素是"能踩的地面"？（绿色区域）
   看物体      开放词汇检测器找"kitchen table"，画框

④ 选点        把 6 张图拼成一张"六宫格"，在绿色地面上撒几个编号白点，
              问 VLM："要走过桌子，该走向哪个白点？" → VLM 答："第 2 张图，3 号点"

⑤ 走过去      执行器盯住这个点（点跟踪），一步步发 前进/左转/右转，
              直到"脚下那片地面从画面底部消失"= 到了

⑥ 记笔记      在"记忆图"里新建一个节点（存 6 张图、检测到的物体、一个特征向量），
              画一条边（存所有动作 + 5 张关键帧）

⑦ 判卷        问 VLM："从上一个节点走到这个节点，'走过厨房桌子'完成了吗？"
              只允许回答 completed / unknown

⑧ 决定下一步  completed → 换下一条子指令，回到 ②
              unknown   → 再探一步；连续两次 unknown → 物理走回上一个可信节点，换方向
              最后一条子指令 completed → 立刻发 STOP，episode 结束

⑨ 打分（考官）  仿真器偷偷量一下：喊 STOP 的地方离终点 ≤ 3 米？是 → 成功
```

下面按模块逐个拆开讲。

---

# 模块 0：考场与"防作弊保险丝"

**文件**：`scripts/habitat_point_navigation.py`（2180 行，总入口）、`scripts/rgb_only_runtime.py`

**输入**：命令行参数（第几道题、用哪个 VLM、输出目录等）+ R2R 数据集 + MP3D 场景文件。

**输出**：`outputs/<run>/` 目录下的 `trajectory.json`（整局总结）、`vlm_calls.json`（每次问 VLM 的原话和回答）、`exploration.mp4`（录像）、`navigation_graph/`（记忆图）、`evaluation_only/`（考官偷看的几何数据，模型永远碰不到）。

**内部逻辑**：

1. 读取第 N 道题，把机器人放到题目给的起点、朝向起始方向（`habitat_point_navigation.py:772-838`）。
2. 建仿真器（`make_sim`，第 173 行）：机器人身上装了一圈摄像头，分辨率 320×240，高 1.25 米。动作只有三个：`move_forward` 走 0.22 米、`turn_left`/`turn_right` 转 15°。
3. **关键设计——`RGBOnlyPolicySimulator`**（`rgb_only_runtime.py:47`）：它像一个"套子"套在真正的仿真器外面。导航代码拿到的是套子，套子只开放两个功能：拍照（只返回 RGB 图）和执行三个动作。如果任何代码试图访问名字里带 `position`/`rotation`/`depth`/`path`/`navmesh`/`geodesic`/`collision` 的属性，套子直接抛异常让程序崩掉（第 90 行）。每个导航模块的构造函数第一行都会调 `require_rgb_only_policy_sim()` 检查"你拿到的是套子而不是原始仿真器"。

   **举例**：你写了一行 `sim.get_agent(0).state.position` 想偷看坐标，程序立刻报 `RuntimeError: privileged Habitat attribute ... is forbidden by rgb_only_v1`。这就是"不许作弊"的物理实现，而不是靠自觉。

4. 跑完后调 `audit_rgb_only_contract.audit_run()` 再扫一遍所有输出文件，发现泄露也报错。

---

# 模块 1：指令拆解（InstructionDecomposer）

**文件**：`scripts/instruction_decomposer.py`、`scripts/instruction_taxonomy.py`、`scripts/point_selection_strategies.py`

**输入**：一整句英文指令。

**输出**：一个有序列表，每项是一个 `SubInstruction`，包含：这一段的原文、地标（`landmark`）、完成线索（`completion_cue`）、该走到什么样的地面（`semantic_spatial_target`）、**类型（`form`）**、以及给选点用的策略说明。

**内部逻辑**（`instruction_decomposer.py:161-240`）：

1. 先把整句话发给 VLM，让它按"一个动作一段"拆开，并用自己的话写出每段的地标和完成标志（prompt 在 `vlm_harness.py:1997`）。
2. 然后用**一套确定性的正则规则**（`instruction_taxonomy.py:16-148`）给每一段贴"类型标签"。VLM 负责理解语义，规则负责分类——这样类型是可控的，不会因为 VLM 心情变化而变。
3. 如果 VLM 偷懒把两个动作合成一句，规则里的分句器（按 `then`、`and + 动词` 切）会再拆开。

**17 种类型**（挑常见的举例）：

| 类型 | 例句 | 意思 |
|---|---|---|
| PASS_LANDMARK | walk past the table | 经过某物 |
| APPROACH_LANDMARK | walk towards the lamp | 走向某物 |
| STOP_WAIT | stop at the doorway | 在某处停 |
| TURN_LEFT / TURN_RIGHT / TURN_AROUND | turn left | 转向 |
| ENTER_REGION / EXIT_REGION | enter the kitchen | 进/出房间 |
| TRAVERSE_PORTAL_REGION | go through the archway | 穿过门/拱 |
| VERTICAL_UP / DOWN | climb the stairs | 上下楼梯 |
| BETWEEN_OBJECTS | walk between the chairs | 从两物之间穿过 |
| CIRCUMNAVIGATE | go around the couch | 绕过某物 |
| FOLLOW_PATH_BOUNDARY | follow the hallway | 沿走廊走 |

**举例**："Walk past the kitchen table and stop at the doorway" 会拆成两段：第一段类型 `PASS_LANDMARK`，目标地面 = "地标之后、沿路线方向的空地"，禁止目标 = "地标之前/旁边的地面或桌子本身的像素"；第二段类型 `STOP_WAIT`，目标 = "满足'在门口'关系的空地"。

为什么要分类型？因为后面选点、判卷都会按类型用不同的规则（"经过"要选桌子**后面**的地面，"走向"要选桌子**前面**的地面）。项目规则第 16.4 条明确禁止"按某个 episode/某个物体名写 if"，只允许按这种抽象类型写逻辑。

## 1.1 扩充细节：把拆指令这件事从头到尾讲透

> 本节 2026-09-11 补写。所有例句都在本机用 `decompose_by_definition` / `HeuristicBackend` 实际跑过（纯 CPU、不联网），拆分结果和类型标签是真实输出，不是猜的；涉及 DeepSeek 真实回答的部分只做了静态阅读。行号以 2026-09-11 快照为准。

### 1.1.1 一句话概括 + 比喻

指令拆解就是**把一段口语化的长指令切成"一次只干一件事"的小任务清单**，每条小任务都要说清楚"到哪儿算完成"。

比喻：你妈让你"去楼下小卖部买瓶酱油，回来的时候顺便把垃圾扔了"。你脑子里会自动列成：① 下楼 ② 到小卖部 ③ 买酱油 ④ 回来路上扔垃圾 ⑤ 回家。每一步都有明确的"完成标志"（下到一楼了 / 看见小卖部招牌了 / 手里拿到酱油了……）。指令拆解模块干的就是这个"在脑子里列清单"的活。

为什么非拆不可？因为后面的选点模块（模块 3）一次只能回答"下一步往哪走"，判卷模块（模块 6）一次只能回答"这一小步完成了没"。你不能把整句"走过厨房桌子在门口停下"一股脑丢给它们——它们需要一条条来。

### 1.1.2 输入是什么，输出是什么

**输入**：一个字符串，就是 R2R 数据集里那句英文指令。比如 `"Walk past the kitchen table and stop at the doorway."`

**输出**：一个 Python 列表，按顺序排好，每一项是一个 `SubInstruction` 对象（`instruction_decomposer.py:12-34`）。它有 14 个字段，逐个用人话解释：

| 字段 | 人话 | 谁填的 | 例子（"walk past the kitchen table"） |
|---|---|---|---|
| `sub_instruction_id` | 第几条（从 0 数） | 代码按顺序编号 | 0 |
| `navigation_instruction` | 这一小段的原文 | VLM 或分句器 | "Walk past the kitchen table" |
| `landmark` | 这一段围绕哪个东西 | VLM（无 VLM 时 = 原文） | "kitchen table" |
| `completion_cue` | 完成的信号 | VLM（无 VLM 时 = 规则表的 arrival） | "the table is behind you" |
| `semantic_spatial_target` | 该走到什么样的地面上 | VLM 优先，规则表补空 | "free floor beyond the table" |
| `spatial_relation` | 目标和地标是什么关系 | VLM（无 VLM 时 = 类型名小写） | "beyond" |
| `visual_arrival_evidence` | 画面里看到什么算到了 | VLM 优先，规则表补空 | "table appears behind / out of front view" |
| `forbidden_target` | 绝对不能选的地方 | VLM 优先，规则表补空 | "floor before the table, table surface" |
| `source_clause` | 规则分句器切出来的那个小句 | 规则 | "Walk past the kitchen table" |
| **`form`** | **类型标签（最重要）** | 规则（正则） | `PASS_LANDMARK` |
| `secondary_forms` | 次要类型（一句话同时命中多个正则时剩下的） | 规则 | `[]` |
| `definition` | 这种类型的一句话定义 | 规则表 | "continue until the referenced landmark lies behind the agent" |
| `point_selection_strategy` | 一份文字版"选点作业指导书" | 规则表 `point_selection_strategies.py` | 见 1.1.9 |
| `metadata` | 杂项：拆分来源、类型是否被改过等 | 代码 | `{}` 或 `{"compound_part_index": 0, ...}` |

要点：**文字性的描述（地标、目标、完成线索）由 VLM 来写，类型标签由确定性规则来贴**。VLM 擅长理解语义，但它同一句话今天答 A 明天答 B；正则规则死板，但稳定可控。项目把"不能变的"交给规则、"需要理解的"交给 VLM。

### 1.1.3 三条不同的进入路线

代码里其实有三种拿到子指令列表的方式（`habitat_point_navigation.py:1020-1051`）：

1. **正式路线（有 VLM）**：`InstructionDecomposer(vlm_harness).decompose(instruction)`。先问 VLM，再套规则。下面 1.1.4～1.1.7 讲的就是这条。
2. **无 VLM 路线**：`InstructionDecomposer(None)`，跳过 VLM，直接用规则分句 + 贴标签，所有文字字段全部用规则表的模板文字。只在测试或某些离线脚本里用。
3. **读冻结产物**：命令行给了 `--decomposition_artifact <json>`，直接从文件里读以前拆好的结果，不再问 VLM。用途是"十 EP 冻结实验"里保证每一轮拆分结果一模一样，不让 VLM 的随机性干扰对比。这条路线**不经过 `decompose()`**，只经过 `SubInstruction.from_mapping()`，所以 1.1.8 讲的"迁移规则"主要就是给它准备的。

### 1.1.4 第一步：问 VLM（`vlm_harness.py:2007-2056`）

**发什么**：一段纯文字 prompt（**不带任何图片**——这一步机器人还没睁眼，只看指令），开头是标记 `STAGE_DECOMPOSITION`，正文大意是：

> 你是 R2R 导航规划器。把指令拆成有序的、视觉上能落地的阶段。**每个阶段的终点必须是一块能用地面像素表示的地方**，不能只是"转弯"或"出去"这种动作。
> 落地规则：
> - "出房间" = 门口**外面**那块地，要远到摄像头已经跨过门框；
> - "进房间" = 门口**里面**那块地；
> - "经过某物" = 那个东西**后面**的地；
> - "左转/右转" = 转过去之后新走廊里的地，**绝不能为了让机器人转身而选一面墙**；
> - "停在 X 附近" = X 旁边保持安全距离的地，不是 X 本身的像素；
> - "直走" = 走廊尽头远处的地。
> 每个阶段写清：地标、精确的目标地面、空间关系、到达时画面里的证据、禁区。保留所有转弯、动作、地标、停止条件。不要编指令里没有的东西，也不要写"前面某处"这种空话。

为什么反复强调"终点必须是地面"？因为下游执行器（模块 4）只会"盯着一个地面像素走过去"，它不会执行"转 90°"这种抽象动作。如果 VLM 拆出一段"turn left"却不说要走到左边哪块地，后面就没法执行。

**要求回什么**：严格的 JSON（`STAGE_SCHEMA`，第 1183 行）：`{"stages": [ {stage_id, navigation_instruction, landmark, completion_cue, semantic_spatial_target, spatial_relation, visual_arrival_evidence, forbidden_target}, ... ]}`。后端强制 `response_format=json_object`、温度 0，所以同一句话重复问，答案基本一致（但不保证完全一致，这就是要有"冻结产物"路线的原因）。

**收到之后怎么检查**（`validate` 函数，第 2032-2054 行）：

- `stages` 必须是非空列表，否则报错；
- 每一条的 `navigation_instruction` 去掉空白后不能为空；
- `semantic_spatial_target`、`spatial_relation`、`visual_arrival_evidence`、`forbidden_target` 四个字段**一个都不能空**（`landmark` 和 `completion_cue` 允许空，会变成空字符串）；
- 通过的话把 `stage_id` 重新按 0、1、2… 编号（不信 VLM 自己编的号）。

**检查不过怎么办**（`_call`，第 1950-1993 行）：把错误原因接在 prompt 后面（"Previous response was invalid: xxx. Return corrected JSON only."）再问一次，默认最多重试 2 次（`--vlm-retries 2`，即总共 3 次机会）。3 次都不行就抛 `RuntimeError("VLM harness exhausted retries for decompose_instruction")`，整个 episode 直接失败。如果是 401/402/403 这类"钥匙不对"的错误则不重试，立刻抛 `VLMProviderFatalError`。

**每次问答都会存档**：`<output-dir>/vlm_calls.json` 里 `task="decompose_instruction"` 那条记录了原始 prompt、原始回答、用了多少 token、请求了几次、HTTP 状态码。想知道"VLM 到底是怎么拆的"，先看这里。

**HeuristicBackend（假 VLM，只给测试用）怎么拆**（第 1039-1050 行）：它不懂语义，只是按句号/问号/感叹号和单词 `then` 切一刀，每段 `landmark="unspecified"`，其它字段填固定的套话。所以用 heuristic 后端时，"Enter the kitchen and walk along the counter, then stop." 会先被它切成两段（`and` 不切），然后靠下面 1.1.7 情况 1 的规则再补一刀。

### 1.1.5 第二步：规则分句器怎么切（`instruction_taxonomy.py:9-13, 160-161`）

不管 VLM 拆没拆，每一段文字都会再过一遍确定性分句器 `split_instruction`。它只在以下四种位置切：

| 切在哪 | 例子 | 说明 |
|---|---|---|
| 句号 / 分号 / 感叹号 / 问号 | "Turn left. Stop at the top" → 2 段 | 最普通的一刀 |
| `then` 或 `and then` | "walk in then stop" → 2 段 | 时间顺序词 |
| `and` + **紧跟一个导航动词** | "past the table **and stop** at" → 切 | 动词表：go walk turn take head continue proceed stop wait exit enter pass cross veer bear make keep follow move travel climb descend leave get |
| 逗号 + **紧跟一个导航动词**（可带 then） | "Exit the room**, walk** down the hall" → 切 | 同上 |

**关键设计：`and` 后面不是导航动词就不切。** "walk between the chairs **and** the table" 里 `and` 后面是 `the`，不切，所以"两把椅子和桌子"不会被劈成两个任务。这条规则也是 1.1.7 情况 1 能安全拆 VLM 结果的前提——只在明确的"连接词 + 新动作"处切。

切完后每段去掉首尾空格和逗号；空段丢掉。

### 1.1.6 第三步：给每一小句贴类型标签（`instruction_taxonomy.py:16-157, 164-187`）

每一小句拿 **18 个正则**逐个试（表 `FORM_DEFINITIONS`，17 种具体类型 + 匹配不上时的 `OTHER`）。正则匹配的是关键词，下面把每种类型的"触发词"翻成人话：

| 类型 | 触发词（大意） | 备注 |
|---|---|---|
| `EXIT_REGION` | exit / leave / **outside**，或 go/walk/head/get/move + out | 注意 "outside" 一词就触发 |
| `ENTER_REGION` | enter，或 go/walk/head/move/get + into | |
| `TURN_LEFT` | turn/veer/bear/make/take/go/head 后 **18 个字符内**出现 left | 距离限制防止 "walk through the door on the left" 被当成左转 |
| `TURN_RIGHT` | 同上，right | |
| `TURN_AROUND` | turn/veer/… 后 20 字符内 around / u-turn / 180 | |
| `TURN_TO_LANDMARK` | turn to / turn towards + 一个名词，且不是 left/right/around/back | "turn to the painting" |
| `VERTICAL_UP` | climb / ascend / upstairs，或 up + the stairs/steps/staircase | |
| `VERTICAL_DOWN` | descend / downstairs，或 down + the stairs/… | |
| `PASS_LANDMARK` | pass / past / passed / passing | |
| `CIRCUMNAVIGATE` | around / circle，或 right/left side of | |
| `CROSS_SPACE` | across / cross / crossing | |
| `BETWEEN_OBJECTS` | between | |
| `SELECT_PORTAL` | first/second/third/last/next/nearest + door/doorway/opening，或 door on the left/right | 挑门 |
| `TRAVERSE_PORTAL_REGION` | through | |
| `FOLLOW_PATH_BOUNDARY` | follow / along，或 keep + wall/hall/path/corridor | |
| `ADVANCE_STRAIGHT` | straight / forward / ahead / keep walking / go down the hall(way) | |
| `APPROACH_LANDMARK` | towards / until / near / beside / next to，或 go/walk/head/move + to | |
| `STOP_WAIT` | stop / wait / stand / remain | |

**一句话命中多个怎么办？** 很常见。比如 "Go straight past the pool" 同时命中 `ADVANCE_STRAIGHT`（straight）和 `PASS_LANDMARK`（past）。代码按一张固定的**优先级表** `PRIMARY_PRIORITY` 排序，排第一的当 `form`，其余塞进 `secondary_forms`：

```
STOP_WAIT > VERTICAL_UP > VERTICAL_DOWN > EXIT_REGION > ENTER_REGION
> TURN_AROUND > TURN_LEFT > TURN_RIGHT > TURN_TO_LANDMARK > SELECT_PORTAL
> PASS_LANDMARK > CIRCUMNAVIGATE > BETWEEN_OBJECTS > CROSS_SPACE
> TRAVERSE_PORTAL_REGION > FOLLOW_PATH_BOUNDARY > ADVANCE_STRAIGHT > APPROACH_LANDMARK
```

排序的思路是"**终点越明确、越硬的排前面，越泛泛的排后面**"：

- "停"最硬——只要句子里有 stop，这段的意义就是停下来；
- 上下楼、出入房间是"换了一个空间"，完成与否非常明确；
- 转弯次之；
- "经过/绕过/穿过"是有具体地标的动作；
- "沿着走/直走/走向"最软，几乎每句都能沾边，所以垫底。

实际跑出来的例子（本机真实输出）：

| 原句 | form | secondary_forms | 为什么 |
|---|---|---|---|
| Go straight past the pool | PASS_LANDMARK | [ADVANCE_STRAIGHT] | past 比 straight 优先 |
| head towards the outside door | **EXIT_REGION** | [APPROACH_LANDMARK] | "outside" 触发了出门；towards 触发走向；出门优先 |
| Walk through the door on the left | SELECT_PORTAL | [TRAVERSE_PORTAL_REGION] | "door on the left" 是挑门；没被当成左转（距离超 18 字符） |
| Turn right at the end of the hall | TURN_RIGHT | [] | |
| Turn to the painting | TURN_TO_LANDMARK | [] | |
| wait by the window | STOP_WAIT | [] | |
| The room is big | **OTHER** | [] | 没有任何导航动词 |
| Start in the bedroom | **OTHER** | [] | "起点描述"，不是动作 |

**一个都没命中** → `form="OTHER"`，定义字段用一套通用文字（"unclassified navigation or observation clause"），选点策略用 `STRATEGIES["OTHER"]`（"靠 VLM 逐句理解"）。

贴完标签，每一小句就变成一个字典：`source_clause`、`form`、`secondary_forms`、`definition`、`semantic_spatial_target`/`visual_arrival_evidence`/`forbidden_target`（从规则表抄）、`navigation_instruction`/`landmark`（都 = 原句）、`completion_cue`（= 规则表的 arrival）、`spatial_relation`（= 类型名小写）、`point_selection_strategy`。

### 1.1.7 第四步：把 VLM 的结果和规则的结果"合体"（`instruction_decomposer.py:161-240`）

这是模块的核心逻辑。对 VLM 返回的**每一段**，取它的 `navigation_instruction` 再跑一遍分句器 + 贴标签，得到 `typed`（可能 1 段也可能多段），然后按下面三种情况处理：

**情况 1：VLM 把两个动作捏成了一段（`typed` 里有 ≥2 个非 OTHER 的小句，且当前是有 VLM 的路线）→ 强行拆开**（第 187-214 行）

例：VLM 返回一段 "Enter the kitchen and walk along the counter"。规则切出 `Enter the kitchen`（ENTER_REGION）和 `walk along the counter`（FOLLOW_PATH_BOUNDARY），两个都是可动作的，于是变成**两条**子指令。

拆开之后每条怎么填？**完全用规则表的模板**，VLM 写的那份地标/目标文字被**整段丢弃**（因为 VLM 写的是针对合并后那句的，分开后对不上）：

- `navigation_instruction` = `landmark` = 小句原文；
- `completion_cue` = 规则表的 arrival 文字；
- `metadata` 记下 `vlm_parent_raw_index`（来自 VLM 第几段）、`vlm_parent_navigation_instruction`（原来那句完整的话）、`compound_action_count`（拆成了几个）、`compound_part_index`（我是第几个）。事后查 `trajectory.json` 时一看 metadata 就知道这条是被强拆出来的。

为什么要拆？注释里写得很清楚：一段里有两个动作，只能贴一个主类型，那么第二个动作的终点就没了——"进厨房沿柜台走"如果只按 ENTER_REGION 判，进门就算完成，柜台那段直接消失。

为什么无 VLM 路线不走这一步？因为无 VLM 时 `raw` 本身就是分句器切出来的，已经是一段一个动作，不需要再拆。

**情况 2：`typed` 里恰好 1 个非 OTHER 小句（或者多个小句但只有 1 个是非 OTHER）→ 保留 VLM 原话，贴上第一个非 OTHER 的标签**（第 215-236 行）

典型例子："Start in the bedroom and head towards the outside door"。分句器切成 `Start in the bedroom`（OTHER）和 `head towards the outside door`（EXIT_REGION）。如果傻乎乎取第一个，整段就是 OTHER，下游什么都做不了。所以代码专门挑**第一个非 OTHER 的**当类型来源。结果：`form=EXIT_REGION`、`secondary_forms=[APPROACH_LANDMARK]`，但 `navigation_instruction` 还是 VLM 写的完整原句（前面那句"起点描述"留着给 VLM 看，对理解上下文有帮助）。

字段合并规则："**VLM 写的优先，规则只补空**"：

- `source_clause`、`form`、`secondary_forms`、`definition`、`point_selection_strategy` 这五个**永远**由规则覆盖（VLM 根本不产出这些）；
- `semantic_spatial_target`、`visual_arrival_evidence`、`forbidden_target` 用 `setdefault`——VLM 有就用 VLM 的，VLM 没给才用规则模板。（实际上 validate 已经保证 VLM 路线下这三项非空，所以这条补空主要对无 VLM 路线有意义。）

**情况 3：`typed` 里一个非 OTHER 都没有 → 类型 OTHER**

例如 VLM 返回一段 "You are now facing the fireplace"。规则找不到任何导航动词，`typed_actionable` 为空，退回 `typed[0]`，`form="OTHER"`。这条子指令会照常进入队列，选点时用 `STRATEGIES["OTHER"]`——也就是完全靠 VLM 看图理解这句话。

**编号与截断**：三种情况处理完后 `sub_instruction_id = len(sub_instructions)`，保证 0、1、2… 连续不跳号（VLM 一段拆成两条时也连续）。最后如果调用方传了 `limit`（对应命令行 `--targets N`），只取前 N 条。

**两种"根本没法开始"的报错**：指令为空字符串 → `ValueError("instruction must be non-empty")`；VLM/规则返回空列表 → `RuntimeError("Instruction decomposition produced no sub-instructions")`。两者都直接终止 episode。

### 1.1.8 第五步：`from_mapping` 里藏着两条"事后纠错"规则（`instruction_decomposer.py:36-136`）

不管走哪条路线，最后每条字典都要经过 `SubInstruction.from_mapping()` 变成对象。它除了填默认值，还做两件"迁移"工作。这两条规则的注释都反复强调"只看语法和文字，不看 episode 编号、不看图、不看示范路径、不看运行结果"——这是对项目规则 16.4（禁止补丁式优化）的自证。

**规则 A：把陈旧的 OTHER 重新贴标签**（第 46-64 行）

背景：早期的冻结产物文件是用旧版正则生成的，那时候"Start in the room and head out the door"这种带起点描述的句子会被整段标成 OTHER（因为旧代码取 `typed[0]`）。后来正则修好了，但冻结文件不能改。于是 `from_mapping` 在**读入时**发现 `form` 是 `OTHER`/`UNCLASSIFIED`/空，就用当前规则对 `navigation_instruction` 重新贴一次标签，拿第一个非 OTHER 的结果覆盖 `source_clause/form/secondary_forms/definition/point_selection_strategy` 五个字段。VLM 写的地标、目标、完成线索**一个字不动**。`metadata["form_normalization"] = {"from": "OTHER", "to": "EXIT_REGION", "policy": "first_actionable_taxonomy_clause"}` 留下痕迹。

本机实测：`{"navigation_instruction": "Start in the room and head out the door", "form": "OTHER"}` → `form=EXIT_REGION`。

如果重贴之后还是没有非 OTHER 的小句（纯描述句），就维持 OTHER 不动。

**规则 B：复合"转弯 + 上/下楼"的终点优先**（第 65-107 行）

背景：冻结文件有时故意把 "turn left and head up the stairs" 留成**一条**，因为在标注者看来这是一个连贯动作（正常的 VLM 路线会被情况 1 拆成两条，所以这条规则主要服务冻结产物）。分句器会把这句切成 `Turn left` 和 `head up the stairs` 两小句，旧产物里记录的主类型是第一小句的 `TURN_LEFT`。问题是：判卷模块看到 `TURN_LEFT`，机器人一转过去踩上第一级台阶它就可能判 completed——但指令真正的终点是**楼梯顶**。

所以 `from_mapping` 做一次检查：把 `completion_cue`、`semantic_spatial_target`、`visual_arrival_evidence` 三个字段拼在一起当"终点描述"；如果规则分句里出现了 `VERTICAL_UP` **并且**终点描述里含 top / upper landing / higher level，就把主类型改成 `VERTICAL_UP`，原来的 `TURN_LEFT` 降级进 `secondary_forms`（作为"路线约束"保留，选点时仍会参考）；`VERTICAL_DOWN` 对应 bottom / lower landing / lower level。`definition` 和 `point_selection_strategy` 换成楼梯那套；`metadata["terminal_form_normalization"]` 记录改动。

本机实测：`{"navigation_instruction": "Turn left and head up the stairs", "form": "TURN_LEFT", "completion_cue": "reach the top landing", ...}` → `form=VERTICAL_UP`，`secondary_forms=["TURN_LEFT"]`。

注意触发条件是"**两个都满足**"：只有 "up the stairs" 但完成线索里没说 top，不改；完成线索说了 top 但句子里没有上楼词，也不改。

**默认值**：字段缺失时的兜底文字——`landmark="unspecified"`、`semantic_spatial_target="walkable floor at the target"`、`spatial_relation="at the target"`、`visual_arrival_evidence="target transition is complete"`、`forbidden_target="non-walkable regions"`、`form="UNCLASSIFIED"`。凡是不认识的键一律塞进 `metadata`，不丢。`stage_id` 是 `sub_instruction_id` 的旧别名，两个名字都认，输出给选点器时会通过 `to_stage_dict()` 把 `stage_id` 再补回去（选点器还在用旧名字）。

### 1.1.9 `point_selection_strategy` 到底是什么

`point_selection_strategies.py` 是一张 18 行的**文字表**（每种类型一份），每份写着这种类型该用什么感知、检测什么、候选地面在哪、约束是什么、怎么排序、到达标准是什么、失败了怎么兜底。例如 `PASS_LANDMARK`：

> 检测：pass/past 后面那个名词短语；候选：路线投影在地标**之后**的连通地面；约束：地标必须在候选点后方、离物体有安全侧向距离、不能只是在旁边或前面；排序：检测文字匹配度 → 超过地标的余量 → 路线连续性 → 间隙；到达：地标已经跑到机器人身后；兜底：没检测到就不瞎猜"后面"，先保持地标可见往前走再重新检测。

它**不是可执行代码**，也**不进入任何 VLM prompt**（在线模块 `vlm_harness.py`、`point_selectors.py`、`semantic_point_strategy.py` 里没有一处读它）：它只是随子指令写进 `trajectory.json` 和冻结产物，供人工复核和契约测试（`tests/test_module_contracts.py:1080` 检查每条子指令都带 `rank`）。真正按类型起作用的是 `form`/`secondary_forms`（见 1.1.11）。表里有些条目提到 "depth"，那是原作者写策略描述时的说法；在 `rgb-only-v1` 契约下没有任何代码能读到深度（模块 0 的套子会直接抛错），这些字只是描述性文字。

### 1.1.10 各种情况处理一览（速查表）

| 遇到的情况 | 系统怎么办 | 在哪 |
|---|---|---|
| 指令是空字符串 | `ValueError`，episode 不开始 | `decompose` 第 163 行 |
| VLM 返回不是合法 JSON / `stages` 为空 / 某段缺四个必填字段 | 把错误贴在 prompt 后重问，最多 3 次；都失败抛 `RuntimeError` | `_call` 第 1983-1993 行 |
| API key 无效（401/402/403） | 不重试，立刻 `VLMProviderFatalError` | `_call` 第 1974 行 |
| VLM 一段含两个动作（"enter … and walk along …"） | 按规则拆成两条，文字字段用规则模板，metadata 记来源 | 1.1.7 情况 1 |
| VLM 一段 = 起点描述 + 一个动作 | 保留原句，类型取第一个非 OTHER | 1.1.7 情况 2 |
| VLM 一段纯描述、无动作 | 类型 OTHER，选点靠 VLM 逐句理解 | 1.1.7 情况 3 |
| 一句命中多个正则 | 按优先级表选主类型，其余进 `secondary_forms` | 1.1.6 |
| `and` 后面不是导航动词（"between the chairs and the table"） | 不切 | 1.1.5 |
| 冻结文件里的 OTHER | 读入时按当前规则重贴标签，VLM 文字不动 | 1.1.8 规则 A |
| 冻结文件里 "turn left and head up the stairs" 标成 TURN_LEFT 且完成线索提到 top | 主类型改 VERTICAL_UP，TURN_LEFT 降为次要 | 1.1.8 规则 B |
| 命令行 `--targets N` | 只保留前 N 条 | `decompose` 第 240 行 |
| 命令行 `--start-sub-instruction-index K` | 从第 K 条开始执行（前面的当作已完成） | `habitat_point_navigation.py:1053` |
| 走 `--decomposition_artifact` 冻结产物 | 完全不问 VLM，只过 `from_mapping` | `habitat_point_navigation.py:1021-1048` |

### 1.1.11 拆出来的东西下游谁在用

- **`form` / `secondary_forms`**：选点收窄地面（`semantic_point_strategy.py:143`，比如 PASS 只留地标后面的地）、检测词抽取（`semantic_detector.py:446`，`TRAVERSE_PORTAL_REGION` 在次要类型里也会触发"门"的检测）、选点 prompt 里按类型加规则（`vlm_harness.py:3147, 3292, 3531`）、判卷的确定性闸门（`instruction_completion_judge.py:246, 427, 704`，比如 `TURN_TO_LANDMARK` 要求正前方对着地标）、外层状态机判断"下一条是不是 STOP_WAIT"（`instruction_sequence_exploration.py:603`，最后一条 STOP 完成就主动喊停）。
- **`landmark`**：从中抽名词给开放词汇检测器（`vlm_harness.py:3235, 5116, 5248`）。
- **`navigation_instruction` / `semantic_spatial_target` / `forbidden_target` / `completion_cue`**：原样写进选点和判卷的 prompt 文字。

所以类型标签一旦贴错，影响是链式的：地面收窄错 → 选点错 → 判卷闸门错。这也是为什么类型交给可回归测试的正则而不是 VLM。

### 1.1.12 已知的边角与坑（静态阅读结论，未专门验证）

- 正则是按**关键词**匹配，不懂语法。"the **outside** door"（外面那扇门）会因为 outside 一词被标成 EXIT_REGION；"go **around** the corner" 会被标成 CIRCUMNAVIGATE（绕行）。多数情况下下游 VLM 看原句还能自我纠正，但闸门是按类型走的。
- `TURN_LEFT/RIGHT` 的"18 个字符内"是经验值：动词和 left/right 之间夹的字太多就不算转弯。好处是 "door on the left" 不误判，代价是 "turn at the next hallway on the left" 这种长句可能漏掉。
- 情况 1 强拆时 VLM 写的地标/目标被整段丢弃、换成模板文字，这意味着这些子指令的 `semantic_spatial_target` 比较泛（"free walkable floor immediately inside the destination region"），没有 VLM 那种具体描述。
- HeuristicBackend 只按句号和 then 切，不按 and 切，所以用它跑测试时"A and B"要靠情况 1 补拆；这和 DeepSeek 真实路径的段落划分未必一致，冒烟通过不代表真实拆分行为一样。

---

# 模块 2：感知——哪儿是地？东西在哪？

**文件**：`scripts/ground_segmentation_backends.py`、`scripts/semantic_detector.py`、`scripts/semantic_point_strategy.py`

## 2a. 地面分割（dense-majority）

**输入**：一张 RGB 图。**输出**：一张同尺寸的黑白掩码（True = 可以踩的地面）。

**内部逻辑**（`ground_segmentation_backends.py:256-308`）：

- 用三个现成的语义分割模型（OneFormer、Mask2Former、SegFormer-B5，都在 ADE20K 数据集上训练）各自给每个像素分类。
- 类别名里含 floor / rug / carpet / ground / stairs / step / landing / road 等词的都算"地面"（第 107-119 行，按名字匹配，不是硬编码 id）。
- **三个模型投票，至少 2 票才算地面**（`votes >= 2`）。楼梯单独投票，然后从地面里减去楼梯。
- 没有任何形态学后处理，就是原始投票结果。

**为什么三模型投票？** 单个模型会把桌面、床面误认为地板；三个不同架构同时犯同一个错的概率小得多。

**举例**：一条铺着地毯的走廊。三个模型分别把走廊标为 `floor`、地毯标为 `rug`（或直接 `floor`），两个都算地面词，所以输出一张覆盖"整条走廊 + 地毯"的连通掩码。

替代后端 `grounded-sam` / `dino-sam`（用文字提示 "floor, carpet, rug, stair tread..." 让检测器框出来再用 SAM 抠图，然后 9×9 形态学闭运算）只用于消融实验。

## 2b. 开放词汇物体检测（DINO + SAM）

**输入**：RGB 图 + 一组文字（从子指令里抽出来的名词，如 `"kitchen table"`、`"doorway"`；`extract_detection_queries`，`semantic_detector.py:433`）。
**输出**：一组 `SemanticDetection`：标签、置信度、框、像素级掩码。

**内部逻辑**：Grounding DINO 按文字找框（阈值 box 0.28 / text 0.22），SAM 把框变成精确掩码。

**用在哪**：只给选点模块用（画在给 VLM 看的图上，并作为文字证据），以及存进记忆节点里当"这个地方有什么"的记录。

## 2c. 按类型收窄候选地面（semantic_point_strategy）

**输入**：子指令类型 + 地面掩码 + 检测结果。**输出**：收窄后的候选地面掩码。

**举例**：类型是 `PASS_LANDMARK`、检测到桌子框，就只保留"桌子框之后"的地面（`:220-225`）；类型是 `BETWEEN_OBJECTS`，只保留两个框之间的缝隙；再把所有检测到的物体膨胀一圈从地面里挖掉（防止选到贴着桌腿的点）。如果收窄后剩得太少就退回全部地面。

## 2.1 扩充细节：感知模块从头到尾讲透

> 本节 2026-09-11 补写。地面词匹配、检测词抽取、按类型收窄这三块纯逻辑都在本机用合成数据实际跑过（纯 CPU、不加载模型），表格里的像素数字是真实输出；三个分割模型和 DINO/SAM 的推理行为只做了静态阅读。行号以 2026-09-11 快照为准。像素换算统一按正式配置的 320×240 画面。

### 2.1.1 一句话概括 + 比喻

感知模块回答两个问题：**"画面里哪些像素是能踩的地？"** 和 **"指令里提到的那个东西在画面里的哪儿？"** 它自己不做任何决定，只是把答案（两张黑白掩码 + 一列带框的检测结果）交给选点模块。

比喻：你蒙着眼被带进一间屋子，只能透过管子看照片。感知模块就是两个助手：一个拿绿色荧光笔把照片上"能走的地板"全涂绿（地面分割）；另一个拿红笔按你念的词（"厨房桌子"、"门口"）在照片上圈出来（开放词汇检测）。然后还有第三个助手，根据这一段指令的类型（"经过桌子"），用橡皮把绿色里"不该去"的部分擦掉（桌子前面的地擦掉，只留桌子后面的）。三个助手都不说话，只交图。

### 2.1.2 先说照片是怎么拍的

正式配置下机器人**不转身拍照**，而是身上装了一圈固定摄像头同时拍（`habitat_point_navigation.py:173-206`）：

- 6 个彩色相机，朝向 0°、60°、120°、180°、240°、300°，每个 320×240 像素，水平视角 90°，装在离地 1.25 m 处、水平不俯仰。`observe_six_rgb()`（`point_selectors.py:80`）一次取回 6 张。因为每张覆盖 90°、相邻只差 60°，相邻两张有 30° 重叠，六张拼起来是完整 360°。
- 另外 8 个相机每 45° 一张（`completion_rgb_*`），**只给判卷模块（模块 6）用**，感知/选点不看。
- 深度相机在仿真器里也装了，但 RGB-only 套子（模块 0）不让任何在线代码取到；`DenseMajorityGroundSegmenter.batch()` 第一行就是 `del depths`（`ground_segmentation_backends.py:279`），`extract()` 收到 depth 直接抛错（`navigation_graph_memory.py:121`）。

平地上像素行与距离的换算（1.25 m 高、焦距 160 px、地平线在第 120 行）：第 120 行 = 无穷远，第 163 行 ≈ 4.6 m，第 206 行 ≈ 2.3 m，最底行 239 ≈ 1.67 m。**比 1.67 m 更近的地面相机根本看不见。**后面很多"0.86h"、"0.72h" 之类的常数都要用这张表来理解。

### 2.1.3 地面分割 dense-majority 逐步（`ground_segmentation_backends.py:107-308`）

**第一步：三个模型各自给每个像素分类。** 三个都是在 ADE20K 数据集（150 个室内外类别）上训练好的现成语义分割网络，直接从本地 HuggingFace 缓存加载（`local_files_only=True`，第 144-157 行），不做任何微调：

| 名字 | 权重 | 架构差异 |
|---|---|---|
| `oneformer_ade20k` | `shi-labs/oneformer_ade20k_swin_tiny` | 通用分割，要显式告诉它 `task_inputs=["semantic"]` |
| `mask2former_ade20k` | `facebook/mask2former-swin-small-ade-semantic` | 掩码分类式 |
| `segformer_ade20k` | `nvidia/segformer-b5-finetuned-ade-640-640` | 纯 Transformer 编码器 + 轻量解码；输出的 logit 网格是固定尺寸，代码用双线性插值缩回 320×240 再取 argmax（第 193-198 行） |

6 张图作为一个 batch 一起送进每个模型（`predict_batch`，第 172 行），三个模型顺序跑。每个模型的输出是一张 `class_map`（每个像素一个类别编号）。

**第二步：哪些类别算"地面"？按类别名里的单词判断，不按编号。** `_is_ground_label`（第 107-119 行）把类别名拆成单词，只要含以下任一词就算：floor / rug / carpet / ground / earth / land / soil / stair(s) / stairway / staircase / step(s) / landing / sidewalk / path / road。用"整词"匹配，所以 `background` 不会因为含 `ground` 四个字母而误判。本机实测：

| ADE20K 类别名 | 算地面？ | 算楼梯？ |
|---|---|---|
| floor, flooring | 是 | 否 |
| rug / carpet | 是 | 否 |
| stairs, steps / stairway, staircase / step, stair | 是 | **是** |
| earth, ground / land, ground, soil / sidewalk, pavement / road, route / path | 是 | 否 |
| bed / table / grass / field / background | 否 | 否 |

为什么名单里有 road、sidewalk、earth？因为 MP3D 是真实房子扫描，有院子、露台、车库，这些地方模型会给 "road"/"earth" 这类标签，物理上照样能走。为什么没有 grass、field？原作者没放；后果是草坪不算地面，这是设计选择不是 bug。

模型加载时如果一个"地面词"都对不上（`ground_ids` 为空）直接 `RuntimeError`（第 168 行），防止静默输出全黑。

**第三步：投票。** 三张 `class_map` 各生成一张"是否地面"的布尔图，逐像素相加得到 0～3 票，`votes >= 2` 的像素算地面（第 285-286 行）。楼梯**单独再投一次**：三个模型各自"是否楼梯词"的票数 ≥ 2 **且**已经是地面的像素算楼梯（第 287-292 行）。

**第四步：打包输出。** 返回 `(majority, detections)`：`majority` 是最终地面掩码（**含楼梯**）；`detections` 是最多两条"伪检测"——标签 `"floor"`（地面减楼梯）和 `"stairs"`（楼梯），各带一个矩形框、一张掩码、一个分数（掩码内平均票数 ÷ 3，比如全部 3 票 = 1.0，全是 2 票 = 0.667）。这两条伪检测的作用是让后面的代码能"按标签把楼梯从地面里减掉"（见 2.1.8）。

**第五步：没有任何后处理。** 不做开闭运算、不填洞、不去小块。投票结果是什么就是什么。`strict_ground_mask = True`（第 269 行）这个类属性告诉下游"我是严格模式，你只能减像素不能加像素"。

**为什么三模型投票而不是一个模型？** 单个模型常把桌面、床面、浴缸底这种"平的浅色面"误判成 floor；三个不同架构同时犯同一种错的概率小得多。代价是显存和时间三倍。

**具体例子：**

- 铺地毯的走廊：模型 A 说 floor、模型 B 说 rug、模型 C 说 carpet → 三个词都在名单里 → 3 票 → 地面。
- 一张白色床单：模型 A 说 bed、B 说 bed、C 说 floor → 1 票 → 不是地面。这就是投票的价值。
- 泳池水面：三个都说 water 或 swimming pool → 0 票 → 不是地面（这是对的，但也是附录 B 里 episode 6 "前方没有任何地面"的来源）。
- 楼梯：三个都说 stairs → 是地面且是楼梯 → 出现在 `majority` 里，同时单列一条 `"stairs"` 伪检测。

**CUDA 强制**：构造时 `device` 不是 `cuda*` 或 CUDA 不可用直接 `ValueError`（第 129、272 行），符合项目规则"视觉模型不得静默回退 CPU"。`close()` 把模型删掉并 `torch.cuda.empty_cache()`（第 218-221 行），给后面的大模型腾显存。

### 2.1.4 消融用的另外两个地面后端（`--floor-segmenter grounded-sam|dino-sam`）

这两个走完全不同的思路：**不是"给每个像素分类"，而是"用文字去找"**（`DinoSamFloorSegmenter`，`semantic_detector.py:393-424`）：

1. 把 8 个词 `floor / wooden floor / tile floor / carpet / rug / walkable ground / stair tread / stair landing` 拼成一句 "floor. wooden floor. tile floor. ..." 交给 Grounding DINO 找框；
2. 每个框交给 SAM 抠成掩码；
3. 所有掩码取并集；
4. 做一次 9×9 的形态学**闭运算**（先膨胀后腐蚀，把小缝小洞填上，第 416-419 行）。

`strict_ground_mask` 在这两个后端上不存在（`getattr` 默认 False），所以下游会启用一堆"地面为空时的兜底先验"（`rgb_lower_floor_prior`，见 2.1.8）——这些兜底在正式的 dense-majority 模式下**全部关闭**。两者的差别不只是模型，还有整套兜底逻辑，比较结果时要记住这一点。

### 2.1.5 开放词汇检测 DINO + SAM 逐步（`semantic_detector.py:86-210`）

正式配置 `--semantic-detector dino-sam`（默认值，`habitat_point_navigation.py:690`）用的是 HuggingFace 上的轻量版：`IDEA-Research/grounding-dino-tiny` + `facebook/sam-vit-base`。`grounded-sam` 是本机 `/workspace/Navi-Agent` 里原版 Swin-T + SAM ViT-H 权重的兼容拼写，行为一样、模型更大。

**输入**：一张 RGB + 一列词（怎么来的见 2.1.6）。**输出**：一列 `SemanticDetection`，每条有 `label`（匹配到的词）、`score`（0～1）、`box_xyxy`（左上右下像素）、`mask`（像素级布尔图）。

**第一阶段，DINO 找框**（`GroundingDinoBoxDetector.detect_boxes`，第 98-139 行）：

1. 把词列表拼成 "kitchen table. doorway." 这种句子（每个词后面加句点，这是 Grounding DINO 的输入约定）；
2. 模型输出一堆候选框，每个框对每个词有一个相似度；
3. 两个阈值：框的最高相似度 ≥ **0.28**（`box_threshold`）才留，相似度 ≥ **0.22**（`text_threshold`）的词才算"这个框对应这个词"。两个都能用命令行 `--detector-box-threshold / --detector-text-threshold` 改；
4. **去重（NMS）**：按分数从高到低，后面的框如果和已保留的某个框重叠面积比（IoU）≥ 0.72 就丢掉（第 122-136 行）。这一步故意放在 SAM 之前，因为 SAM 贵。

**第二阶段，SAM 抠图**（`SamBoxSegmenter.segment_boxes`，第 151-169 行）：每个框喂给 SAM，SAM 会给 3 个候选掩码和各自的自评 IoU，取自评最高的那个（第 165-168 行）。SAM 只看框，不看词。

**边界检查**：SAM 返回的掩码数和框数不一致直接 `RuntimeError`（第 194-197 行）。

**记录去敏**：`SemanticDetection` 有个 `median_depth_m` 字段，是旧 RGB-D 契约留下的；在线路径全部调 `rgb_prompt_record()`（第 64-68 行），它会把这个字段删掉再写进 prompt 和日志。这是"深度字段存在但永远为 None 且不外传"的兼容写法。

**如果一个词都没有**（比如子指令 "Turn left" 抽不出名词），`detect_boxes` 直接返回空列表，不调模型。

### 2.1.6 检测词从哪里来（`extract_detection_queries`，`semantic_detector.py:433-616`）

这是一个 180 行的纯文字处理函数，输入是一条子指令字典，输出是一列词。它的设计原则写在注释里：**只做词法处理，不看图、不看 episode、不看示范路径**；宁可少给也不给整句话（Grounding DINO 面对"walk towards the sink"这种句子表现很差，面对 "sink" 就好得多）。逐步：

1. **楼梯特例**：类型是 `VERTICAL_UP/DOWN` → 直接返回 `["stairs", "staircase", "landing"]`，后面全跳过。
2. **门类型先加门词**：类型是 `EXIT_REGION / ENTER_REGION / SELECT_PORTAL / TRAVERSE_PORTAL_REGION`，或次要类型里有 `TRAVERSE_PORTAL_REGION` → 先放 `doorway / open door / hallway opening`。理由：这类指令的下一个物理动作是穿门，只找"厨房"找不到门在哪。
3. **加 landmark**：VLM 写的 `landmark` 字段，≤ 8 个词且不是 none/unspecified/unknown 才加。
4. **从完成线索里抓"下一个区域"**：把 `completion_cue + semantic_spatial_target + visual_arrival_evidence` 拼起来，找 `towards / into / inside / beyond / through / leads to / leading to` 后面 1～4 个词。例如 "heading towards the kitchen area" → `kitchen area`。
5. **从原句里抓关系宾语**：`past/near/beside/towards/by + X`、`between X and Y`、`around / left side of / right side of + X`。
6. **清洗**：去掉 "walk towards the X" 这种动作前缀只留 X；在 then/until/and stop/and to the left 处截断；去掉冠词；**还含动作词的整条丢弃**；超过 12 个词的丢弃；去重。
7. **扩展**：每个词再派生几个变体——去掉 "corner of / center of / far side of" 这类前缀（"corner of the bar" → 加 "bar"）；去掉 "with… / leading to…" 后缀；小同义词表（couch↔sofa、rug↔carpet、entryway/entrance→doorway、staircase/steps→stairs）；多词短语再加**最后一个词**当"头名词"（"kitchen table" → 加 "table"），但 room/area/hall/hallway/corridor 结尾的不拆（"living room" 不会变成 "room"）；短语里含 doorway/door/opening/entrance/portal/archway 的把这个词单独加上；"leads to / into X" 的把 X 单独加上。
8. **manel 特例**：R2R 数据里有把 mantel（壁炉架）拼成 manel 的，代码同时给 mantel、panel、fireplace mantel 三个候选，让后面的视觉去分辨。这是唯一一处针对具体拼写的处理，注释里专门解释了它是词法层面、与场景无关。

本机实测（`landmark` 用 VLM 常见的写法，其余字段用规则模板文字）：

| 子指令 / landmark | 类型 | 抽出的词 |
|---|---|---|
| "stop at the doorway" / doorway | STOP_WAIT | `doorway` |
| "head up the stairs" | VERTICAL_UP | `stairs, staircase, landing` |
| "Go around the couch" / couch | CIRCUMNAVIGATE | `couch, sofa` + 两个杂词（见下） |
| "stop near the corner of the bar" / corner of the bar | STOP_WAIT | `corner of the bar, bar` |
| "Walk through the doorway on the left leading to living room" | SELECT_PORTAL | `doorway, open door, door, hallway opening, opening, doorway on the left leading to living room, doorway on the left, left, living room, portal` + 杂词 |
| "stop next to the manel" / manel | STOP_WAIT | `mantel, panel, fireplace mantel` |
| "Turn left" / unspecified | TURN_LEFT | **空**（不调检测器） |

**已观察到的副作用**：第 4 步的正则会把规则模板文字里的 "beyond its portal"、"obstacle agent clears the" 也当成短语抓出来，再经第 7 步派生出 `the`、`is`、`enters` 这种杂词。本机实测 "Go around the couch" 得到 `['couch', 'sofa', 'obstacle agent clears the', 'the']`。当 VLM 写的完成线索是正常句子时杂词少一些，但不为零。杂词的影响是多几个几乎匹配不上的查询词（DINO 对 "the" 一般不会给高分框），代价主要是一点推理时间；这是静态阅读加合成输入观察到的，没有在真实 episode 上统计过杂词造成了多少误检。

### 2.1.7 按类型收窄地面 `constrain_floor_candidates`（`semantic_point_strategy.py:97-281`）逐步

**输入**：子指令（主要看 `form`）、地面掩码、`depth=None`（在线一律 None）、检测列表、可选的"物体掩码"。**输出**：收窄后的候选掩码 + 一条 `record` 说明用了哪条规则、有没有退回。

**第 0 步：先把家具脚下的地擦掉**（第 106-129 行）。如果传入了 `object_mask`（所有检测到的家具/门/墙的掩码并集，怎么来的见 2.1.8），先把它膨胀 7×7 像素（约 3 像素安全边），从地面里减掉。但有个**合理性检查**：物体掩码占画面 > 60%，或减完之后地面剩不到 20% → 认为这个掩码不靠谱（比如 DINO 把整个"bathroom"框成一个物体），**不减**，只记录。本机实测：给一张全画面的物体掩码，`accepted_for_floor_subtraction=False`，地面一个像素没少。

**第 1 步：这个类型需要检测吗？** `DETECTION_GROUNDED_FORMS`（第 10-14 行）列了 12 种需要的类型。不在里面的（`TURN_LEFT / TURN_RIGHT / TURN_AROUND / CROSS_SPACE / FOLLOW_PATH_BOUNDARY / ADVANCE_STRAIGHT / OTHER`）直接返回第 0 步的结果，`mode="small_seg_safe_floor"`。也就是说**"左转"、"直走"这类指令根本不用检测结果收窄地面**，靠 VLM 看图。

**第 2 步：需要检测但一个都没检测到** → 原样返回，`fallback="no_matching_detection"`。

**第 3 步：按类型画一个矩形（或楔形）候选区。** 下表用 320×240 画面、地面 = 第 110 行以下全部、桌子框 (120,100)–(200,150)（宽 80）、门框 (140,60)–(180,160) 做的真实计算：

| 类型 | 规则（人话） | 代码里的矩形 | 实测结果 |
|---|---|---|---|
| `BETWEEN_OBJECTS`（≥2 个检测） | 取分数最高的一个，再取和它水平距离最远的一个；两框之间的缝隙、从较高框顶到画面底 | 缝隙 `[left.x1, right.x0]`；缝隙 ≤ 0 时取两框中心 ±10% 画面宽 | 两把椅子 (60-100) 和 (220-260) → 候选 x∈[100,219] |
| 门类（EXIT/ENTER/SELECT/TRAVERSE，或 PASS 且次要类型含门） | **只认标签含 door/doorway/opening/hallway/portal 的检测**，分数再高的床/沙发也不能冒充门；没有门检测 → `fallback="no_portal_detection"` 退回全部地面 | 门框内缩 12%、从框的垂直中点到框底 +16% 画面高；框贴左/右边 5% 以内时横向拉宽到 15%–85%；再把地面向上膨胀 31×9 后与"门框 ±20% 宽、0.48h 以下"的楔形相交并入 | 桌子 + 门 → 选中 `doorway`，候选 x∈[132,187], y∈[115,197]；只有桌子 → 退回全部 41600 像素 |
| `VERTICAL_UP/DOWN` | 楼梯检测掩码膨胀 17×17，并上框到框底 +12% 高 | | |
| `PASS_LANDMARK` | 地标框左右各**外扩 45% 框宽**、从框顶到画面底 | `[x0-0.45w, y0, x1+0.45w, h]` | x∈[84,235]，桌子本身被挖掉 |
| `CIRCUMNAVIGATE` | 看原句有 left / right 决定哪一侧；框外侧留 12% 画面宽的间隙，再取一个框宽的走廊；都没写就两侧都要 | 左：`[x0-0.12W-w, y0, x0-0.12W, h]` | "left side of the table" → x∈[2,81] |
| `TURN_TO_LANDMARK` | 地标方位上的整条地面带，左右外扩 55%，上沿比框顶还高 5% | | x∈[76,243] |
| `APPROACH_LANDMARK` / `STOP_WAIT` | 地标**近侧**：左右外扩 65%，从**框底往上 8% 画面高**到画面底 | `[x0-0.65w, y1-0.08h, x1+0.65w, h]` | x∈[68,251], y∈[131,239] |

对比 PASS 和 APPROACH 就能看懂"类型决定方向"：同一张桌子，PASS 的候选从桌子**顶部**（y=100，画面里更远处）开始往下都算，意思是"桌子后面到脚下这一条通道都行，VLM 再挑远的"；APPROACH/STOP 只从桌子**底部附近**（y=131）开始，意思是"只能停在桌子这一侧"。（代码里 PASS 分支还有 `depth >= object_depth + 0.25` 这种深度判断，但 `object_depth` 在线永远是 None，那几行是死代码。）

**第 4 步：和地面相交，再挖掉物体本身**（第 254-269 行）。候选区 `&= base`；然后把这一步用到的那几个检测（`selected`）的掩码膨胀 9×9 从候选里减掉——**但掩码占画面 > 60% 的不减**（比如 "bathroom" 这种区域级检测，减了就没地了），记进 `ignored_region_masks`。

**第 5 步：太少就退回**（第 277-280 行）。候选剩下 < 24 像素 → `fallback="relation_mask_too_small"`，返回第 0 步的全部安全地面。

### 2.1.8 每个视图上的接线：感知结果是怎么一步步变成"这张图的候选地面"的（`point_selectors.py:1034-1449`）

`select_ground_target` 对 6 张图各调一次 `build_candidate`。下面只讲其中属于感知的部分（锚点采样、VLM 问答属于模块 3）。**顺序很重要**，这是正式 strict 模式下的实际流程：

```
① mask, ground_detections = dense-majority 结果（含 floor/stairs 伪检测）
② detections = DINO+SAM.detect(rgb, 检测词)            ← 没词就跳过
③ object_mask = detections 里"算障碍"的掩码并集
④ base = targetable_ground_mask(mask)                   ← 裁到内圈带
⑤ 指令不是上下楼 → base &= ~stairs 伪检测掩码
⑥ [非 strict 模式才有] 地面空/太小时用 rgb_lower_floor_prior 兜底 —— strict 下全部跳过
⑦ target, record = constrain_floor_candidates(stage, base, None, relation_detections, object_mask)
⑧ PASS_LANDMARK：把原始 mask 的 0.72h–0.96h、10%–90% 中央下带补回来（挖掉物体）
⑨ STOP_WAIT：只留 0.62h 以下 + 补回原始 mask 0.72h 以下中央带
⑩ target &= mask                                        ← strict 铁律：最后必须落在原始投票掩码里
⑪ select_ground_point(target) 选一个代表像素；选不出 → 这张图 point=None
```

逐条解释：

- **③ 哪些检测算"障碍"**（第 1149-1164 行）：标签含 floor/ground/walkable/carpet/rug/hallway/corridor/intersection/opening/landing 的是"路"，不算障碍；**但**标签同时含 door/doorway/portal/wall/panel 的算障碍。所以 "hallway" 不挖，"doorway" 挖（门板不能踩），"table"、"couch" 挖。
- **④ 内圈带**（`targetable_ground_mask`，第 113-122 行）：清掉上 42%（第 100 行以上，平地上在地平线之上）、下 14%（第 206 行以下，≈2.3 m 以内）、左右各 10%（32 列）。裁空了退回整张 mask。详细讨论见附录 B。
- **⑤ 减楼梯**（第 1177-1191 行）："走过桌子"这种指令不该走上旁边的楼梯，所以把 `"stairs"` 伪检测的掩码减掉；`VERTICAL_UP/DOWN` 保留。`ground_mask_source` 会记成 `dense_majority_without_non_route_stairs`。
- **PASS 的弱检测门槛**（第 1133-1142 行）：类型是 `PASS_LANDMARK` 时，分数 < 0.40 的检测**不参与收窄地面**（只作为"可能看到了"的文字证据给 VLM）。原因：DINO 低分匹配特别爱把墙板、门板认成地标，拿它来画"地标后面"的矩形会把机器人引向墙。
- **⑥ 为什么 strict 下没有兜底**：`rgb_lower_floor_prior` 是一块固定的矩形（0.54h–0.84h、8%–92%），在 grounded-sam 模式下地面为空时当"假地面"用。dense-majority 的契约是"语义只能减像素不能加像素"，所以所有 `if not strict_ground` 包着的兜底全部不执行。**结果：三模型都不投地面的视图，在正式配置下就是没有候选。**
- **⑧ PASS 下带补回**（第 1306-1339 行）：绕一张房间中央的桌子时，"桌子后面"的矩形可能全落在桌子上或墙上，唯一能绕过去的通道在画面下沿。所以把原始 mask 的 0.72h–0.96h（≈1.7–3.7 m）中央带补回来。补回的是**原始投票掩码**的像素，不违反"只减不加"。
- **⑨ STOP 近带**（第 1348-1385 行）："停在 X 旁边"必须停在 X 的这一侧，不能穿过去。所以只留 0.62h 以下（第 149 行以下，≈6.8 m 以内），再把原始 mask 的 0.72h 以下（≈3.7 m 以内）中央带补回来（这一带被 ④ 的下 14% 裁掉了一部分，而近侧停车点恰恰在那儿）。
- **⑩ 最后一道锁**（第 1447-1449 行）：无论中间做了什么，`target &= mask` 再选一次点。注释原话："every numbered VLM anchor and every downstream point cluster is provably on the original 2/3 majority mask"。

**跨视图策略**（`apply_cross_view_semantic_policy`，`semantic_point_strategy.py:21-76`；调用在 `point_selectors.py:1511-1518`）：6 张图各自处理完后还有一道"横向比较"。如果类型需要检测，而其中某几张图真的用检测收窄出了候选（`mode` 不是 floor_only），那么剩下那些"只有地面、没检测到东西"的图：

- `hard_detection_gate`（默认）：直接 `point=None`，不给 VLM 看；
- `soft_detection_evidence`：保留，但在记录里标注"别的视图有检测证据"。

`STOP_WAIT / BETWEEN_OBJECTS / ENTER_REGION / TRAVERSE_PORTAL_REGION` 四种类型**强制用 soft**（第 1514-1517 行），理由：一个房间可能有好几块地毯/沙发，检测器分数最高的那个未必是指令说的那个，硬砍会把正确方向砍掉。其它类型按选点 prompt 版本决定：v3～v35 全部在 soft 名单里（`vlm_harness.py:1886-1927`），只有 v1/v2 这两个最早的版本还是 hard。命令行默认版本是 `v10_approach_relation_router`，所以**正式配置下实际生效的是 soft**——"只有地面的视图"会保留给 VLM，附带一句"别的视图有检测证据，请比较"的提示。

### 2.1.9 记忆节点里的检测（`navigation_graph_memory.py:20-145`）

每次执行器停下建节点时，感知模块还要再跑一遍 DINO+SAM，但目的不同：不是为了选点，是为了给节点写一份"这里有什么"的清单，供判卷和回溯用。

- 词表固定 14 个：doorway / open door / hallway / corridor / stairs / stair landing / table / chair / sofa / bed / cabinet / counter / rug / carpet，再加当前子指令的 `landmark`（≤ 8 词）。
- 6 张图各跑一次，每张按分数保留前 12 条。
- 输出 `labels`（去重的标签列表）和 `label_scores`（每个标签的最高分），写进 `navigation_graph.json` 的节点里。记录用 `rgb_prompt_record()`，没有深度。
- 这一步**不用 dense-majority**——注释明确说"dense-majority is deliberately floor-only and is never reused here"（`habitat_point_navigation.py:1237-1238`）。

### 2.1.10 各种情况处理一览

| 情况 | 系统怎么办 | 在哪 |
|---|---|---|
| `--device cpu` 或无 CUDA | 构造时 `ValueError`，不静默回退 | `ground_segmentation_backends.py:129, 272` |
| 模型类别表里没有任何地面词 | `RuntimeError` | 第 168 行 |
| 三模型只有 1 个说是地面 | 不算地面 | `votes >= 2` |
| 三模型 ≥2 个说是楼梯 | 算地面 + 单列 `"stairs"` 伪检测；非楼梯指令时被减掉 | 第 287-294 行、`point_selectors.py:1177` |
| 水面 / 玻璃 / 草坪 | 不算地面（没有对应词） | `_is_ground_label` |
| 某视图三模型都不投地面（strict） | 该视图无候选，**没有任何兜底** | `point_selectors.py:1222` 的 `not strict_ground` 条件 |
| 子指令抽不出检测词（"Turn left"） | 不调 DINO，`detections=[]` | `semantic_detector.py:101` |
| 类型不需要检测（左转/直走/沿走廊） | 地面只减家具安全边，不按检测收窄 | `constrain_floor_candidates` 第 130 行 |
| 需要检测但没检测到 | 退回全部安全地面，`fallback=no_matching_detection` | 第 134 行 |
| 门类指令只检测到床/沙发、没检测到门 | 不拿床当门，退回全部地面 `no_portal_detection` | 第 192-195 行 |
| 门框贴着画面左/右边 | 横向拉宽到 15%–85% 再画楔形 | 第 188-191 行 |
| PASS 的检测分数 < 0.40 | 不用于收窄，只作文字证据 | `point_selectors.py:1133` |
| 检测掩码占画面 > 60% | 不当障碍挖、不当物体减，只当语义证据 | `semantic_point_strategy.py:118, 259` |
| 物体掩码减完地面剩 < 20% | 不减 | 第 118 行 |
| 收窄后 < 24 像素 | 退回全部安全地面，`relation_mask_too_small` | 第 277 行 |
| 内圈带裁空 | 退回整张 mask | `targetable_ground_mask` |
| 别的视图有检测、这张只有地面 | 代码默认值是硬砍，但 v3～v35 的选点 prompt（含默认 v10）都切成软保留，只加提示不砍视图 | `apply_cross_view_semantic_policy` |
| 有人传了 depth | `del depths` / 抛错 | `ground_segmentation_backends.py:279`、`navigation_graph_memory.py:121` |

### 2.1.11 下游谁在用

- 地面掩码 `mask`：选点模块画绿色叠加、在上面撒编号锚点（模块 3）；执行器把跟踪点吸附到地面上（模块 4）。
- 收窄后的 `target_mask` 和代表点 `point`：只给选点模块决定"这张图能不能给 VLM 看、锚点撒在哪"。
- 检测结果 `detections`：画彩色框给 VLM 看、写成文字证据进 prompt（模块 3）；节点语义清单（模块 5）；判卷时做"上一节点看得见桌子、现在看不见"的差分（模块 6）。
- `record` / `strategy_application`：全部写进 `trajectory.json` 每个候选视图的记录里，事后审计用。

### 2.1.12 已知的边角与坑（静态阅读结论）

- **正式模式没有任何"地面为空"的出路。** 所有 `rgb_lower_floor_prior` 兜底都在 `if not strict_ground` 里。这是有意设计（只减不加），但代价是泳池边、大玻璃前、纯草坪这类场景可能六张图全空，直接触发附录 B 讲的 "No floor-bearing candidate"（现已改为正常结束 episode）。
- **检测词抽取会产生杂词**（见 2.1.6），尤其是文字字段用规则模板时。没有统计过这会带来多少误检。
- `constrain_floor_candidates` 的矩形全是**画面比例**（45% 框宽、8% 画面高之类），没有物理米数概念；远处的小桌子和近处的大桌子用同一套比例。
- `_is_ground_label` 名单是静态的：没有 grass/field/deck/patio/mat/platform，这些表面在 MP3D 户外区域会被当成不可走。
- DINO 的 NMS 阈值 0.72 是"重叠 72% 以上才去重"，所以同一张桌子可能以 "kitchen table" 和 "table" 两个标签各留一个框（它们是同一个框时才会被去掉一个）；记忆节点每视图前 12 条里会有这种重复。
- `SemanticDetection.median_depth_m`、`_depth_valid`、`constrain_floor_candidates` 里的深度分支都是旧契约残留的死代码，在线路径永远走不到；读代码时不要被它们误导以为用了深度。

---

# 模块 3：选点——问 VLM"往哪走"

**文件**：`scripts/point_selectors.py`（`InstructionVLMPointSelector`，第 2063 行；`choose_view`，第 961 行）、`scripts/vlm_harness.py`（`select_ground_target`，第 2308 行；各版本 prompt）

**输入**：6 张（或 8 张）环视 RGB 图 + 当前子指令 + 地面掩码 + 检测结果 + "被拉黑的方向"（前面失败过的）。
**输出**：一个 `(view_index, pixel_x, pixel_y)`——"第几张图的哪个像素"，外加 VLM 的理由。

**内部逻辑**：

1. **拼图**：把 6 张图排成 3×2 的"六宫格"（`_point_selection_contact_sheet`，`vlm_harness.py:2088`）。给 VLM 发**两张**：一张干净原图，一张标注图——地面涂绿、检测到的物体画彩色框、绿地上撒**编号的白色圆点**（"锚点"）。
2. **锚点怎么撒**：在地面掩码上做"最远点采样"，并偏向掩码内部（远离边缘），保证每个点都稳稳落在地上（`point_selectors.py:179-206`）。
3. **问法**：prompt 说"先在原图里认出地标/房间/门；选一张含有目标的图；**只能从编号白点里挑一个**；如果指令说 past/through，要挑地标**之后**的地面而不是之前的"；末尾还附一句"禁止使用深度/3D/navmesh 信息"。
4. **回答格式**（`POINT_SCHEMA`，第 1195 行）：`{"view_index": 2, "anchor_index": 3, "reason": "..."}`。
5. **落实到像素**：直接查表 `anchors[view][3]` 得到像素。因为 VLM 只能选编号，它**物理上不可能**指到墙上。如果换了一个会返回自由坐标的 VLM 后端，还有一道保险：把坐标吸附到最近的地面像素（第 1727-1744 行）。
6. 对"经过/直走"这类前进型指令，会把与来路相反 ≥90° 的视图排除，避免掉头。

**Prompt 版本**：所有 prompt 都在 `vlm_harness.py`，选点 prompt 从 `v1_baseline` 迭代到 `v35`，完成判定从 `v1` 到 `v24`，老版本一律保留不删（`docs/vlm_point_selection_optimization.md` 只追加不改）。最近一轮（Round 021）冻结用的是选点 V31 / 判定 V24 / 八视图 V28。

**VLM 后端**（`DeepSeekBackend`，第 834 行）：图片转 JPEG → base64 → OpenAI 兼容格式发到 DMXAPI；强制 JSON 输出、温度 0、关闭思考模式（否则每次 60 秒）；408/429/5xx 退避重试 3 次，401/402/403 直接终止。`HeuristicBackend` 是一个不联网的假 VLM（永远选第一张允许的图的 0 号点），只用于跑测试。

**举例**：子指令"walk past the kitchen table"。检测器在第 2 张图框出桌子；六宫格上第 2 张图桌子后面的地面有白点 3 号。VLM 回答 `{"view_index": 2, "anchor_index": 3, "reason": "floor beyond the table along the corridor"}`。选点模块输出：视图 2、像素 (230, 175)。

## 3.1 扩充细节：选点模块从头到尾讲透

> 本节写于 2026-09-11，在 `anchor_navi` conda 环境（`source local_env.sh`）里用合成掩码/角度实际跑过下面标注"实测"的算法片段（方向排除锥角度判定、最远点式锚点采样），未加载任何 GPU 模型、未联网调 VLM。其余内容（VLM prompt 拼装、`DeepSeekBackend` 请求细节、各 prompt 版本分支）为通读 `scripts/point_selectors.py` 和 `scripts/vlm_harness.py` 源码后的静态分析，行号以本次快照为准。正式在线路径只讨论 `rgb_only_v1` 契约；旧的 `legacy_rgbd_geometry`（依赖 navmesh/深度的选点后处理）作为历史代码一并提及，但明确标出"这段在 RGB-only 模式下整段跳过"。

### 3.1.1 一句话概括 + 比喻

如果说模块 2（感知）是"把地面涂成绿色、把桌子椅子框出来"，模块 3 就是"把这张涂过色的照片拿给一个既看不见深度、也不能瞎指的向导（VLM），逼它像做选择题一样，在绿色地面上**编号的几个白点里挑一个**，而不是让它随便在图上点一个坐标"。这个"只能选编号、不能瞎点"的设计，是整个模块最关键的安全网：VLM 的回答格式压根不允许它给出墙上的某个像素坐标，因为白点从一开始就只撒在地面掩码内部。

### 3.1.2 输入输出

- 输入：`PointSelectionRequest`（`point_selectors.py:2004-2037`）——`sim`（RGB-only 包装句柄，读取会在越权属性上报错）、当前朝向 `yaw`、来路朝向 `back_yaw`、已探索过的"被拉黑方向" `blocked_yaws`、当前子指令 `stage`、历史动作 `previous_action_history`；`rgb_only_v1` 下 `position`/`reference_path` 必须为 `None`，否则 `choose_view` 直接 `ValueError`（`point_selectors.py:1005-1009`）。
- 输出：`PointSelectionResult(chosen, candidates)`——`chosen` 是被选中那一张视图的完整候选字典（含最终像素 `point`、朝向 `yaw`、掩码等），`candidates` 是全部 6（或 8）张视图的候选，供 `trajectory.json`/审计使用。

### 3.1.3 逐视图候选构建：`choose_view` → `build_candidate`（`point_selectors.py:961-1499`）

这是模块 2 的 `constrain_floor_candidates` 之外，把"这一张图能不能被选"这件事钉死的地方。按执行顺序：

1. **反向排除锥先定宽度**：默认来路方向 ±50° 是排除区（"来路"太窄的话掉头会看起来像合法路线）。但如果当前子指令是"经过/直走/沿边界走"（`PASS_LANDMARK`/`ADVANCE_STRAIGHT`/`FOLLOW_PATH_BOUNDARY`）且不是刚从一次转弯的边继承来的（`turn_carryover.active` 为假），这个排除锥会按 `route_backtrack_exclusion` 重新算（通常更宽，防止侧后方 70°-80° 的射线把机器人送回上一个房间）；反过来，如果正处在"转弯后继承"的状态，反向排除直接归零（`elif bool(turn_carryover.get("active")): backtrack_exclusion = 0.0`）——因为转弯之后合法的走廊本来就可能落在上一条边的"背后"。
2. **拍 6 张（或 8 张）RGB**：`rgb_only_v1` 下只有 RGB，没有深度数组（`depths = [None] * len(rgbs)`）。
3. 对每张图调用内部函数 `build_candidate`，依次做：
   - **宽而均匀的"假地面"剔除**：如果某个检测到的地面候选面积占比 ≥70% 且下方 20% 区域支持度不比上方 20% 高多少（`lower_support <= top_support * 1.15`），判定为"墙/门板被误当成地面"，整张剔掉（只有非严格模式，即 grounded-sam/dino-sam 消融后端才会触发，dense-majority 严格模式不做这个二次修正）。
   - **算两个排除角度**：`backtrack_delta`（与来路夹角）、`blocked_delta`（与所有"被拉黑方向"里最近一个的夹角）。**实测**：用与源码一致的角度函数验证过，夹角 <50° 触发排除（正好卡在 50° 边界时会因为浮点误差偶尔判成"排除"，但真实朝向都是 60° 的整数倍，不会撞在这条边界上）。`blocked_yaws` 命中的排除是 `hard_excluded`（硬性，绝不重开），单纯来路排除是软性的 `excluded`（后面在特定条件下可以重新打开）。
   - **调检测器**：只有当前子指令的类型属于 `extract_detection_queries` 能提取出查询词的情况才调用语义检测器（模块 2 已讲）。
   - **"软重开"来路/拉黑方向的正切光线**：对 `STOP_WAIT`/`EXIT_REGION`/`ENTER_REGION`/`SELECT_PORTAL`/`TRAVERSE_PORTAL_REGION` 这类"终点关系"子指令，如果这张图确实有检测器证据（分数 ≥0.20 for STOP_WAIT，否则 ≥0.28）、且夹角已经跳出了 25° 的"硬核心区"，即使原本因为来路排除被判 `excluded`，也会被重新打开（`semantic_backtrack_override = True`）。直觉：房间尽头一个真正被检测到的目标物，哪怕方向上有点像"往回走"，也不该被一刀切掉。
   - **弱检测降级**：对 `PASS_LANDMARK`，分数 <0.40 的检测不参与地面收窄（避免一个不太靠谱的噪声框把整条走廊地面切歪），但仍作为"弱证据"记录进 prompt。
   - **构造"禁区物体掩码"**：把检测到的物体（排除"floor/hallway/opening"这类路线类词）做成 `object_mask`，后面从地面里挖掉。
   - **楼梯清理**：非垂直类子指令会把 `ground_detections` 里带 stair/step/landing 标签的部分从地面掩码里删掉（避免直走指令被带上侧边楼梯）。
   - **RGB 兜底先验**（仅非严格模式）：当收窄后地面（或其"下半部中心区"）完全为空，且子指令属于 portal 类/转弯/PASS，用固定比例的"画面下半部中央矩形"（`rgb_lower_floor_prior`）顶上，保证探测器一时失灵不会让整张图直接判死。
   - **调用模块 2 的 `constrain_floor_candidates`** 拿到关系约束地面 `target_mask`。
   - **几种"把边缘带子救回来"的补丁**（都是形式级、非按场景的规则）：`STOP_WAIT` 只保留画面下 38%（`y ≥ 0.62h`）的近侧地面带，防止选点选到landmark"那一边"导致执行器直接穿过去；严格模式下 `PASS_LANDMARK`/`CIRCUMNAVIGATE` 等会从原始 2/3 多数票地面里再挖一次下半部中心区，只为兜住"桌子挡住大半张图、可走的窄边被关系约束切没了"的情况。
   - **最后再选一次代表点** `select_ground_point(target_mask)` 只用来给这张视图打分（不是最终发给 VLM 的点）。
   - **严格模式的最终收口**：dense-majority 模式下，任何补丁之后都要再和原始 2/3 多数票掩码做一次交集，保证"语义可以删地面像素，但永远不能凭空加出地面像素"这条物理契约不被破坏。
   - **打分**：`score = 地面像素占比×8 + 代表点位置项`，`excluded` 扣 10 分，`hard_excluded` 扣 100 分（这个分数只用于排序展示，真正决定"能不能被 VLM 选"的是下一步的 `allowed` 列表，不是这个分数）。
4. **跨视图 soft/hard 政策**（模块 2.1 已讲过公式）：这里补一个模块 3 独有的细节——`STOP_WAIT`/`BETWEEN_OBJECTS`/`ENTER_REGION`/`TRAVERSE_PORTAL_REGION` 这四种形式**无论 CLI 传的策略是什么，都被强制改成 `soft_detection_evidence`**（`point_selectors.py:1514-1517`）：即使某个视图检测到了目标，其它视图的纯地面候选依然保留在候选池里，只是被标注"另一个视图有证据"。

### 3.1.4 把候选交给 VLM：两种拼图、锚点怎么撒、Prompt 怎么写（`vlm_harness.py:2099-2118, 2192-2316, 6690-6778`）

- **拼图**：6 张视图排成 3×2（`columns=3`），8/9 张排成 4×2（`columns=4`），`_point_selection_contact_sheet`（`vlm_harness.py:2099-2118`）。对 `v13`~`v19`、`v18_history_safe_refinement` 等八视图/精修版本，改成发**两张原生分辨率大图**（IMAGE1 干净原图定位身份，IMAGE2 标注图选锚点），而不是拼接缩图，目的是避免缩图后小地标看不清。
- **锚点采样**：真正喂给 VLM 的"编号白点"由 `NavigationVLMHarness._ground_anchors`（`vlm_harness.py:2192-2214`，普通形式）或 `_vertical_ground_anchors`（`2271-2316`，只用于 `VERTICAL_UP`/`VERTICAL_DOWN`，沿楼梯方向按分位数采样）生成——**不是**已有文档提到的 `point_selectors.py:179-206`（那个函数叫 `_ground_anchor_options`，是 VLM 选完点**之后**用于 `legacy_rgbd_geometry` 契约下 navmesh 可达性修复的独立采样器，`rgb_only_v1` 下根本不会调用它；两者算法几乎一样——都是"先取掩码内离边缘最远的一点，再反复取一个与已选点集合距离最远、同时也够靠内部的点"——但服务的阶段完全不同）。**实测**：用同样的最远点采样逻辑在一条 80×240 像素的合成地面矩形上跑了一遍，6 个锚点会均匀撒开、且都精确落在掩码内（不会撒到掩码外一步）。
  - 每个视图独立采样，默认 `count=6`（`POINT_SCHEMA` 的 `anchor_index` 上限正是 `maximum: 5`，见 3.1.5）。
  - 采样完之后会过 `history_safe_anchors`（`vlm_harness.py:2436-2517`）——但这一步**只在 `self._history_safe_refinement` 为真时生效**，而它只在 prompt 版本是 `v18_history_safe_refinement`/`v19_pixel_ray_history_and_reverse_override`/`v31_circumnavigate_forward_competitor`/`v35_qualified_region_detector_grounding` 之一时才打开（`vlm_harness.py:1776-1781`）。**CLI 默认的 `v10_approach_relation_router` 不在这个集合里**，所以默认配置下这一步是直接原样透传锚点、什么都不做的空函数。打开后它会再去掉"射线方向落进来路/拉黑核心角度"的锚点，锚点全被滤空时还会退化成对整张掩码重新采样一次（换成更宽的安全列集合）。
- **Prompt 内容**（`vlm_harness.py:6699-6735`，节选要点）：当前子指令原文 + landmark + 完成线索 + 空间关系 + "禁止目标区域"；每个视图的检测证据、地面分割证据、**"按视图编号的合法地面锚点列表"**（`ground_anchor_evidence`）；到达上一个目标用过的动作历史；一串按 prompt 版本条件拼接的"方向/朝向/关系路线"提示段落；最后明文写"HARD REQUIREMENT: 必须选可见的编号白色锚点之一，返回它的 view_index 和 anchor_index，不许自己编坐标"，以及一句 "DEPTH-PROHIBITION: 只能用 RGB/DINO+SAM 掩码/2D 锚点位置做决定，没有深度/3D/navmesh/示范路径"。

### 3.1.5 回答格式与落地（`POINT_SCHEMA`，`vlm_harness.py:1205-1212`；`validate`，`vlm_harness.py:6780-6820`）

```
{"view_index": <int>, "anchor_index": <int, 0-5>, "reason": "<string>"}
```

- `validate()` 先检查 `view_index` 必须在这一轮真正 `allowed` 的视图列表里（不允许 VLM"发明"一个被排除的视图），再检查 `anchor_index` 必须落在该视图 `ground_anchors` 长度范围内，然后**直接查表** `point = anchors[anchor_index]` ——因为白点从一开始就只可能落在地面掩码内部，VLM 物理上不可能选到墙上、桌子上或画面外的像素。
- **供后续接力的"同车道最远锚点"修正**（仅在"支持部分完成继续前进"的路由延续场景下触发，`vlm_harness.py:6790-6819`）：如果这条边被标记为"上一步只完成了一部分、要继续走同一条走廊"，且被选锚点所在的车道（横坐标误差在画面宽度 25% 内）上还有更远的锚点，会自动把回答改写成那个更远的锚点——这不是新的语义决策，只是"沿着 VLM 已经选定的方向尽量往远处走"的几何延伸，`APPROACH_LANDMARK`/`STOP_WAIT`/`VERTICAL_UP`/`VERTICAL_DOWN` 这几个"到达型"关系不做这个延伸（因为它们的语义就是要停在近处）。
- 老的自由坐标格式 `STEP_SCHEMA`（`x_norm`/`y_norm`，`vlm_harness.py:1213-1220`）仍然存在，但只用于节点回溯（`STEP_GROUND_TARGET_SELECTION`，见模块 7）等非主选点路径；主选点路径固定走编号锚点这一套，不接受自由坐标。

### 3.1.6 视图筛选与"没有一张图能选"的崩溃点（`vlm_harness.py:2318-2677`）

`select_ground_target` 拿到全部候选后，先按下面顺序尝试收窄出一份 `allowed` 视图列表（简化描述，实际条件比这更细）：

1. 默认：`point is not None` 且没被 `excluded`/`hard_excluded` 标记的视图。
2. 如果按 1 收窄后空了，且当前是"终点关系"子指令、有软重开的检测证据，尝试放开来路方向里"切向"（离来路中心线 ≥25°）的射线。
3. 如果还是空的，且当前用的是 `_history_safe_refinement` 版本、且路线走廊生效，进一步放开被拉黑方向里"切向"（离拉黑中心线 ≥20°）的射线。
4. 如果不是 `_history_safe_refinement` 版本、且 1-3 都空，退回"只要有点、且不是硬排除"的宽松条件。
5. **如果连第 4 步都是空的**，直接抛 `RuntimeError("No floor-bearing candidate can be sent to the VLM")`（`vlm_harness.py:2677`）——这就是附录 B 讲的那类"没有地面可选"崩溃的报错来源；2026-09-11 起（272c223）序列策略把它接住转成正常结束的 episode，`scripts/evaluate_point_navigation.py:195` 也专门捕获这条消息归类为已知失败模式，而不是让整个评测进程崩掉。

拿到 `allowed` 之后才会给每个允许的视图撒锚点、拼图、发 prompt、拿回答、`validate()` 落地。如果 8 视图/精修版本启用了 `refinement_provider`，VLM 在正式作答前还可以请求最多 3 次局部转向探测（每次转到指定角度拍一张新图，转回来），这些探测是"事务性"的：只有被最终采纳的那一个才会真正提交进 `candidates` 变成第 9 张视图，其余探测过的画面只留在诊断记录里，不进最终的九宫格审计。

### 3.1.7 rgb_only_v1 与 legacy_rgbd_geometry 在这里的关键分叉（`point_selectors.py:1586-1786`）

拿到 VLM 选好的 `(chosen_index, vlm_point)` 之后，代码有一大段"navmesh 修复/最小推进距离拒绝"逻辑，**但这一整段用 `if not rgb_only and (...)` 包起来**（`point_selectors.py:1586`），也就是说：

- `legacy_rgbd_geometry` 契约下：会调用 `repair_selected_ground_point`（`point_selectors.py:344-`），用 `sim.pathfinder.snap_point`/`ShortestPath` 检查选中像素反投影到世界坐标后是否在 navmesh 上可达，不可达就在同一视图或相邻 ≤45° 视图里换一个更近的、真正可达的锚点；还会检查"选中点离当前位置的测地距离是否小于 `minimum_progress_distance_m`"，太近就整段拒绝重选。
- **`rgb_only_v1` 契约下，这两段完全不执行**——因为它们都需要 `sim.pathfinder`/`position`，这正是 RGB-only 硬约束（第 0 节）明令禁止在线路径读取的信息。RGB-only 模式下，"选的点能不能走到"这件事完全交给了模块 4（点导航执行器 + TAPIR 点跟踪 + 到达判定）在执行阶段用纯 RGB 手段兜底，选点阶段本身不再有任何物理可达性校验。

选点之后唯一还会做的收口是**严格模式的"吸附"**（`point_selectors.py:1727-1744`）：如果用了 dense-majority 严格地面分割，且 VLM 选中的像素因为之前某个补丁扩张过的中间掩码而落在了"最终掩码"之外，就在最终掩码里找离它最近的合法像素吸附过去（只在已经可见的 `target_mask` 内挑，不引入任何新矩形/深度/navmesh）；这一步与"VLM 只能选编号锚点"是两道独立保险——前者防 VLM 瞎指，后者防中间过程的掩码扩张让编号点本身站到了严格地面之外。

### 3.1.8 VLM 后端：`DeepSeekBackend` 与 `HeuristicBackend`

- **`DeepSeekBackend`**（`vlm_harness.py:834-` 起）：走 DMXAPI 中转站的 OpenAI 兼容 Chat Completions 接口，配置优先级"显式参数 > 环境变量 > `.env.deepseek` > `local_env.sh` > 内置默认"（与 CLAUDE.md 描述一致）；请求里强制 `response_format=json_object`、`thinking: disabled`（否则每次思考 6k+ token、约 60 秒）；`disable_proxy` 默认打开，用一个空 `ProxyHandler` 屏蔽掉继承来的 `http(s)_proxy`，保证调用不依赖手动 mihomo 代理是否开着；408/429/5xx 与超时按指数退避重试（默认最多 3 次），401/402/403 判定为凭据问题、立即抛 `VLMProviderFatalError` 不重试。
- **`HeuristicBackend`**（`vlm_harness.py:1035-1164`）：不联网的确定性假后端，只用于冒烟测试。它对选点请求的处理极其简单粗暴——正则解析 prompt 里的 `"allowed view from [0, 2, 4]"` 这句话，直接返回列表里**第一个**允许的视图号，`anchor_index` 恒为 `0`，`reason` 固定写死成 `"first safe ground anchor"`。它完全不看 RGB 内容、不理解指令语义，唯一的作用是让没有 API key 的环境也能把整条流水线跑通（`--vlm-backend heuristic`）。

### 3.1.9 Prompt 版本历史

选点 prompt 从 `v1_baseline` 迭代到 `v35_qualified_region_detector_grounding`，全部定义在 `NavigationVLMHarness.POINT_SELECTION_PROMPT_VERSIONS`（`vlm_harness.py`），旧版本一律不删（`docs/vlm_point_selection_optimization.md` 只追加）。CLI 默认值是 `v10_approach_relation_router`（`habitat_point_navigation.py:668-673`，`evaluate_point_navigation.py` 同款默认），也就是说**默认在线路径走的既不是最早的 baseline，也不是最新的 v35，而是 v10**；3.1.4/3.1.6 里提到的"`_history_safe_refinement`/`_first_step_route_guard` 默认关闭"正是因为它们只对 v18/v19/v31/v35（历史安全）和 v20/v32/v34（首步路线护栏）这几个特定后续版本生效，v10 走的是更早、更简单的关系路由逻辑（按 landmark/关系文本路由到对应的选点策略，不做锚点方向的二次历史校验）。

### 3.1.10 各种情况处理一览（速查表）

| 情况 | 处理 |
|---|---|
| 某视图检测器把墙/门误判成大片"地面" | `broad_uniform_vertical_support` 规则整张剔除该检测（非严格模式） |
| 与来路夹角 <50° | 软排除 `excluded`，扣分但不必然出局 |
| 与"被拉黑方向"夹角 <50° | 硬排除 `hard_excluded`，扣 100 分且 `select_ground_target` 默认绝不重开 |
| 终点关系子指令、来路排除内但有强检测证据、跳出 25° 核心区 | 软重开该切向射线（`semantic_backtrack_override`） |
| 所有视图收窄后都没有合法射线 | 依次尝试切向重开(来路→拉黑)，全部失败则 `RuntimeError("No floor-bearing candidate can be sent to the VLM")` |
| `PASS_LANDMARK` 但检测分数 <0.40 | 该检测不参与地面收窄，只作弱证据进 prompt |
| `STOP_WAIT`/`BETWEEN_OBJECTS`/`ENTER_REGION`/`TRAVERSE_PORTAL_REGION` | 强制用 soft 跨视图策略，不因别处有检测证据而清空本视图候选 |
| `VERTICAL_UP`/`VERTICAL_DOWN` | 锚点用 `_vertical_ground_anchors` 按图像纵向分位数采样，而不是最远点采样 |
| VLM 返回的 `anchor_index` 越界/`view_index` 不在允许列表 | `validate()` 直接抛 `ValueError`，触发 VLM 调用重试（同 STAGE_DECOMPOSITION 的重试机制） |
| dense-majority 严格模式下选中像素落在最终掩码外 | 在最终掩码内吸附到最近合法像素，记入 `strict_ground_snap` |
| `rgb_only_v1` 契约 | 完全跳过 navmesh 可达性修复与"最小推进距离"拒绝，这两项只属于 `legacy_rgbd_geometry` |
| 8 视图/精修版本，VLM 想先看看别的角度再决定 | 最多 3 次事务性局部转向探测，只有被采纳的那次提交为正式第 9 张视图 |
| `--vlm-backend heuristic`（无网络/无 key） | 恒选第一个允许视图的 0 号锚点，仅用于冒烟测试，不代表真实选点质量 |

### 3.1.11 下游谁在用

`choose_view` 返回的 `chosen` 字典（含最终像素 `point`、朝向 `yaw`、`target_mask`、`vlm_selection` 里的完整决策审计）直接喂给模块 4 的点导航执行器；`candidates`（全部视图）连同 `vlm_calls.json`/`vlm_calls_attempts.json` 一起写入 `trajectory.json` 供 `validate_single_point_run.py`/`verify_round_stage_completions.py` 事后审计；`InstructionVLMPointSelector.select()`（`point_selectors.py:2079-2101`）是 `habitat_point_navigation.py` 里在线策略实际持有的封装对象。

### 3.1.12 已知的边角与坑（静态阅读结论，部分经实测验证角度/采样部分）

- `history_safe_anchors` 的"历史安全"过滤默认对 CLI 常用配置（v10）是空操作；只读代码容易误以为这层保护在所有版本下都生效。
- 本文档模块 3 原文（会话整理稿）把"锚点怎么撒"归到 `point_selectors.py:179-206`（`_ground_anchor_options`），但那其实是 `legacy_rgbd_geometry` 选点后修复用的采样器；真正喂给 VLM 看的锚点由 `vlm_harness.py` 里几乎同构但独立维护的 `_ground_anchors`/`_task30_ground_anchors`/`_vertical_ground_anchors` 生成——两套代码逻辑相似但物理上互不调用，未来改其中一个采样算法很容易漏改另一个。原文保留不改，以本节为准。
- `repair_selected_ground_point`/`minimum_progress_distance_m` 拒绝逻辑在 `rgb_only_v1` 下完全不存在，意味着这条正式路径里"选的点是否物理可达"从选点阶段起就没有任何几何校验，全部责任下放给模块 4；一旦模块 4 的跟踪/到达判定本身有 bug，选点阶段不会有任何提前拦截。
- `_task30_ground_anchors`（横向分位数采样，能覆盖侧边入口）虽然定义了，目前只在 `vlm_harness.py:6138` 一处、特定版本分支下被真正实例化调用，多数版本仍用默认的 `_ground_anchors`（只偏向掩码内部最深连通分量）——如果地面掩码是"细长走廊一侧突然多出一个岔口"这种形状，默认采样器可能不会在岔口那侧放锚点，VLM 想选也选不到。
- `HeuristicBackend` 的选点行为是"选第一个允许视图的 0 号锚点"，这意味着用它跑的冒烟测试即便"通过"，也完全不能说明选点语义逻辑是对的，只能证明流水线接线没断。

---

# 模块 4：点导航执行器——"盯着那个点走过去"

**文件**：`scripts/point_navigation_executor.py`（2284 行）、`scripts/image_goal_policy.py`、`scripts/track_cluster.py`

这是唯一真正让机器人动起来的模块，也是**唯一有权说"我物理上到了"**的模块。它不懂指令、不调 VLM。

**输入**（`PointNavigationRequest`，第 762 行）：当前 RGB、目标像素、地面掩码。
**输出**（`PointNavigationResult`，第 811 行）：`arrived`（到没到）、`end_reason`（`rgb_only_dense_stop_cluster_arrival` 到了 / `rgb_navigation_cluster_lost` 跟丢 / `max_steps` 步数用完）、完整动作历史、最终画面。

**内部逻辑，分四步：**

## 4a. 做"目标裁图"（`crop_goal`，第 536 行）

GNM/ViNT 这类"图像目标策略"是这样工作的：给它"你现在看到的"和"你想到达的地方长什么样"两张图，它输出往哪走。但我们只有一个像素，没有"目的地照片"，所以要造一张：以目标像素为裁剪框的**底边中点**，往上裁到与它关于画面中线对称的高度（至少 1/3 画面高），按 85:64 的比例裁下来。直观理解：**"把目标点脚下那块地面连同前方景象裁成一张小照片，假装这就是目的地照片"**。

## 4b. 撒三组跟踪点（`dual_crop_tracking_clusters`，第 586 行）

在裁图上撒三组点，再全部吸附到地面掩码上，然后交给 **TAPIR**（一个逐帧在线点跟踪模型，`track_cluster.py:71`）盯着：

- **导航簇**（3×3 = 9 点，裁图正中）：负责"转向"——看这些点在画面里偏左还是偏右。
- **目标簇**（3×3 = 9 点）：负责每步更新裁图位置。
- **到达簇**（5×9 = 45 点，裁图最底部一条带）：负责"判断到了没有"。它有独立的 TAPIR 状态，不会干扰导航簇。

## 4c. 每一步怎么决定动作（第 1076-1289 行）

1. `tracker.step(rgb)` 更新所有点的位置和"是否可见"。
2. 用最近 6 帧 + 目标裁图喂给 GNM/ViNT（`image_goal_policy.py:102`，图缩到 85×64、ImageNet 归一化），拿到 5 个航点，取第 2 个航点算出一个角度。
3. 用导航簇可见点的像素中心算出另一个角度：`atan((中心x − 画面中心) / (0.8×画面宽))`。
4. **融合：80% 用点簇角度 + 20% 用 GNM 角度**（第 1191 行）。也就是说主要靠"盯着点"，学习策略只是辅助。
5. 死区 ±7°：角度 > 7° → `turn_right`，< −7° → `turn_left`，否则 `move_forward`。

## 4d. 怎么判断"到了"（`rgb_only_dense_stop_v1` 配置，第 482 行）

思路：**当你走到目标点上时，目标点脚下那片地面就会从画面底部掉出去看不见**。所以：

- 到达簇 45 个点里 ≥ 50% 不可见，且这个状态连续 3 帧；
- 同时目标簇全部不可见；
- 同时必须真的走过：至少发过 3 次 `move_forward`，且至少 2 帧画面变化量 ≥ 2.0（灰度平均差，防止原地转圈假到达）。

三条同时满足 → `arrived=True`。导航簇连续 3 帧全丢会先降级用目标簇/到达簇撑一下，撑不住就 `rgb_navigation_cluster_lost`。默认最多 32 步。

**举例**：目标像素在 640×480 画面的 (320, 400)。裁图从 y=79 到 y=400、宽 426 像素。63 个跟踪点撒下去。第 0-2 步点在正前方 → 连续 `move_forward`；第 3 步点偏右 9° → `turn_right`；第 4-7 步继续前进，目标点在画面里越来越大、越来越靠下；第 8 步起底部 45 个点开始掉出画面；第 10 步满足"≥50% 不可见连续 3 帧 + 走过 ≥3 步 + 画面确实在变" → 判定到达，返回 `arrived=True`。

## 4.1 扩充细节：导航执行器怎么"盯点走路"、又怎么知道"走到了"

> 本节 2026-09-11 补写。`crop_goal`／`dual_crop_tracking_clusters`／到达状态机这几块纯几何和状态机逻辑，在本机用合成数据（假的 `sim`/TAPIR 跟踪器/GNM 策略，不加载任何 GPU 模型、不连网）实际跑通过，下文带"实测"字样的数字都是真实程序输出，不是手算。`image_goal_policy.py`（GNM/ViNT/NoMaD 网络本身的推理）、`track_cluster.py` 里真正的 TAPIR 前向、以及 `legacy_rgbd_geometry` 那条历史分支只做了静态阅读。行号以 2026-09-11 快照为准，像素统一按正式配置 320×240（上面 4d 小节举的 640×480 例子是会话整理稿里的错误单位，下面 4.1.7 用 320×240 重新给了实测过的轨迹，以此为准；原文保留不改）。

### 4.1.1 一句话 + 比喻

这个模块就是"让机器人真的用脚走过去，并且自己判断有没有走到"。前面的模块（VLM）只给了一个像素坐标，相当于在地图上画了个×；这个模块要做的事像是：蒙着眼睛（看不到地图、不知道坐标、不知道距离），只凭眼前的画面和"我刚刚发了什么指令"，一步步挪过去，并且只能靠"那个×标记的地方是不是已经从我脚下消失了"来判断自己是不是已经站到上面。

### 4.1.2 输入输出：哪些字段是"允许"的，哪些是"绝对不能出现"的

入口是 `PointNavigationExecutor.execute()`（第 1309 行），它先看 `request.policy_input_contract`：
- `"rgb_only_v1"` → 走 `_execute_rgb_only()`（第 943 行），这是唯一的正式路径；
- `"legacy_rgbd_geometry"` → 走后面 1316 行起的老分支，里面大量读取 `selected_point_depth_m`、`selected_point_navmesh_xyz`、`self.sim.pathfinder`（深度/navmesh/geodesic），这是 RGB-only 规则冻结前的几何版本，`TRACKING_CLUSTER_PROFILES` 里 `legacy_3x3` 到 `dense_stop_motion_recovery_v28_endpoint_loss_guard` 这二十多个 profile（第 25-475 行）全部只服务这条老分支，**正式 CLI 永远不会走到它们**，保留只是为了老单测和历史复现实验能跑。
- 其它字符串 → 直接 `raise ValueError`。

`_execute_rgb_only` 进门第一件事是"查违禁品"（第 954-969 行）：把 `selected_point_depth_m`、`selected_point_reachable`、`selected_point_initial_geodesic_m`、`selected_point_navmesh_xyz`、`max_travel_distance_m`、`reference_path` 这六个字段挨个检查，只要有一个不是 `None` 就直接抛 `ValueError`，报错信息里会列出具体哪个字段"泄漏"了。**实测**：在 `tests/test_rgb_only_contract.py:70-81` 的既有测试基础上确认过，只要随便塞一个 `selected_point_depth_m=2.0` 进 `rgb_only_v1` 请求，执行器连跟踪器都不会调用就直接报错——这是代码层面对项目规则第 0 条的硬执行，不是靠人自律。第二件事是检查 `tracking_cluster_profile` 必须等于 `"rgb_only_dense_stop_v1"`，否则同样拒绝。

输出 `PointNavigationResult`（第 811 行）里 `end_reason` 在 `rgb_only_v1` 路径下只有四种取值（实测逐条触发过前三种，第四种是 `max_steps` 用完时的兜底，合成实验里也触发到了）：

| `end_reason` | 含义 | 触发位置 |
|---|---|---|
| `rgb_only_dense_stop_cluster_arrival` | 判定到达 | 第 1144 行 |
| `rgb_navigation_cluster_lost` | 导航簇（以及备用簇）全部跟丢，提前放弃 | 第 1164 行 |
| `no_rgb_crop` | `crop_goal` 返回空裁图 | 第 1175 行 |
| `max_steps` | 跑满步数预算仍未到达/跟丢 | 第 1291 行（`for...else` 分支） |

`no_rgb_crop` 这一条，静态读下来几乎打不到：到达这步之前代码已经先判过 `control_visible.any()`，不满足就已经 `break` 成 `rgb_navigation_cluster_lost` 了，所以能走到 `crop_goal` 这一行时 `crop_tracks/crop_visible` 理论上必然非空——这是一条写出来的防御性分支，不代表真实会命中，合成实验里没能人为制造出来，只能说"静态读代码判断几乎不可达"。

### 4.1.3 造"目标照片"：`crop_goal` 到底裁的是哪一块（第 536 行）

GNM/ViNT/NoMaD 这类图像目标策略要喂两张图："现在看到的" + "想去的地方长什么样"。我们手里只有一个像素，`crop_goal` 的做法是：

1. 把当前可见的锚点簇取中位数，当成"底边中点" `bottom`；
2. 以画面中线为镜子，算出 `bottom` 的镜像点当"顶边中点" `top`（公式：`top = (bottom_x, 2×画面中心y − bottom_y)`）；
3. 如果这段 `top→bottom` 的长度小于画面高度的 1/3（320×240 下是 80 像素），就把 `top` 沿竖直方向拉到正好 80 像素远——这是为了防止目标点太靠近画面中心时裁出一张几乎没有空间信息的"扁图"；
4. 按输出比例 85:64（GNM/ViNT 的 `image_size`）算出左右宽度，用透视变换把这个四边形裁出来、缩放到 85×64。

**实测**（320×240，地面 mask 是第 110 行以下，目标像素选在 (160, 200)）：`bottom=(160,200)`、镜像得到的 `top=(160,39)`（因为画面中心纵坐标是 119.5，2×119.5−200=39），轴长 161 像素，大于最小高度 80 像素所以不需要拉伸。这和 4d 小节用 640×480 画面算出的"裁图从 y=79 到 y=400"不是同一套单位，**正式配置下没有 640×480 这档分辨率**，下面 4.1.7 会给一条完整的 320×240 实测轨迹替换掉那个例子。

还实测了一次"地面只剩最下面 5 行"的极端情况（目标点几乎贴底部）：`top` 被直接顶到第 1 行，轴长 237 像素——说明当选中点已经在画面最底部时，这张"目标照片"会几乎把整张画面都裁进去，这是合理的退化行为，不是 bug。

### 4.1.4 撒三组跟踪点：`dual_crop_tracking_clusters`（第 586 行）

这一步在裁图坐标系里放三种网格点，再分别映射回原始 320×240 画面、吸附到地面 mask 上最近的合法像素（`snap_points_to_mask`，第 504 行，用暴力最近邻，逐点去重，保证一张图里不会有两个跟踪点重合在同一像素）：

- **导航簇**：裁图中心九分之一区域（即裁图宽高各自的 [1/3, 2/3] 区间）里撒 `rows×cols` 个点，`rgb_only_dense_stop_v1` 配置是 3×3 = 9 点（第 483-484 行）。
- **到达簇**：裁图最底部 16%（从 84% 高度到 99%）的横向 [32%, 68%] 区间里撒 5×9 = 45 点（第 485-490 行）。
- **目标簇**（代码里单独用 `profile="legacy_3x3"` 再调一次这个函数得到）：同样在中心九分之一区域撒，但固定是旧配置的 3×3 = 9 点（第 1000-1002 行）。

三组点加起来 9+45+9 = **63 个跟踪点**，实测确认过这个数字。它们被分给两个独立的 TAPIR 实例：导航簇 + 目标簇（18 点）交给主 `self.tracker`，到达簇（45 点）交给独立的 `self.arrival_tracker`（第 1003-1015 行）——两个 TAPIR 各自维护自己的在线 causal 状态，互不干扰，所以"到达簇看丢了"不会影响导航簇的跟踪质量。

**如果网格点落的位置没有地面**，有两层兜底（第 654-668 行），**实测过第一层**：
1. 如果中心九分之一区域里一个地面像素都没有（比如目标点已经贴到画面边缘，中心区域全是墙），导航簇/到达簇就退化成"裁图里随便一块地面都算"，而不是凭空造一个没分割过的矩形——用"地面只剩最下面 5 行"的极端 mask 实测触发了 `navigation_region_ground_fallback: True`，到达簇因为本来就贴底所以没触发。这体现了项目"只能减像素不能加像素"的同一条硬约束在执行器层也生效。
2. 如果整张裁图在原始 mask 上完全没有交集（例如目标贴在裁图的最上边缘，OpenCV 反向光栅化漏掉了边界像素），会把原始 mask 的像素正向投影回裁图坐标、容差 1 像素内贴回去（`boundary_projection_used`，第 621-640 行）——这条没能在合成实验里触发，只读了代码，注释里说是"高处目标（楼梯/上层平台）"场景的补丁。

### 4.1.5 GNM / ViNT / NoMaD 三选一：差异不只是模型，输入格式也不同

`image_goal_policy.py` 里三个模型共用同一个 `predict()` 函数，但配置完全不同（`models/visualnav-transformer/train/config/*.yaml`，实测读取）：

| | `context_size`（上下文帧数） | `image_size` | `len_traj_pred`（航点数） | 输出方式 |
|---|---|---|---|---|
| GNM | 5 | 85×64 | 5 | 一次前向直接出 1 条轨迹 |
| ViNT | 5 | 85×64 | 5 | 一次前向直接出 1 条轨迹 |
| NoMaD | **3**（不是 5） | **96×96**（不是 85×64） | **8**（不是 5） | 扩散模型，要跑 `num_diffusion_iters` 步去噪，`samples` 份候选轨迹里只用第 0 份 |

**`--policy` 的默认值是 `gnm`，不是 ViNT**——核实 `evaluate_point_navigation.py:345-346` 和 `habitat_point_navigation.py:663` 的 argparse 定义后确认，两处默认值都写的是 `"gnm"`，`run_e2e_eval.sh` 也没有覆盖它。执行器调用时还强制传了 `samples=4`（覆盖 `image_goal_policy.py` 自己的默认值 8），但这个参数**只对 NoMaD 有意义**——GNM/ViNT 的前向代码根本不读 `samples`，所以默认配置下这个参数形同虚设。

`PointNavigationExecutor.policy_heading()`（第 739 行）统一处理三种模型的输出：取第 0 条轨迹的第 2 个航点（下标 1，如果只有 1 个航点就退化成下标 0），算 `atan2(y, max(x, 1e-4))` 当成"策略建议的朝向角"。GNM/ViNT 每次只给 1 条轨迹，NoMaD 给 `samples` 条但只取第 0 条——也就是说 NoMaD 那套"扩散采样产生多个候选路径"的能力在这里完全没被用上，只是顺带跑了一次扩散而已。这点是静态读代码得出的，没有实际跑 NoMaD 验证（需要 GPU + 权重）。

### 4.1.6 每一步怎么决定转/走：角度融合与死区（第 1180-1204 行）

1. 先用**导航簇**（或下面讲的应急替补簇）当前可见点的像素中心算一个"盯点角"：`pixel_angle = atan((中心x − 画面宽/2) / (0.8 × 画面宽))`。320 宽时分母是 256。举例：中心点在 x=200（比画面中心偏右 40 像素）→ `atan(40/256) ≈ 8.85°`。
2. 用 GNM/ViNT/NoMaD 给的 `navigation_angle` 先截断到 ±0.6 弧度（≈ ±34.4°）再参与融合，防止学习策略在训练分布外给出离谱的转向建议。
3. **融合公式**：`desired_turn = 0.8 × pixel_angle + 0.2 × clip(navigation_angle, −0.6, 0.6)`（第 1191-1193 行）——80% 听"盯点"的，20% 听学习策略的，学习策略只是微调，不是主导。
4. **死区 ±7°**（`turn_deadband_deg`）：`desired_turn` 超过 7° 就 `turn_right`，低于 −7° 就 `turn_left`，否则 `move_forward`。每次转动固定转 `turn_step_deg`（默认 15°），不会按 `desired_turn` 的大小比例转，只有"转/不转"两档。

**导航簇跟丢时的应急替补链**（第 1097-1112 行，`navigation_loss_grace_frames=3`）：如果导航簇这一帧完全不可见，但跟丢连续帧数还在 3 帧宽容期以内，就依次尝试用目标簇→到达簇→"上一次还看得见时的导航簇坐标"顶上去算转向角；超过宽容期、且三者都没有，才会判定为 `rgb_navigation_cluster_lost` 提前结束。这是对"门口/转角处 TAPIR 短暂丢点"的容错，不等于"到达"。

### 4.1.7 判断"到了"：两条独立路径，外加实测完整轨迹

`rgb_only_dense_stop_v1` 的到达规则不是 4d 小节写的"一条规则"，而是**两条并行判据**（第 1131-1141 行），满足任意一条即可：

- **路径 A（常规路径，`ordinary_arrival`）**：目标簇（9 点）这一帧**完全不可见**，且到达簇（45 点）可见比例 ≤50% 的状态已经**连续 3 帧**（`arrival_confirmation_frames`），同时"已经真走了"的证据成立（见下）。
- **路径 B（三簇同时消失，`terminal_consensus`）**：导航簇、目标簇、到达簇在**同一帧**一起消失（不需要等 3 帧确认），同时"已经真走了"的证据也成立。这是后来才加的规则（配置里叫 `terminal_all_cluster_loss_is_arrival`），理由是三个互相独立跟踪的簇一帧内同时失踪，本身已经是比"到达簇慢慢消失"更强的证据，没必要再等。

"已经真走了"（`action_progress`）统一要求：累计发过 ≥3 次 `move_forward`（`minimum_forward_commands`），且**最近一次转向之后**至少发过 1 次 `move_forward`（`forward_commands_after_turn`，防止用很久以前、转向之前积累的前进次数冒充"刚刚还在走"），且至少有 2 帧画面变化量（前后帧灰度均值绝对差）≥2.0（`minimum_rgb_motion_frames`/`rgb_motion_threshold`，防止原地打转或卡住不动时画面几乎不变也被判到达）。

**实测 1（路径 B，三簇同时消失）**：用假 TAPIR 让导航簇/目标簇/到达簇在同一帧一起从"可见"变"不可见"，结果第 4 次动作（全是 `move_forward`）之后立刻判定到达，`record["terminal_cluster_consensus"] = True`——确认了"同时消失不需要等 3 帧"。

**实测 2（路径 A，常规 3 帧确认）**：用假 TAPIR 让导航簇永远可见、只让目标簇+到达簇在第 3 次动作后消失，结果是又多走了 **2 步**（总共 5 次 `move_forward`）才判定到达，`terminal_cluster_consensus = False`。也就是说从"目标簇第一次看不见"到"真正判到达"，中间还要再撑过 2 帧——这 2 帧里机器人仍在按上一次有效的转向角继续走。

**实测 3（到达簇单独消失、导航簇/目标簇一直可见）**：到达簇单独消失、导航簇和目标簇（都属于同一个 TAPIR 实例）始终可见时，规则**永远不会触发**——因为路径 A 强制要求"目标簇这一帧完全不可见"，路径 B 要求三簇同时消失；单独一个到达簇消失、另外两簇一直在，20 步全跑满最后只会是 `max_steps`。换句话说，**"到达簇大半不可见"本身不是到达的充分条件，还必须搭配目标簇的消失**，这一点 4d 小节的措辞（"到达簇 45 个点里 ≥50% 不可见…同时目标簇全部不可见"）其实已经写对了，这里只是拿真实运行把它坐实。

用这三组实测替换 4d 里 640×480 的手算例子：在 320×240、目标像素 (160,200) 的配置下，裁图轴长 161 像素（不是旧例子的 426 像素宽），63 个跟踪点撒法不变；如果机器人一路朝目标走、TAPIR 跟踪稳定，最快会在连续 3 次 `move_forward` 之后、目标簇+到达簇同帧消失时的下一帧就判到达（路径 B，实测 4 步），比目标簇单独先消失、到达簇再慢慢降到 50% 以下的常规路径（路径 A，实测至少多 2 步）更快。

### 4.1.8 各种情况速查表

| 情况 | 执行器怎么处理 |
|---|---|
| 导航簇这一帧丢了，丢之前 ≤3 帧 | 用目标簇/到达簇/"上一次看得见的导航簇坐标"顶替着算转向角，继续走 |
| 导航簇连续丢 >3 帧、且目标簇/到达簇也没有 | `end_reason = rgb_navigation_cluster_lost`，提前结束，不算到达 |
| 目标簇全不可见 + 到达簇 ≤50% 可见连续 3 帧 + 已走够 3 步 + 画面确实在变 | 路径 A 到达 |
| 导航簇+目标簇+到达簇同一帧全部消失 + 已走够 3 步 + 画面确实在变 | 路径 B 到达（不需要等 3 帧） |
| 只有到达簇消失，导航簇/目标簇一直可见 | 永不判到达，只能等 `max_steps` 超时 |
| 裁图网格点落的格子里没地面（如目标贴边缘） | 退化成"裁图里随便一块地面都算"，不会凭空造矩形 |
| 策略选了 NoMaD | `context_size`/`image_size`/`len_traj_pred` 全部换成 3/96×96/8，`samples=4` 才真正生效 |
| 策略选了 GNM（默认）或 ViNT | `samples=4` 无意义，一次前向直出 1 条轨迹 |
| `request.policy_input_contract="legacy_rgbd_geometry"` | 走第 1309 行之后的老分支，读深度/navmesh，只服务历史测试，正式 CLI 不会触达 |
| 请求里混入了 `selected_point_depth_m` 等六个违禁字段之一 | 还没碰跟踪器就直接 `ValueError`，报出具体字段名 |

### 4.1.9 下游谁在用

- `habitat_point_navigation.py` 的主循环按子指令逐个构造 `PointNavigationRequest`，拿到 `PointNavigationResult` 后把 `action_history`、`edge_keyframes`、`record`（完整逐步轨迹）原样写进 `trajectory.json` 和节点/边；`arrived`/`end_reason` 决定要不要进入模块 6 的完成判定。
- 模块 5（`NavigationGraphMemory`）用 `edge_keyframes`（均匀抽样的关键帧，默认 5 张）和完整 `action_history` 作为边的内容，从不读本模块内部的跟踪点坐标或裁图几何。
- 模块 8（事后打分）只读 `record["arrival_signal"]`/`end_reason` 和 `action_history`，隐藏几何评分走的是完全独立的 `path_projection.py`，不经过这个模块的任何字段。

### 4.1.10 已知坑 / 历史遗留

- `TRACKING_CLUSTER_PROFILES` 里从 `legacy_3x3` 到 `dense_stop_motion_recovery_v28_endpoint_loss_guard` 的二十多个 profile 都是 RGB-only 规则冻结前的几何实验史（不少直接用 depth/navmesh/geodesic），只有最后加的 `rgb_only_dense_stop_v1` 是正式可达的；读代码时很容易被这一长串历史 profile 干扰，**正式路径只认这一个名字，写死在 `_execute_rgb_only` 的断言里**。
- `make_sim()` 函数签名里 `forward_step` 默认值是 0.25 米，但 CLI 实际默认传的是 `--forward-step 0.22`（`habitat_point_navigation.py:732`，两处默认值不一致，容易看错；实际跑的永远是 0.22，因为 `args.forward_step` 总会覆盖函数默认值。附录 B 里"每帧前进 0.25 m"的说法按此应读作 0.22 m）。
- `samples=4` 这个硬编码传参只在选了 NoMaD 时才有意义；默认策略是 GNM，不是某些文档措辞给人的 ViNT 印象。
- `no_rgb_crop` 这个 `end_reason` 静态读下来几乎是不可达的防御分支，没能在合成实验里构造出触发条件。
- `boundary_projection_used` 这条边界光栅化补丁只做了静态阅读，没有实测触发。
- `legacy_rgbd_geometry` 分支（第 1313-2284 行，约 970 行代码）完整保留了旧的 `stall_recovery`/`online_endpoint_geodesic`/`near_geodesic` 等机制，但因为正式 CLI 的 `policy_input_contract` 只会是 `"rgb_only_v1"`，这条分支现在只被历史单测和 `evaluate_point_arrival_profiles.py` 这类离线 ablation 脚本调用，不在生产闭环里。

---

# 模块 5：记忆图（NavigationGraphMemory）

**文件**：`scripts/navigation_graph_memory.py`

**输入**：每次执行器停下来时的 6 张图 + 8 张图 + 检测结果 + 这段路的动作历史。
**输出**：磁盘上的 `navigation_graph/navigation_graph.json` + `nodes/node_XXXX/view_*.jpg`。

**内部逻辑**：

- **节点** = "我停过的一个地方"。存：6 张环视图、8 张 45° 图（只给判卷用）、DINO+SAM 检测到的环境物体列表、一个 256 维**视觉指纹**（`CompactVisualEmbedder`，第 56 行：8×8 缩略图 + HSV 直方图 + 边缘方向直方图，不是神经网络，可以以后换成 CLIP/DINO）。**坐标和朝向字段必须为空**（第 155 行注释），这是 RGB-only 规则。
- **边** = "从 A 走到 B 的过程"。存：完整动作序列、按步数均匀抽的 5 张关键帧、边的种类（正常前进 / 回溯尝试 / 回到老地方的闭环）。
- 起点建一个 `episode_origin` 节点，之后**执行器每停一次建一个节点**（第 372 行），不管是到达还是失败。

**举例**：起点 node_0000；走过桌子停下 → node_0001，边 0000→0001 存着"前进×8、右转×1、前进×3"和 5 张关键帧；再到门口 → node_0002。以后要"回到走过桌子的地方"，就是拿 node_0001 的 6 张图当参考。

## 5.1 扩充细节：把记忆图模块从头到尾讲透

> 补写说明（2026-09-11）：本节基于对 `scripts/navigation_graph_memory.py`（全文 591 行）的通读，以及对四处在线调用点（`scripts/rgb_only_instruction_sequence.py:236,453`、`scripts/instruction_sequence_exploration.py:2606`、`scripts/habitat_point_navigation.py:1649`）的静态阅读。`CompactVisualEmbedder` 的维度、归一化和余弦相似度数值已用 `source local_env.sh` 后的合成图像**实际跑过**（见 5.1.3 的实测表格）。节点/边"只在成功到达时才写入"这一点已对四处调用点逐一核对源码，属于静态确认（不是跑通整条流水线跑出来的），下文会标出。**注意：原文第三条"执行器每停一次建一个节点，不管是到达还是失败"经核对是不准确的，以 5.1.4 为准；原文保留不改。**

### 5.1.1 一句话 + 比喻

记忆图模块就是导航过程中的"相册 + 路书"。每次执行器真的走到了一个子指令该到的地方，它就拍一张全景照片、记一笔"我是怎么走到这儿的"，然后把这一页贴进相册。如果以后要回头找某一页，可以按贴的顺序原路翻回去；如果发现眼前的场景和相册里某一页很像，还能拿出两张照片比对"像不像"。相册里绝不写"我当时站在地图上的第几号坐标"——按项目规则，这本相册只能看，不能看"我在哪"。

### 5.1.2 输入输出

**节点（`NavigationNode`）**存的东西：

| 字段 | 内容 | RGB-only 下的值 |
|---|---|---|
| `node_id` | `node_0000`、`node_0001`…按创建顺序编号 | 正常写入 |
| `node_kind` | `episode_origin` / `point_navigation_arrival` / `point_navigation_stop` | 见 5.1.4 |
| `position_xyz` / `base_yaw_rad` | 仿真器位姿 | **强制 `None`**（第 155-159 行注释明确写"生产 RGB-only 跑必须为 `None`"） |
| `six_views` | 6 张全景图路径 + 相对偏航角（+ 若有 base_yaw 则附带绝对偏航，仅用于人工复核，线上不读） | 正常写入 |
| `environment_semantics` | 该节点六视图上的 DINO+SAM 环境检测（14 词固定词表 + 指令 landmark，见 2.1.9） | 正常写入 |
| `visual_embedding` / `visual_embedding_model` | 256 维视觉指纹 + 模型名 `compact_rgb_hsv_edge_v1` | 正常写入 |
| `arrival_signal` | 执行器给出的到达信号字符串 | 正常写入 |
| `metadata` | 自由字典，各调用方塞不同东西（`policy_input_contract`、`point_target_arrived` 等） | 正常写入 |

**边（`NavigationEdge`）**存的东西：`source_node_id`/`target_node_id`、`departure_purpose_sub_instruction`（这条边是为哪句子指令走的）、`action_history`（完整的逐步动作序列）、`control_step_count`、`traveled_distance_m`（RGB-only 下永远是 `0.0`，因为在线动作历史里不含真实位移）、`edge_kind`。

### 5.1.3 `CompactVisualEmbedder`：256 维指纹怎么拼出来的

这是全模块里唯一"算东西"的地方，纯 numpy/cv2，不加载任何模型权重。对每一张全景视图（RGB）：

1. **8×8 缩略图（192 维）**：用 `cv2.resize` 的 `INTER_AREA`（区域平均）把整张图缩成 8×8×3，摊平成 192 个数，除以 255 归一化到 0~1。这部分本质是"整张图的低分辨率色块布局"。
2. **HSV 三通道直方图（16×3=48 维）**：转 HSV 后对 H/S/V 各自算 16 桶直方图，每个直方图自己除以自己的和（各占比之和=1）。
3. **边缘方向直方图（16 维）**：转灰度图，Sobel 算 x/y 梯度，转成幅值+角度，按角度分 16 桶、用幅值加权累加，再整体除以总和归一化。这部分捕捉"画面里边缘朝哪些方向"，是唯一带一点"形状"信息的部分。
4. 三部分拼成 256 维向量，整体做一次 L2 归一化（除以自身模长）。
5. 六个视图各自算出 256 维向量后**取平均**，平均完再做一次 L2 归一化，得到这个节点最终的 256 维指纹。

比对两个节点像不像，用的是**余弦相似度**（`node_backtracking.py` 里的 `_cosine`：两个向量点积 / 两个模长之积），因为两个向量已经是单位向量，点积本身就约等于余弦值。

**实测表格**（`source local_env.sh` 后用合成图跑 `CompactVisualEmbedder.embed`，非 GPU、非网络）：

| 场景 | 余弦相似度 | 说明 |
|---|---|---|
| 同一张纯灰色图 vs 自己 | 1.0000 | 符合预期，验证了归一化正确 |
| 纯灰色图 vs 六张随机噪声图 | **0.9656** | ⚠️ 出乎意料地高 |
| 纯红色图 vs 纯蓝色图 | 0.0395 | 接近正交，色调差异能被强烈区分 |
| 灰色图 vs 纯红色图 | 0.5620 | 中等 |
| 两张红蓝分界线，分界线错位 10 像素（320px 宽图上 160→170） | 0.9952 | 高，符合"轻微位移应该判定为很像" |

这张表说明一个重要事实：**这个指纹主要由"整体色调/平均亮度"主导，而不是细节纹理**。灰色图和随机噪声图的余弦相似度高达 0.97，是因为 8×8 缩略图（占 256 维里的 192 维、75% 的分量）用区域平均把噪声平均掉了，两者平均色都接近中灰；边缘直方图虽然对噪声图会有响应，但只占 16/256 维，权重太小压不过色调分量。**只有当两张图的整体色调/主体颜色明显不同时，这个指纹才能可靠地把它们分开**（红 vs 蓝这种）；同一个房间里换个角度、换点光影，指纹很可能仍然判定"像"。这是静态设计决定的行为，代码注释里也直说了这是"可以以后换成 DINO/CLIP/Qwen-VL 的占位实现"（`VisualEmbedder` 是个 `Protocol`）。

### 5.1.4 节点什么时候被创建：只在真正到达时，不是每次停下都记

原文说"执行器每次停下（不管成功失败）都会建节点"，逐一核对了全部四处在线调用点后，**这个说法是错的**，实际规则是：

- `scripts/rgb_only_instruction_sequence.py:432` 附近：`if not navigation.arrived:` 就直接进恢复/回溯分支或 `break`，**根本不会走到** `add_navigation_stop_node`（第 453 行）；只有 `navigation.arrived` 为真才会往下建节点。
- `scripts/instruction_sequence_exploration.py:2605` 附近：显式判断 `if physical_arrival:` 才调用 `add_navigation_stop_node`，注释原话是"A node represents a verified point-navigation arrival... never turn a premature off-screen signal into a node"。
- `scripts/habitat_point_navigation.py:1648`：同样 `if physical_arrival:` 才建节点。
- 回溯控制器（`rgb_only_instruction_sequence.py:234`）：`if navigation.arrived and similarity_after >= self.minimum_visual_similarity:` 才建节点——不仅要求物理到达，还要求视觉相似度达标。

也就是说 `_add_node` 内部虽然定义了 `node_kind` 会在 `arrival_signal` 为假时退化成 `"point_navigation_stop"`（区别于 `"point_navigation_arrival"`），但从这四处实际调用看，传进去的 `arrival_signal` 参数永远是执行器给出的信号值（`navigation.signal` / `navigation_result.signal`），而且都被 `if 已到达` 的条件卫护——**失败的尝试永远不会污染记忆图**，它们只留在 `record`/`target_record` 里供事后复核，绝不会变成一个节点或一条边。`arrival_signal` 具体在什么条件下可能是"到达了但信号是假值"这种边角情况，属于执行器（`point_navigation_executor.py`）内部逻辑，本节未深入核对，留给模块 4。

### 5.1.5 磁盘布局与持久化时机

```
<output_dir>/
  navigation_graph/
    navigation_graph.json        # 整个图的 JSON 快照（原子写：先写 .tmp 再 replace）
    nodes/
      node_0000/
        view_0_000deg.jpg ... view_5_300deg.jpg      # 6 张全景视图，60° 一张
        completion_view_0_000deg.jpg ... _7_315deg.jpg # 可选的 8 张判定专用视图，45° 一张
      node_0001/
        ...
```

每次 `add_origin_node` / `add_navigation_stop_node` / `set_sub_instruction_match` / `set_node_metadata` / `add_loop_closure_edge` 调用完都会立刻 `self.save()`——**图是"边跑边落盘"的，不是跑完才写**，中途崩溃也能保留已完成的部分。`save()` 用临时文件+原子 `replace()`，避免写一半就被读到。

起点节点（`add_origin_node`）只能在图为空时调用一次，`node_kind="episode_origin"`；这是整条图的根，之后所有节点都通过边追溯到它。`import_graph()` 是断点续跑用的：把上一次运行的 `navigation_graph.json` 连同已存的视图文件整体搬进新的输出目录，保持节点 id、历史边、keyframe 路径原样不变——代码注释特别强调"只重建位姿而不搬图会让续跑的节点被判定成一个假的新起点，让序列判定器把没真正走过的一跳当成走过"。

### 5.1.6 各种情况处理一览

| 情况 | 记忆图怎么处理 |
|---|---|
| 执行器没到达/回溯失败 | 不建节点、不建边，只留在 `record`/`attempts` 里 |
| 成功到达但视觉相似度不够（回溯场景） | 同样不建节点，继续下一次尝试 |
| 成功到达 | 建一个 `point_navigation_arrival`（或历史代码里的 `point_navigation_stop`）节点 + 一条边 |
| 视觉回溯真正找回目标节点 | 除了建新的到达节点/边，额外加一条 `edge_kind="node_revisit_loop_closure"` 的闭环边，把新节点和被回溯的旧节点标记为"同一个地方的两次访问" |
| 断点续跑 | `import_graph()` 整体复制旧图的节点、边、图片文件，不允许在非空图上再调用 |
| 查询不存在的节点/边 | `get_node`/`get_edge`/`get_edge_by_id` 抛 `KeyError` |
| 请求某条边的 keyframe 但没存 | `load_edge_keyframes` 抛 `RuntimeError` |
| 请求某节点的 8 视图判定全景但没存 | `load_node_completion_views` 抛 `RuntimeError` |
| `embed()` 传的视图数不是 6 | 抛 `ValueError`（已实测） |
| 追溯祖先路径时出现环 | `ancestor_path` 抛 `RuntimeError("cycle detected...")` |
| 追溯祖先路径超过 `max_hops` | 抛 `ValueError` |

### 5.1.7 下游谁在用

- `instruction_completion_judge.py`：读某节点的 `six_views`/`environment_semantics`/上一条边的 `action_history` 和 keyframe，判定这条边是否完成了对应子指令（模块 6）。
- `node_backtracking.py`：用 `visual_embedding` 做余弦相似度匹配，找"哪个存过的节点长得最像当前画面"，用 `ancestor_path`/`predecessor` 沿边物理走回去（模块 7）。
- `rgb_only_instruction_sequence.py` 的回溯控制器：拿目标节点的 `visual_embedding` 当"找回目标"的判据（5.1.4 已提到）。
- 评测/审计脚本（`validate_single_point_run.py`、`audit_rgb_only_contract.py`）：扫 `navigation_graph.json` 检查有没有 `position_xyz`/`base_yaw_rad`/`geodesic` 等违规字段混进去（见 8.1.4）。

### 5.1.8 已知的边角与坑

- **视觉指纹对纹理不敏感、对色调敏感**：5.1.3 的实测已经证明，灰色图和随机噪声图能有 0.97 的"相似度"；如果两个地点整体色调接近（比如都是白墙走廊），回溯匹配可能把它们混淆。这是设计上的已知局限，代码本身也把它标成"未来可换成神经网络 embedding 的占位实现"，不是 bug。
- **`traveled_distance_m` 在 RGB-only 下恒为 0**：`action_history` 里没有真实位移字段（RGB-only 契约不允许发出这类量），所以这个字段对 RGB-only 跑的边没有实际含义，只是历史 schema 留下的槽位；`audit_rgb_only_contract.py` 里专门写了一条豁免规则（`key == "traveled_distance_m" and child == 0.0` 不算违规），说明这是已知且被显式放行的情况。
- **节点是否被建取决于四处调用点各自的判断逻辑，不是记忆图模块自己决定的**：`NavigationGraphMemory` 本身没有"到达与否"的概念，完全信任调用方传进来的 `arrival_signal`；如果某个新调用点忘了加 `if 已到达` 的卫护，理论上就能建出一个污染节点。这属于调用约定而非类型约束，没有代码层面的强制。
- **`node_kind="point_navigation_stop"`（非到达）在当前四处在线调用中实际不可达**：因为调用前都已经用 `arrived`/`physical_arrival` 卫护过；这个分支目前更像是给历史/非 RGB-only 代码路径留的兼容口，但未在全部调用点穷举验证（未读 `point_navigation_executor.py` 内部是否存在"arrived=True 但 signal 为假值"的组合）。

---

# 模块 6：完成判定（NodeTransitionInstructionCompletionJudge）

**文件**：`scripts/instruction_completion_judge.py`、`scripts/instruction_completion_evidence.py`

**输入**：上一个节点（图 + 物体列表 + 指纹）、这条边（动作序列 + 5 张关键帧）、当前节点（同上）、当前子指令。
**输出**：**只有两个值**：`completed` 或 `unknown`，附置信度。注意它**从不说"你走错了"**——"unknown"只表示"我不能确认完成"，走错与否交给外层策略判断。

**内部逻辑**：

1. **整理证据**（`instruction_completion_evidence.py`）：把动作序列压缩成"前进 8 步 → 右转 15° → 前进 3 步，累计转角 +15°"这种摘要；把两个节点的检测结果做差（"上一节点看得见桌子，现在桌子在后方"）；算两个节点指纹的相似度。
2. **问 VLM**：给它上一节点的环视图、5 张关键帧、当前节点的 8 张 45° 图（标了 FRONT / FRONT_LEFT / LEFT ... 方位），加上上面的文字证据，问"'walk past the kitchen table'完成了吗"。V24 prompt 里有约 19 条规则，例如"turn ... to the doorway 时门框跨两个相邻方位也算完成"、"pass a room 视为拓扑上下文的转换"。
3. **VLM 回答之后还有一堆确定性闸门**（第 252-794 行）：比如 `TURN_TO_LANDMARK` 类型要求当前正前方真的对着地标；`BETWEEN_OBJECTS` 要求当前画面里两个参照物一左一右夹着；"来路方向与上一节点的出发方向夹角余弦 < −0.20"（明显走回头路）直接否决。这些闸门只能把 completed 拉成 unknown 或在少数明确情况下反向提升，并且都按类型写、不按具体物体写。
4. 置信度 ≥ 0.5 才接受。

**举例**：子指令"turn left at the end of the hallway"，上一节点在走廊起点，当前节点在走廊尽头。证据：动作是"前进 12 步 + 左转 6 次（−90°）"；上一节点前方检测到 hallway，当前节点 LEFT 方位检测到 hallway、FRONT 方位是新房间。VLM 说 completed，置信度 0.8；转向闸门检查当前正前方确实对着新方向 → 最终 `completed`。

## 6.1 扩充细节：判卷模块从头到尾讲透

> 本节 2026-09-11 补写。`instruction_taxonomy`/`instruction_completion_judge.py`/`vlm_harness.py` 相关函数逐字读过；`RGBOnlyNodeTransitionInstructionCompletionJudge` 的入参组装、`_strip_privileged_semantics`、`_keyframe_storyboard`、`validate()` 里"空动作边不算完成"的规则，以及 `tests/test_rgb_only_edge_judge.py` 的 4 个用例（关键帧数 1/5/6/7 都不崩溃、prompt 里能看到真实的转弯计数）本机用 `anchor_navi` 环境（纯 CPU、`HeuristicBackend`、不联网）实际跑过并全部通过；`judge_edge_instruction_completion_rgb_only` 里 DeepSeek 真实会怎么回答、三个 ADE20K 模型/DINO+SAM 的检测质量对判卷证据的影响，仍是静态阅读结论。行号以 2026-09-11 快照为准（`git show 2a3dbd9`、`87f1241` 已并入本节）。

### 6.1.1 一句话概括 + 比喻，以及一个重要更正

判卷模块回答一个极简的问题：**"从上一站走到这一站，这一段小任务算不算做完了？"**，只能回答"做完了"或"不确定"，从不说"你走错了"。

比喻：还是"下楼买酱油"那个例子。判卷模块就是那个不太会说话但很谨慎的旁观者——你回来了，他只会说"看起来你确实下楼了/确实拿着瓶子回来了"（completed），或者"我不确定你到底下没下楼"（unknown）；他绝对不会说"你走错方向了"，因为那已经超出他的职责——走错了怎么办是"外层策略"（模块 7）的事。

**必须先更正原文模块 6 的一个描述**：原文第 2-3 条说"VLM 回答之后还有一堆确定性闸门（第 252-794 行）……`TURN_TO_LANDMARK` 要求正前方对着地标、`BETWEEN_OBJECTS` 要求两个参照物一左一右、反向车道余弦否决……"——这段描述精确对应的其实是**另一个类**：`NodeTransitionInstructionCompletionJudge`（`instruction_completion_judge.py:140-848`，带 19 条 V13–V24 规则和十几道语义闸门）。经核实，`rgb_only_v1` 正式路径（唯一在 `habitat_point_navigation.py` 里被真正实例化的判卷类，见 `rgb_only_instruction_sequence.py:303`）走的是另一个类——`RGBOnlyNodeTransitionInstructionCompletionJudge`（`instruction_completion_judge.py:851-911`），它只有约 50 行逻辑、**没有上面任何一条闸门**。旧类连同它的 700 多行闸门代码，是 2026-09-09 RGB-only 重构之前的产物，目前只在离线评测/回放工具（如 `evaluate_instruction_completion_triplets.py`）和历史的 Round 021 成绩里生效，`rgb_only_v1` 在线循环碰不到它一行代码。下面 6.1.2 起只讲当前真正生效的 `RGBOnlyNodeTransitionInstructionCompletionJudge` + `judge_edge_instruction_completion_rgb_only`；旧类放在 6.1.9 作对照。

### 6.1.2 输入输出

**输入**（`instruction_completion_judge.py:863-864`）：`current_node`（刚建好的节点对象）、`expected_sub_instruction_id`（当前该完成第几条子指令）、`current_views`（这次判卷用的画面，生产环境一律是 8 张 45° 完成专用视图）、`edge_keyframes`（这条边上按步数抽的若干张按时间顺序排的画面，生产环境至少 1 张，通常 5 张，转弯跳会多 1 张"转弯前"帧）。

**输出**：一个 `InstructionCompletionResult`（`instruction_completion_judge.py:106-137`），公开只关心三个字段：`status`（`"completed"` 或 `"unknown"`）、`instruction_completed`（布尔）、`confidence`。其余字段（`endpoint_evidence`/`decision_gates`/`exit_endpoint_side_audit` 等）在 `RGBOnlyNodeTransitionInstructionCompletionJudge` 里**全是 `None`**——那些字段是旧类专用的，接口共用同一个 dataclass 只是为了让外层代码不用分两套读法。

### 6.1.3 `RGBOnlyNodeTransitionInstructionCompletionJudge.judge()` 逐步（`instruction_completion_judge.py:863-911`）

1. **找上一站**：`graph_memory.predecessor(current_node.node_id)`——在记忆图里找"指向这个节点的那条边"，边的起点就是上一站（`navigation_graph_memory.py:495-500`）。如果这个节点没有任何入边（比如它是起点），返回 `(None, None)`，`judge()` 就 `raise ValueError`。**实测结论**：生产调用路径（`rgb_only_instruction_sequence.py:470-472`）只在 `graph_memory.add_navigation_stop_node()` **成功建好一条新边之后**才调用 `judge()`，所以这条 `ValueError` 在当前唯一的调用点上永远不会触发——它是防御性代码，为"以后可能有别的调用方在起点上误调 `judge()`"兜底，不是当前会发生的真实分支。
2. **要几张上一站的图**：`use_eight = len(current_views) == 8`。生产环境 `current_views` 永远是 `observe_eight_rgb()` 拍的 8 张（`rgb_only_instruction_sequence.py:452`），所以永远走 `load_node_completion_views`（读上一站存的 8 张 45° 图），不会走六视图分支——六视图分支只在单测里被直接构造 `views=6` 时触发（`tests/test_rgb_only_edge_judge.py:89-91`）。
3. **调 VLM**：`self.vlm_harness.judge_edge_instruction_completion_rgb_only(...)`，把当前子指令、下一条子指令（仅供参考，防止"抢跑"）、上一站/这一站的节点 id、8 张图、边上的动作历史、边关键帧、两站的环境语义一并传过去。**没有传** `position_xyz`、`base_yaw_rad`、`visual_embedding`——这三个字段是旧类才传的（对比 `judge()` 第 222-245 行 vs 878-888 行），新类的函数签名里根本没有这几个参数位置，不是"传了但没用"，是**接口上就不存在**这条泄漏通道。
4. **最后一道锁——物理到达信号**（第 889-892 行）：
   ```python
   accepted = bool(
       raw["status"] == COMPLETED and
       float(raw["confidence"]) >= self.minimum_confidence and
       current_node.arrival_signal)
   ```
   VLM 说"完成了"还不够，还要求这个节点确实带着一个非空的 `arrival_signal`（执行器物理到达时打上的标记，见模块 4/5）。**实测结论**：在当前唯一的生产调用路径里，只有 `navigation.arrived` 为真时代码才会往下走到建节点、调 `judge()`（`rgb_only_instruction_sequence.py:429-432` 的 `if not navigation.arrived: ... continue`），所以传进来的 `current_node` 永远带着 `arrival_signal`。这一条和上一条的 `ValueError` 一样，在当前调用点上是**双保险而非活跃闸门**——但如果哪天新代码在还没物理到达时就调 `judge()`，它会立刻把结果摁成 `unknown`，不会被绕过。

### 6.1.4 `judge_edge_instruction_completion_rgb_only` 内部逐步（`vlm_harness.py:7563-7666`）

**第一步：入口检查**（第 7601-7606 行）：`previous_views`/`current_views` 长度必须相等且在 `{6, 8}` 之内，否则 `ValueError("RGB-only completion requires matching 6- or 8-view panoramas")`；`edge_keyframes` 不能是空列表，否则 `ValueError("RGB-only completion requires chronological RGB keyframes")`。这两个 `ValueError` 在生产路径上没有 `try/except` 包着（`rgb_only_instruction_sequence.py` 里 `completion_judge.judge(...)` 这一行是裸调用），一旦真的触发会像附录 B 讲的"No floor-bearing candidate"一样让整个 episode 进程崩溃、拿不到 `trajectory.json`。**这是静态阅读结论、未在真实数据上复现**：目前看不出生产路径上什么情况会传出空 `edge_keyframes`——执行器至少给 5 张关键帧，有转弯时 `prepend_turn_start_keyframe` 还会再加 1 张，`current_views`/`previous_views` 也都固定走 8 张。这条"入口检查会崩进程"的风险目前是理论性的。

**第二步：动作历史去敏 + 压缩摘要**（第 7607-7621 行）。逐条动作记录只保留 7 个字段：`step`、`action`、`commanded_turn_deg`、`forward_commanded`、`rgb_motion_score`、`orientation_only`、`phase`——凡是名字里带 `position`/`pose`/`yaw`（全局朝向）之类的字段一律不保留（这一步没有名单，是白名单机制：不在保留名单里的字段直接被丢弃）。然后统计出 `action_summary`：`control_steps`（总步数）、`forward_command_count`/`left_turn_command_count`/`right_turn_command_count`（三种动作各多少次）、完整的 `chronological_actions` 列表、外加一句提示 VLM 的话："Commands and RGB-change scores only; no pose, metric distance, depth, collision or navmesh feedback."

**第三步：环境语义去敏**（`_strip_privileged_semantics`，第 7546-7562 行）：递归遍历上一站/这一站的检测语义字典，凡是键名在 `{median_depth_m, depth, depth_m, position_xyz, pose, yaw_rad, absolute_yaw_rad, geodesic_distance_m, navmesh, world_xyz}` 里的整个键值对删掉，其余原样保留。这是模块 2.1.5 提到的"`SemanticDetection.median_depth_m` 是死字段"在判卷这一侧的对应清理动作——即使上游没删干净，这里会兜底删一遍。

**第四步：拼图**（第 7614-7618 行）：`panorama_sheet = _completion_contact_sheet if 8 张 else _contact_sheet`——生产环境固定 8 张，用 `_completion_contact_sheet`（`vlm_harness.py:2121-2142`），4×2 排布，每张图左上角贴黑底白字标签 `VIEW 0 FRONT 0deg` … `VIEW 7 FRONT_RIGHT -45deg`，这样 VLM 不用自己数第几张图是哪个方向。关键帧用 `_keyframe_storyboard`（第 2160 行起），**不要求固定张数**，每张缩到 320×240、贴 `KF i (chronological)` 标签，最多 3 列自动换行拼接。三张图按固定顺序发给 VLM：`[上一站 8 视图, 时序关键帧拼图, 这一站 8 视图]`。

**第五步：发送的 prompt 要点**（第 7619-7635 行，原文见下）：

> Decide whether the REAL observed transition from the previous node, through the chronological RGB keyframes, to the current node COMPLETED the active sub-instruction. The alternative UNKNOWN includes still on the way, wrong direction, blocked/stationary, ambiguous identity, or insufficient evidence. … Use visible semantic and temporal change. Do not infer metric distance, elevation, collision, global heading, pose or map structure. Seeing a landmark without reaching the full semantic spatial relation is UNKNOWN.

翻成人话：**"只凭画面里看得见的东西和三张图的时间先后关系判断，不许脑补距离、高度差、碰撞、绝对朝向、地图结构；光是看见地标但还没达到指令要求的那种空间关系（比如'经过'要求地标在身后），也算没完成。"** 这段话和 `POINT_SCHEMA`/选点 prompt 里"禁止使用深度/3D/navmesh"是同一条项目规则在判卷侧的落地。

**第六步：返回结构校验 `validate()`**（第 7660-7666 行）：`status` 必须是 `"completed"`/`"unknown"` 之一，`confidence` 必须在 `[0,1]`，否则 `ValueError` 触发 `_call` 的重试机制（和模块 1 的拆指令、模块 2/3 的选点用的是同一套"报错→重问→最多 3 次→`RuntimeError`"逻辑，见 1.1.4）。**唯一一条确定性后处理规则**（第 7666-7668 行）：

```python
if status == "completed" and not sanitized_actions:
    status = "unknown"
    confidence = min(confidence, 0.49)
```

**空动作边不能算完成**——如果这条边上一步动作记录都没有（理论情况，正常边至少有若干前进/转弯记录），VLM 却说完成了，代码强行把它压成 unknown、置信度封顶 0.49。这是当前 `judge_edge_instruction_completion_rgb_only` 里**唯一**的确定性闸门，和旧类里十几条按类型（PASS/BETWEEN/STOP_WAIT/TURN_TO_LANDMARK…）写的闸门完全不是一回事，作用范围也小得多——它只管"有没有发生过至少一次动作"，不管方向对不对、地标在哪个方位。

### 6.1.5 转弯类子指令怎么被判卷看见（`git show 2a3dbd9`，2026-09-11 当天修的 bug）

这是一个刚修好、值得单独讲的真实案例，因为它精确示范了"6.1.4 第二步的动作摘要哪里来的"这件事有多脆弱。

**背景**：选点模块（模块 3）选中一个视图之后，会调 `continuous_turn` 把机器人原地转向那个视图对应的朝向，这几步 `turn_left`/`turn_right` 命令**只写进了 `motion_log`**（供录像/调试用的旁路日志），没有写进执行器的 `action_history`——执行器的动作历史是从"转完之后开始走"才计数的。后果：对于 `TURN_LEFT`/`TURN_RIGHT`/`TURN_TO_LANDMARK` 这类"指令本身就是转弯"的子指令，判卷模块拿到的 `action_summary` 里 `left_turn_command_count`/`right_turn_command_count` 永远是 0——机器人明明转了，判卷看到的动作记录里却一次转弯都没有。2026-09-11 的一轮 OpenNav100 评测里，**90 个 TURN_* 判定中有 78 个被判 unknown**，原因就是这个。

**修法**（`rgb_only_instruction_sequence.py` 新增 `turn_to_selected_target_actions`/`prepend_turn_start_keyframe`）：在选点转向发生的那一段 `motion_log` 里，把 `phase == "turn_to_selected_target"` 且属于当前这一跳的 `turn_left`/`turn_right` 记录捞出来，转换成和执行器自己的动作记录同样格式的条目（`step` 用负数排在最前面、`orientation_only=True`、`forward_commanded=False`），拼接到 `action_history` 最前面一起交给判卷；同时把"转弯开始前"那一帧画面存成关键帧 0（`prepend_turn_start_keyframe`），让时序拼图里能看到"转之前"和"转之后"的对比。八视图的探测性转动（refinement probe，转出去看一眼又转回来的那种）明确被排除，不计入这段历史。

**本机实测**（`tests/test_rgb_only_edge_judge.py::test_turn_to_view_actions_are_counted_and_kept_in_prompt`，真实跑过、通过）：构造 6 条 `turn_right` 记录 + 2 条 `move_forward` 记录喂进去，最终发给 VLM 的 prompt 文本里确实出现 `"right_turn_command_count": 6`、`"forward_command_count": 2`、`"control_steps": 8`，说明修复后转弯确实进了判卷的证据链。

### 6.1.6 HeuristicBackend（假 VLM）怎么判卷

`vlm_harness.py:1069-1077`：只要 prompt 里含 `"You are the RGB-only edge completion judge"` 这句话，直接返回固定结果：`status="unknown", confidence=0.5, reason="heuristic rgb-only edge judge test backend"`。注释原话："A rule stub cannot see semantic change, so it never claims completion; this keeps key-free smoke runs alive past arrival."——翻译：这个假后端**永远不会说"完成了"**，只是为了让不带 key 的冒烟测试能跑过"物理到达"这一步而不崩，不代表它验证了判卷逻辑本身。**实际含义**：任何用 `--vlm-backend heuristic` 跑的冒烟测试，子指令游标永远不会推进（见模块 7），也就永远不会走到"最后一条子指令 completed → 主动 STOP"这条路；heuristic 冒烟只能验证到"能不能正常走到第一次物理到达+判卷不崩"，验证不了 STOP 触发链路。这与 1.1.4/2.1 节 HeuristicBackend 的"只给测试用、不代表真实拆分/检测行为"是同一类限制。

### 6.1.7 各种情况处理一览

| 遇到的情况 | 系统怎么办 | 在哪 |
|---|---|---|
| 当前节点没有入边（起点上误调判卷） | `ValueError`，理论分支，生产调用点摸不到 | `instruction_completion_judge.py:871-872` |
| `current_views` 是 8 张 | 走 `load_node_completion_views` 取上一站 8 张图 | `instruction_completion_judge.py:873-877` |
| `current_views` 是 6 张（只有单测会这样构造） | 走 `load_node_views` 取上一站 6 张图 | 同上 |
| VLM 说 completed 但这条边动作记录为空 | 强制改 unknown，置信度封顶 0.49 | `vlm_harness.py:7666-7668` |
| VLM 说 completed、有动作记录，但节点没有 `arrival_signal`（理论分支） | 强制 unknown | `instruction_completion_judge.py:889-892` |
| VLM 返回的 `status` 不是 completed/unknown，或 `confidence` 越界 | 重问，最多 3 次；仍失败 `RuntimeError` | `_call`，同 1.1.4 |
| 关键帧张数是 1、5、6、7 甚至其它任意正整数 | 都能正常拼图，不再要求恰好 6 张（已修复的旧 bug，见附录 A） | `_keyframe_storyboard` |
| `previous_views`/`current_views` 张数不在 {6,8} 或两边不等长 | `ValueError`，无 try/except，会让 episode 进程崩溃（理论风险，未观测到真实触发） | `vlm_harness.py:7601-7603` |
| `edge_keyframes` 为空列表 | 同上，`ValueError` | `vlm_harness.py:7604-7606` |
| 转弯类子指令（TURN_LEFT/RIGHT/TO_LANDMARK） | 选点时的原地转向命令会被补记入边的动作历史和一张"转前"关键帧，否则判卷看到的转弯次数永远是 0 | `rgb_only_instruction_sequence.py`（2a3dbd9） |
| 用 `--vlm-backend heuristic` 冒烟 | 永远返回 unknown，游标不会推进，测不出 STOP 链路 | `vlm_harness.py:1069-1077` |
| 环境语义里混进了深度/坐标字段 | 判卷侧再兜底删一遍，不管上游删没删干净 | `_strip_privileged_semantics` |
| 下一条子指令的信息 | 只作为"不要抢跑"的参考传给 VLM，不参与判定当前这条 | `vlm_harness.py:7625` prompt 原文 |

### 6.1.8 下游谁在用

- `instruction_completed`（布尔）：外层状态机（模块 7，`rgb_only_instruction_sequence.py:474-477`）用它决定 `belongs_to_sequence`，进而决定子指令游标是否 `+1`；游标推到最后一条且 completed → 主动发 STOP（模块 8）。
- `status`/`confidence`：写进每一跳的 `record`（最终落进 `trajectory.json`），供 `verify_round_stage_completions.py` 等事后审计脚本按边复核。
- 连续 `unknown` 次数：交给模块 7 的状态机计数，达到阈值（RGB-only 下是连续 2 次）触发物理回溯（`RGBOnlyGraphBacktracker`，见 7.1.3），和判卷模块本身无关——判卷只管"这一次的信号是什么"，不管"连续几次该怎么办"。

### 6.1.9 与旧类 `NodeTransitionInstructionCompletionJudge` 的对照（历史代码，仅供理解 Round 021 成绩）

旧类（`instruction_completion_judge.py:140-848`）除了三态输出（`completed`/`on_route`/`unknown`，`ARRIVED`/`ON_ROUTE` 常量）之外，`judge()` 里还有一长串按 `form` 分支的确定性覆盖/否决规则，例如：

- `TURN_TO_LANDMARK` 的"当前 RGB 朝向对齐见证"覆盖（`current_rgb_turn_landmark_alignment_supported`，第 61-103 行）；
- `PASS_LANDMARK`/`ADVANCE_STRAIGHT` 的"正前方还看得见地标就不能算过去了"闸门（第 276-305 行）和"新节点几何上在来路后方就否决"闸门（第 315-352 行，用两个节点的 `position_xyz` 算余弦，**这一条直接依赖坐标**）；
- `BETWEEN_OBJECTS` 的"下一条地标是否新出现"覆盖（第 477-624 行）和"路线完整性"否决（第 629-635 行）；
- `EXIT_REGION`/`ENTER_REGION`/`SELECT_PORTAL`/`TRAVERSE_PORTAL_REGION` 的"结构化穿越见证"覆盖（第 644-695 行）；
- `STOP_WAIT` 的"地标面积增长"覆盖（第 704-777 行）。

这些规则全部**只存在于旧类**，`RGBOnlyNodeTransitionInstructionCompletionJudge` 没有继承、没有复用、也没有调用它们一行。旧类目前在 `habitat_point_navigation.py` 的正式入口里**没有任何实例化点**（唯一出现的地方是 `evaluate_instruction_completion_triplets.py` 这类离线复核脚本）；Round 021（2026-09-08）跑出的 5/10，用的正是这个旧类 + 依赖坐标/深度的一整套闸门，这一点附录 A 已经用代码证据讲清楚，这里不重复。原文模块 6 的行文风格（"VLM 回答之后还有一堆确定性闸门"）读起来像是在描述当前生效的判卷逻辑，**实际描述的是这个历史类**，本节 6.1.1 已经把这处更正标出来。

### 6.1.10 已知边角与坑

- `instruction_completion_evidence.py` 里的 `summarize_motion`/`summarize_semantic_transition`/`summarize_visual_transition` 三个"结构化证据"函数虽然在 `vlm_harness.py` 顶部被 import，但实测调用点全部在旧类背后的 `judge_edge_instruction_completion`（第 7881、7893 行）里，`judge_edge_instruction_completion_rgb_only` 一次都没调用它们——这个证据整理文件对 `rgb_only_v1` 在线路径完全不起作用（用 `grep` 核实，未见反例），是旧契约的残留依赖。
- `judge()` 里两处 `ValueError`（无入边、面板张数不符）在当前唯一调用点上都是不可达的防御代码，但它们**没有**被包一层 `try/except` 转成"优雅结束 episode"——如果以后有新调用方在不满足前提条件时调用它们，行为会是"进程崩溃"而不是"这条 episode 判 unknown"，这和附录 B 里"No floor-bearing candidate"曾经的表现是同一种模式，值得在新增调用点时留意。
- 判卷的唯一确定性规则（空动作边不算完成）粒度很粗，只挡得住"完全没动过就说完成"这种极端情况；它挡不住"动了但动错方向"，这类问题在旧类里有专门的余弦否决闸门，新类完全没有对应机制，全靠 VLM 自己看图判断方向对不对。这是设计上的取舍（不读坐标就不可能算余弦），但意味着方向性错误更依赖 VLM 本身的视觉判断质量。
- `following_sub_instruction` 只是"不要抢跑"的上下文参考，prompt 里明确说"do not complete it early"，但这只是文字提示，没有代码层面的强制——如果 VLM 没听话提前判了下一条完成，目前没有确定性规则拦截这种情况。

---

# 模块 7：外层策略——完成了怎么办？没完成怎么办？

**文件**：`scripts/rgb_only_instruction_sequence.py`（`run()`，第 255 行）、`scripts/instruction_sequence_exploration.py`（状态机 `observe()`，第 920 行）、`scripts/node_backtracking.py`

这是把上面所有模块串起来的"总指挥"，用一个状态机决定下一步：

```
for 每一跳 (最多 30 跳):
    选点 → 执行器走
    if 没走到（跟丢/超步）:
        回溯到上一个"可信节点"，把这个方向拉黑，重来
    else:
        建节点 → 判卷
        completed        → 游标 +1；是最后一条 → 结束并 STOP
        unknown (第 1 次) → 再往前探一跳（"也许走了一半还没完成"）
        unknown (连续 2 次) → 物理走回分叉起点的那个节点，拉黑这个方向
```

- 每个节点最多拉黑 5 个方向，都黑了就放弃（`all_candidate_directions_blocked_at_verified_node`）。
- 上下楼梯/出房间这类天然需要多跳的类型允许更多次"on_route unknown"（4 次 / 3 次）。

**回溯怎么在纯 RGB 下"走回去"**（`RGBOnlyGraphBacktracker.recover`，第 101 行）：拿当前 6 张图算指纹，和目标节点存的指纹比余弦相似度；≥ 0.75 就算已经在那儿了；否则问 VLM"朝哪个方向走能回到这张参考图的位置"，用执行器走一跳，再比一次，最多试 4 次。成功后在图里加一条"闭环"边。

**举例**：从桌子旁边选了个点走到一个储物间，判卷说 unknown；再探一跳还是 unknown。策略决定回溯到 node_0001（桌子旁），VLM 对着 node_0001 的六视图指方向，走回去后相似度 0.82 → 回溯成功；把"储物间方向"拉黑，下次选点时那个方向的视图不给 VLM 看。

## 7.1 扩充细节：外层策略——完成了怎么办？没完成怎么办？

> 本节写于 2026-09-11。标注「✅ 实测」的内容是本次编写过程中直接运行 `InstructionSequenceStateMachine`（`scripts/instruction_sequence_exploration.py:843`）得到的真实状态转移结果（见 7.1.4，用手工构造的 `classification` 字典驱动，不依赖 GPU/网络）；标注「📖 静态读码」的内容是逐行读 `scripts/rgb_only_instruction_sequence.py`（536 行）、`scripts/instruction_sequence_exploration.py` 前 1550 行、以及 `git show 272c223` / `git show 2a3dbd9` 两个真实 diff 得出的结论，尚未跑端到端 episode 验证。凡本节行号，均以本次读取时的文件快照为准；原文标注的 `run()` 第 255 行、`RGBOnlyGraphBacktracker.recover` 第 101 行已过期——两次最近的修复提交往文件前部插入了新函数，现在 `run()` 在第 318 行、`recover()` 在第 164 行（类本身从第 120 行开始）。**原文里"最多试 4 次回溯"和"上下楼梯/出房间允许 4 次/3 次 on_route unknown"两处说法经核对对当前 rgb_only 路径不成立，见 7.1.7；原文保留不改。**

### 7.1.1 一句话 + 类比

外层策略是整个导航流程的「项目经理」：它不管怎么走路（那是模块 4 执行器的事），只管两件事——**这一跳完成了没有**（读模块 6 判定给的 `completed`/`unknown`），以及**没完成该怎么办**（再看一眼、还是往回走）。

类比一次考试的多道题：每道题（子指令）只能按顺序做，做完一道才能翻到下一道。如果监考老师（判定模块）说「这道看起来没做完」，项目经理不会立刻判你不及格——会先让你「再看一眼、再检查一步」（`explore_once_more`，多给一跳机会）；但如果**连续两次**都说没做完，项目经理就认定你走错片区了，喊你「退回上一个签到点，并且把刚才走的这个方向拉黑，别再往那边走」（`backtrack_and_block`）。同一个签到点被拉黑满 5 个方向（相当于房间四面墙都试过了都不通），就直接判定这一关卡死，任务结束。

### 7.1.2 输入输出

`RGBOnlyInstructionSequenceExplorationStrategy.run()`（`rgb_only_instruction_sequence.py:318`）的输入输出：

- **输入**：`initial_action_heading`（浮点弧度，动作累计朝向的起点，默认 0.0）、`initial_global_step`（int，全局步数计数起点）；构造函数还需要 `graph_memory`（已含第 0 个起点 node）、`point_selector`、`point_navigation_executor`、`completion_judge`、`vlm_harness`、`segmenter`、`sim`（必须是 `RGBOnlyPolicySimulator` 包装句柄，`require_rgb_only_policy_sim` 会在构造时校验，不满足就抛错，fail-closed）。
- **输出**：`InstructionSequenceExplorationResult`（`instruction_sequence_exploration.py` 定义，RGB-only 侧复用同一个 dataclass），关键字段 `success`（= `self.state.complete`）、`end_reason`、`hops`（实际执行的跳数）、`state_snapshot`（`InstructionSequenceStateMachine.snapshot()`，含 `cursor`/`off_sequence_nodes`/`blocked_yaws`/`terminated_reason` 等）。这个 `end_reason` 字符串是后续模块 8（评测脚本）唯一用来分类失败原因的信号，见 7.1.6。

`InstructionSequenceStateMachine.observe(node_id, classification, selected_yaw)`（`instruction_sequence_exploration.py:920`）的输入输出：

- **输入**：`node_id`（这一跳新建的 node id）、`classification`（dict，RGB-only 侧由 `run()` 在第 ~470 行左右手工拼出，字段只有 `matched_sub_instruction_id`、`belongs_to_sequence`、`confidence`、`unknown_disposition`（**RGB-only 恒为 `None`**，见 7.1.7）、`active_form`）、`selected_yaw`（这一跳选点时用的动作朝向，用于失败时拉黑方向）。
- **输出**：`SequenceDirective`（`action` 取值见 7.1.3 状态转移表；`backtrack_target_node_id`；`direction_to_block_yaw_rad`；`reason`）。

"朝向"在 RGB-only 下是什么：不是仿真器的 yaw，而是**自己发过的转向动作累加出来的"动作朝向"**（每次 `turn_left/right` ±15°），起点是 `initial_action_heading`。拉黑方向、回溯时构造候选视图的 60° 偏移，全部在这个动作坐标系里算。

### 7.1.3 逐步流程：`run()` 主循环 + `observe()` 状态机

**`run()` 主循环**（`rgb_only_instruction_sequence.py:318`），`for hop_index in range(self.max_exploration_hops)`，默认 `max_exploration_hops=30`。进循环前 `end_reason` 的初始值是 `"max_sequence_exploration_hops"`（第 326 行）——也就是说如果 30 跳跑完中途没有任何分支改写它，最后的 `end_reason` 就是这个：

1. 先查 `self.state.complete` 或 `self.state.terminated_reason` 是否已经非空——已完成或已终止就直接跳出循环，不再多走一跳（后者会把 `end_reason` 设成 `terminated_reason`，例如 `all_candidate_directions_blocked_at_verified_node`）。
2. 取当前子指令 `stage = sub_instruction.to_stage_dict()`；调用点选择器 `self.point_selector.select(...)` 之前先记一帧 `turn_start_rgb = observe(self.sim)`（原地转向前的六视图快照，供后面补写 keyframe 用）。
3. **选点失败分支（📖，272c223 修的）**：`select()` 被 `try/except RuntimeError as exc` 包住——如果地面 mask 没有任何「贴地」候选点（异常信息里写的是 `"No floor-bearing candidate..."`），会走到 `except` 分支，`end_reason = f"vlm_selection_failed: {exc}"`，`break` 掉整个循环，episode 正常结束而不是进程崩溃。若 `select()` 正常返回但 `chosen is None`（VLM 没选出任何点），`end_reason = "rgb_vlm_selection_returned_no_ground_point"`，同样 `break`。
4. **原地转向记录（📖，2a3dbd9 修的）**：`turn_to_selected_target_actions()` 把选点器为了对准目标而做的原地转身，从 `motion_log` 里筛出 `phase == "turn_to_selected_target"` 的记录，转成带负数 step 号的合成动作记录（负数保证排在执行器自己 `0..N` 步之前）；随后 `execute_point_navigation` 跑真正的点导航；最终 `action_history = turn_actions + list(navigation.action_history)`。如果 `turn_actions` 非空，还会把第 2 步存的 `turn_start_rgb` 通过 `prepend_turn_start_keyframe()` 塞进这条边的 keyframe 0（其余 keyframe 序号整体后移一位）。
5. **物理未到达分支**：`navigation.arrived` 为假（执行器判定没走到目标点附近），直接调用 `self.backtracker.recover(self.state.last_verified_node_id, ...)`（目标是回到"最后一个确认过的节点"，而不是任意节点）；失败就 `end_reason = "rgb_only_physical_failure_recovery_failed"` 并结束；成功则调用 `self.state.block_failed_physical_direction(selected_heading)` 把刚才失败的朝向拉黑，然后 `continue` 进入下一跳——注意这个分支**不调用** `self.state.observe()`，因为压根没有新节点产生，不涉及语义判定。
6. **物理到达分支**：新建一个 stop node，调用 `self.completion_judge.judge(...)`（模块 6 简化版判定），拼出 `classification` 字典（`unknown_disposition` 恒为 `None`，见 7.1.7），调用 `directive = self.state.observe(stop_node.node_id, classification, selected_heading)`，再按 `directive.action` 分支处理：
   - `"backtrack_and_block"` → 调 `self.backtracker.recover(directive.backtrack_target_node_id, ...)`；失败则 `self.state.on_backtrack(False)`，`end_reason = "rgb_only_sequence_recovery_failed"`，结束；成功则 `self.state.on_backtrack(True, current_node_id=recovery.recovered_node_id)`，继续下一跳。
   - `"complete"` → `end_reason = "instruction_sequence_complete"`，`break`（这是唯一的成功终止路径）。
   - 其余（`"advance_sequence"` / `"continue_current_instruction"` / `"explore_once_more"`）→ 不终止，继续下一跳。

**`InstructionSequenceStateMachine.observe()` 状态转移**（✅ 实测，见 7.1.4 的完整代码与输出）：

| 触发条件 | 动作 (`directive.action`) | 说明 |
|---|---|---|
| `matched_sub_instruction_id == expected_sub_instruction_id` 且 `belongs_to_sequence=True`、置信度达标 | `advance_sequence`（若是最后一条子指令则为 `complete`） | `cursor += 1`，`consecutive_on_route_unknowns` 清零 |
| 第一次「不属于当前子指令」（`belongs_to_sequence=False` 且非 on_route） | `explore_once_more` | 记入 `off_sequence_nodes`，允许再走一跳观望 |
| **连续第二次**「不属于当前子指令」 | `backtrack_and_block` | `backtrack_target_node_id` 指向上一次 `off_sequence_nodes` 记的那个节点；`direction_to_block_yaw_rad = selected_yaw` |
| `unknown_disposition == "on_route"` 且未超过该 `active_form` 的连续预算 | `continue_current_instruction` | `consecutive_on_route_unknowns += 1`；预算见下方 |
| `on_route` 预算耗尽 | `backtrack_and_block` | 同上 |

`consecutive_on_route_unknowns` 的预算按 `active_form` 分级（`VERTICAL_UP`/`VERTICAL_DOWN` 上限 4，`EXIT_REGION` 上限 3，其余默认 1——实测见 7.1.4 的 Case D，V0–V3 都返回 `continue_current_instruction`，第 5 次即 V4 才转 `backtrack_and_block`，`consecutive` 停在 4 而非清零）。**但这条分级预算在当前 `rgb_only_v1` 生产路径下是不可达的死代码**（见 7.1.7）——因为触发它的前提 `unknown_disposition == "on_route"` 需要靠 `infer_unknown_disposition()` 这个只吃「旧版丰富判定证据」的函数计算，而 `rgb_only_instruction_sequence.py` 拼 `classification` 时硬编码 `"unknown_disposition": None`，永远不等于 `"on_route"`。因此 RGB-only 路径下每一个 `unknown` 实际只会走「非 on_route」分支：第一次给一次 `explore_once_more`，第二次必 `backtrack_and_block`——原文"上下楼梯/出房间允许 4/3 次"的说法对当前生产路径不成立，只对旧版 `InstructionSequenceExplorationStrategy`（未加 `-rgb-only` 后缀那个）成立。

`block_failed_physical_direction(yaw)`（`instruction_sequence_exploration.py:904`）与 `_add_block`（896）：把 `yaw` 记到当前 `verified` 节点的拉黑列表，20° 以内视为同一个方向去重；累计满 `max_blocked_directions_per_node`（默认 5）个不同方向后，`terminated_reason = "all_candidate_directions_blocked_at_verified_node"`，任务判定卡死结束（✅ 实测见 7.1.4 Case E）。

**`RGBOnlyGraphBacktracker.recover()`**（📖，`rgb_only_instruction_sequence.py:164`）：

1. 先拍当前六视图全景，算 256 维视觉指纹（`CompactVisualEmbedder`），跟目标节点存的指纹算余弦相似度；**不动一步**就 ≥ `minimum_visual_similarity`（默认 0.75）的话，直接判定 `"target_panorama_reobserved"` 成功返回。
2. 否则循环最多 `max_attempts` 次（构造函数默认 **2**，来自 `RGBOnlyInstructionSequenceExplorationStrategy.__init__` 的 `recovery_backtrack_attempts_per_hop=2` ——⚠️ 这与原文写的「最多 4 次」不符：4 是**旧版** `InstructionSequenceExplorationStrategy.__init__` 的默认值，旧版走的是 `NodeBacktrackingController`（见 7.1.7），跟 RGB-only 这条自包含的 `RGBOnlyGraphBacktracker` 是两套独立实现）：以当前动作朝向为基准，构造 6 个候选视图（偏移 60°/120°/…/300°，注释明确写着 "Action-frame heading, never simulator/world yaw"），让 VLM（`select_backtrack_ground_target_rgb_only`）挑一个视图+一个地面点，执行一次点导航朝那边走一跳，再重新算相似度；达标就成功退出。
3. 成功后新建一个 stop node（`add_navigation_stop_node`），再加一条闭环边回到目标节点（`add_loop_closure_edge`，`metadata.verification="rgb_panorama_similarity_only"`）——这条边的可信度标注得很明确：只是视觉相似度核验，不是几何核验。

结合 5.1.3 的实测（灰图与噪声图的指纹相似度都有 0.97），0.75 这个阈值在色调相近的室内环境里很容易被"错误的地方"满足，回溯成功不等于真的回到了原地。

### 7.1.4 实测例子（✅ 实际运行过）

用手工构造的 `SubInstruction`/`classification` 直接驱动 `InstructionSequenceStateMachine`（无需 GPU/网络/Habitat），得到的真实输出：

```python
sm = InstructionSequenceStateMachine(subs, "node_origin", max_blocked_directions_per_node=5,
                                      minimum_classification_confidence=0.5)
# Case A：正确匹配当前子指令 -> 推进序列
sm.observe("node_A", {"matched_sub_instruction_id": 0, "belongs_to_sequence": True,
                       "confidence": 0.9, "unknown_disposition": None, "active_form": "PASS_LANDMARK"},
           selected_yaw=0.1)
# -> action="advance_sequence", cursor: 0 -> 1

# Case B：连续两次「不属于当前子指令」-> 第 2 次触发回溯拉黑
# B1 -> action="explore_once_more"
# B2 -> action="backtrack_and_block", backtrack_target=B1 节点, block_yaw=0.3

# Case C：on_route unknown，PASS_LANDMARK 默认预算=1 -> 第 2 次就回溯
# C1(on_route) -> "continue_current_instruction"；C2(on_route) -> "backtrack_and_block"

# Case D：VERTICAL_UP 预算=4 -> 连续 5 次 on_route unknown 才在第 5 次回溯
# V0..V3 -> "continue_current_instruction"（consecutive 1..4）；V4 -> "backtrack_and_block"

# Case E：5 次方向拉黑 -> 终止
for i in range(5):
    sm.block_failed_physical_direction(1.0 * i)
# terminated_reason == "all_candidate_directions_blocked_at_verified_node"
# blocked_yaws() == [0.0, 1.0, 2.0, 3.0, 4.0]
```

Case D 的实测结果直接证明了「按 `active_form` 分级预算」这段逻辑本身是存在且工作正常的——问题不在这段代码有没有 bug，而在于 RGB-only 的调用方永远不会传出 `unknown_disposition == "on_route"`，导致这段代码在生产路径下形同虚设（见 7.1.3 结尾、7.1.7）。

### 7.1.5 边界情况对照表

| 情况 | 触发位置 | 处理结果 |
|---|---|---|
| 选点找不到任何贴地候选点（`RuntimeError`） | `run()` 第 3 步，`point_selector.select()` | `end_reason = "vlm_selection_failed: ..."`，`break`，episode 优雅结束（272c223 修复前会让整个进程崩溃，2026-09-11 某轮 100 条里影响 75 条） |
| VLM 没选出任何点但没报错 | `run()` 第 3 步，`chosen is None` | `end_reason = "rgb_vlm_selection_returned_no_ground_point"` |
| 选点器原地转向的动作没有写进边的 `action_history` | `run()` 第 4 步 | 2a3dbd9 修复前，判定模块看不到转身动作会以为「零转身指令」而拒绝 `TURN_*` 类子指令，某轮 90 个 `TURN_*` unknown 里 78 个是这个原因；修复后靠 `turn_to_selected_target_actions` + `prepend_turn_start_keyframe` 补全 |
| 执行器判定没走到目标点（`navigation.arrived=False`） | `run()` 第 5 步 | 直接回溯到 `last_verified_node_id`，**不调用** `state.observe()`（没有新节点，谈不上语义判定），成功则拉黑失败方向继续下一跳，失败则 `end_reason = "rgb_only_physical_failure_recovery_failed"` |
| 第 1 次「不属于当前子指令」 | `observe()` | `explore_once_more`，记 `off_sequence_nodes` |
| **连续第 2 次**「不属于当前子指令」 | `observe()` | `backtrack_and_block`，目标是上一次记的节点 |
| on_route unknown 在预算内（PASS_LANDMARK 等默认 1 次，`VERTICAL_UP/DOWN` 4 次，`EXIT_REGION` 3 次） | `observe()` | `continue_current_instruction`——**⚠️ 在 rgb_only_v1 生产路径下不可达**，因为 `unknown_disposition` 恒为 `None`（见 7.1.7） |
| 同一验证节点累计拉黑满 5 个不同方向（20° 内视为同向） | `_add_block` | `terminated_reason = "all_candidate_directions_blocked_at_verified_node"`，下一跳开头即结束，`end_reason` 取同名值 |
| `RGBOnlyGraphBacktracker` 未动就已相似度达标 | `recover()` 第 1 步 | 立即成功，`"target_panorama_reobserved"` |
| `RGBOnlyGraphBacktracker` 试满 `max_attempts`（默认 2）仍不达标 | `recover()` 第 2 步 | 返回失败，由调用方决定 `end_reason` |
| 最后一条子指令判定完成 | `observe()` | `action="complete"` → `run()` 里 `end_reason = "instruction_sequence_complete"`，`state.complete=True`，唯一的成功终止 |
| 达到 `max_exploration_hops`（30）仍未完成 | `run()` 循环耗尽 | 循环自然结束，`state.complete` 仍为 `False`，`end_reason` 保持初始值 `"max_sequence_exploration_hops"`（第 326 行，📖 静态读码确认） |

### 7.1.6 下游消费者

- `scripts/evaluate_point_navigation.py:188` 的 `termination_category(reason)` 是**唯一**把 `end_reason` 字符串翻译成统计口径的地方：`"instruction_sequence_complete"` → 同名类别；`"sequence_recovery_backtrack_failed"` → 同名类别；以 `"vlm_selection_failed:"` 开头的再细分——包含 `"No floor-bearing candidate"` → `"no_floor_bearing_candidate"`，包含 `"did not return JSON"` → `"vlm_invalid_json"`，其余 → `"vlm_selection_failed_other"`；其他一律原样返回或 `"unknown"`。这个函数的输出直接进最终 `summary.json` 的 `termination_category_counts`（`evaluate_point_navigation.py:943`），是失败归因统计的唯一入口。
- `state.snapshot()`（`instruction_sequence_exploration.py:1092`）——`cursor`/`off_sequence_nodes`/`blocked_yaws`/`terminated_reason` 等——写进 `InstructionSequenceExplorationResult`，最终落盘到每个 episode 输出目录的 `trajectory.json`，供 `verify_round_stage_completions.py`、`audit_active_stop_round.py` 等事后审计脚本读取复核。
- `navigation_graph/navigation_graph.json` 里每条边的 `action_history`（含 `turn_to_selected_target_actions` 补的负数步）和 keyframe 序列，是模块 6 判定和事后人工复核（`docs` 里各类 round log）的唯一动作证据来源。
- `success=True` 是模块 8 发 STOP 的第一个必要条件（`sequence_completed_in_order`）。

### 7.1.7 已知坑 / 历史代码

- **`node_backtracking.py`（`NodeBacktrackingController`）与 RGB-only 无关**（📖，已用 grep 核实）：`rgb_only_instruction_sequence.py` 全文没有 import 它；它在 `habitat_point_navigation.py:1760` 被实例化的唯一位置是 `args.backtrack_target_node is not None` 这个独立的手动回溯调试 CLI 分支（与两种探索策略完全无关），真正在序列策略内部被使用的只有旧版 `InstructionSequenceExplorationStrategy.__init__`（`instruction_sequence_exploration.py:1143`）。RGB-only 的回溯完全自包含在 `RGBOnlyGraphBacktracker` 里，不依赖 `node_backtracking.py` 任何一行代码。原文"文件"一栏列出的 `node_backtracking.py` 在正式路径下不参与。
- **`chained_stop_wait_eligible()`（STOP_WAIT 链式判定，`instruction_sequence_exploration.py:595`）与 `selection_stage["form"]` 的 `TURN_TO_LANDMARK` 提前查看下一条子指令重写逻辑（约 1453–1560 行）都是旧版 `InstructionSequenceExplorationStrategy.run()` 内部专用的**（📖，已用 import 清单核实）：`rgb_only_instruction_sequence.py` 只从 `instruction_sequence_exploration` import 了 `InstructionSequenceExplorationResult` 和 `InstructionSequenceStateMachine` 两个名字，既不引用 `chained_stop_wait_eligible`，也不构造 `next_sub_instruction_context`/`following_sub_instruction_contexts` 这类字段，因此这两段逻辑在 rgb_only_v1 路径下完全不会被执行，只作历史代码理解旧版策略时参考。
- **回溯尝试次数：2，不是 4**——`RGBOnlyGraphBacktracker.__init__` 的 `max_attempts` 默认值是 2（由 `RGBOnlyInstructionSequenceExplorationStrategy.__init__` 的 `recovery_backtrack_attempts_per_hop=2` 传入），"4" 是旧版 `InstructionSequenceExplorationStrategy.__init__` 的 `recovery_backtrack_attempts_per_hop=4` 默认值。两套策略的类名、构造参数名几乎一样，读代码时极易张冠李戴，务必以类名 + `mode` 字符串（`"instruction-sequence-recovery-rgb-only"` vs. `"instruction-sequence-recovery"`）区分。
- **on_route 分级预算（4/3 次）是死代码**——见 7.1.3 结尾与 7.1.4 Case D：逻辑本身跑起来是对的，但 RGB-only 的 `classification["unknown_disposition"]` 硬编码为 `None`，触发条件永远不满足。如果未来要真的启用这条分级预算，需要在 `rgb_only_instruction_sequence.py` 里补一个 RGB-only 版本的「on_route 判定」（例如用几何/语义相似度判断"没走到但确实还在路上"），而不是照抄旧版 `infer_unknown_disposition()`（它依赖旧版判定的 `endpoint_evidence`/`temporal_evidence`/`motion_evidence` 等字段，RGB-only 的简化判定压根不产出这些）。
- **行号漂移**：272c223 和 2a3dbd9 两次修复往 `rgb_only_instruction_sequence.py` 前部插入了约 63 行新代码（`turn_to_selected_target_actions`、`prepend_turn_start_keyframe`、`try/except RuntimeError` 及相关 import），导致 `run()` 从原文记录的第 255 行漂移到第 318 行，`RGBOnlyGraphBacktracker.recover()` 从第 101 行漂移到第 164 行；`instruction_sequence_exploration.py` 因两次提交均未触碰，`InstructionSequenceStateMachine.observe()` 仍稳定在第 920 行。

### 7.1.8 新增默认回溯：动作历史逆向回放（`--backtrack-method action-reversal`，2026-09-11）

> 本节写于 2026-09-11。7.1.3 描述的「VLM 指方向 + 点导航 + 指纹相似度 ≥ 0.75」回溯保留为 `--backtrack-method visual`，**不再是默认**；默认改为本节的 `action-reversal`。两者只在 `RGBOnlyInstructionSequenceExplorationStrategy` 内部切换，其余模块不感知。

**一句话**：一跳是怎么走出去的，就原样倒着走回来。一跳的完整命令序列 `H`（选点器转向 `turn_to_selected_target_actions` + 执行器每步动作）是自己发出去的、完全已知的，所以「回到出发节点」不需要再当成一个新的导航问题去解，而是一个固定动作串：

1. 原地 `turn_left` × (180 / turn_step)，默认 15° → 12 次；
2. 按 `reversed(H)` 逐条回放，`turn_left`↔`turn_right` 互换，`move_forward` 不变；
3. 再 `turn_left` × 12，把动作坐标系的朝向精确还原到这一跳出发时的朝向 h0。

例：`F,F,R,F,L,F` → `12×L, F, R, F, L, F, F, 12×L`。朝向记账只累加 `commanded_turn_deg`（h0+Δ → +π → −Δ → +π ≡ h0），全程不读仿真器姿态；纯逻辑在 `plan_action_reversal(action_history, turn_step_deg)`，执行在 `RGBOnlyActionReversalBacktracker.recover()`（`scripts/rgb_only_instruction_sequence.py`，紧跟 `RGBOnlyGraphBacktracker` 之后）。CLI 层会拒绝 180 不是 `--turn-step-deg` 整数倍的配置。

**门槛只看 VLM**：回放完拍六视图，调用新的 `NavigationVLMHarness.judge_node_revisit_rgb_only()`（prompt 标签 `RGB_ONLY_NODE_REVISIT_CONFIRMATION`，版本 `v1_two_panorama_same_place`）——上排是目标节点存的六视图、下排是当前六视图、视角一一对应，让 VLM 回答 `same_place`。`CompactVisualEmbedder` 余弦相似度仍然算并记进 `attempts[0].target_panorama_similarity_after`，但**不参与判定**（5.1.3 实测过它对灰图/噪声图都能给 0.97，不可靠）。VLM 说不是同一地点 → 回溯失败，episode 直接以既有的 `rgb_only_physical_failure_recovery_failed` / `rgb_only_sequence_recovery_failed` 结束，**不回退到 visual 方法**。

**两处调用点的差异**（`run()` 里通过 `self._recover(..., create_revisit_node=...)` 分派）：

| 触发 | 回溯目标 | 回放的历史 | 成功后 | `end_reason` |
|---|---|---|---|---|
| ① 点导航没 arrive（`if not navigation.arrived`） | **这一跳的出发节点** `graph_memory.nodes[-1]`（visual 方法用的是 `state.last_verified_node_id`，两者在 explore_once_more 之后会不同） | 刚失败那一跳的 `action_history` | 不建节点、不建边（失败的尝试永远不进记忆图）；`block_failed_physical_direction(selected_heading)`；`previous_action_history`/`has_incoming_edge` 恢复成这一跳开始前的值；代码断言朝向 == h0 | `action_reversal_return_confirmed` |
| ② 判定连续 unknown → `backtrack_and_block` | `directive.backtrack_target_node_id`（= B），代码断言它就是当前节点 C 的 `predecessor` | 刚写进 B→C 边的那份 `action_history` | 新建回访节点 B′（`edge_kind="rgb_only_action_reversal_backtrack"`，`arrival_signal="action_reversal_node_revisit_confirmed"`）+ `add_loop_closure_edge(B, B′)`（`verification="vlm_node_revisit_confirmation"`），然后 `state.on_backtrack(True, current_node_id=B′)` 照旧 | `action_reversal_return_confirmed_node_recorded` |

其它 `end_reason`：`action_reversal_return_rejected_by_vlm`、`action_reversal_vlm_error`（harness 重试耗尽；`VLMProviderFatalError` 仍向上抛）。这些只出现在 `recovery_records[*].end_reason` / `attempts[0]` 里，策略顶层 `end_reason` 不变，`termination_category` 无需改。每条 recovery 记录新增 `backtrack_method` 与 `trigger`（`physical_failure` / `backtrack_and_block`）。

**事后审计**：回放前后各发一次 `emit_evaluation_event`（`action_reversal_backtrack_started` / `_finished`，带 `same_place`、`confidence`、`similarity`），考官在 `evaluation_only/evaluation_geometry.json` 的 `point_events` 里能看到隐藏坐标，可直接量「真的回到了几厘米以内」——这条通道是写入型的，策略读不到。录像里逆向回放帧的 phase 为 `action_reversal_turn_around` / `action_reversal_replay` / `action_reversal_restore_heading`。

**✅ 实测**（2026-09-11，`--vlm-backend heuristic`、EP index 0、`--max-steps-per-target 12`、`outputs/action_reversal_smoke`）：第一跳 12 步用完没 arrive，隐藏坐标从 (15.069, −4.485) 走到 (12.865, −4.411)；逆向回放 36 个动作后回到 (15.065, −4.411)，XZ 误差 ≈ 0.07 m，朝向 1.047 rad 与出发时完全一致；图里没有为失败跳建节点，出发方向被拉黑，第二跳换方向后正常到达；`rgb_only_contract_audit.json` `passed=true`。启发式后端固定回答 `same_place=true`，因此这次只验证了机械回放与数据流，**VLM 判定质量尚未验证**。单元/契约测试 `tests/test_rgb_only_action_reversal_backtrack.py`（12 个）与全量 375 个测试全部通过。

**规则衔接**：`project_rulle.md` §5.7 / §7.7 现行措辞要求回溯成功同时满足「0.75 m 平面距离 + 六视图相似度 ≥ 0.75」；`action-reversal` 的成功门只有 VLM 判定，相似度与隐藏距离只记录不判定。按 §16 规则只能在实验前改口径，本次未改 `project_rulle.md`，需要在真实 VLM 十 EP 实验前由规则维护者补一句「`--backtrack-method action-reversal` 的回溯成功门为 VLM `same_place` 判定；余弦相似度与隐藏几何距离仍必须记录并在事后审计中报告」。

---

# 模块 8：STOP 与打分（考官）

**文件**：`scripts/r2r_stop_success.py`、`scripts/path_projection.py`、`scripts/audit_active_stop_round.py` 等

- **主动 STOP**：只有"所有子指令按顺序全部 completed"时才发出 STOP（`build_stop_action_record`，第 9 行）。
- **成功**（`simulator_stop_success`，第 29 行）= 发了 STOP **且** STOP 位置到终点的测地距离 ≤ 3 米。距离由考官（拿着原始仿真器的评测代码）事后量，模型全程看不到。
- 项目规则第 16.6 条：只有一个冻结配置对固定的十道题（EP 0, 3, 6, 9, 18, 27, 45, 126, 204, 219）**10/10 全部成功**才算过关；"事后补 STOP"、"只是路过了终点"都不算。
- `path_projection.py` 只是把隐藏的标准路线画在录像上给人看，模块开头就写明永远不进模型输入。

## 8.1 扩充细节：把 STOP 与打分从头到尾讲透

> 补写说明（2026-09-11）：本节基于对 `scripts/r2r_stop_success.py`（全文 31 行）的通读并**实际跑过**其单元测试（`tests/test_r2r_stop_success.py`，3 个用例全部通过）；`habitat_point_navigation.py` 里 STOP 记录与打分的写入逻辑（约第 1760-1910 行）、`path_projection.py`（全文 101 行）、`audit_rgb_only_contract.py`（全文 85 行）为静态阅读；`audit_active_stop_round.py`（592 行）、`verify_round_stage_completions.py`（826 行）只读了与本节直接相关的片段（开头文档字符串、`FIXED_EPISODES`、`summary.json` 写入点），未通读全文，因此这两个脚本内部更细的打分算法留待需要时再深入；`project_rulle.md` 第 16.6 节原文已核对。**本节不涉及任何跑 GPU/网络的操作。**

### 8.1.1 一句话 + 比喻

这是"考官"模块：机器人自己什么时候能喊"到了，我不走了"（STOP），以及事后怎么给这次导航打分及格还是不及格。规则很严格——**必须是机器人自己在正确的时候、自己判断走完了、自己喊停**，喊停的位置又必须真的落在目标附近，两个条件都满足才算通过；旁边还站着一群"事后核查员"（审计脚本），专门检查机器人有没有偷看了不该看的东西、有没有作弊喊停。

### 8.1.2 STOP 什么时候由谁发出

`scripts/r2r_stop_success.py` 是这套 STOP 语义的唯一定义来源，只有两个函数，纯逻辑、不依赖仿真器：

```python
def build_stop_action_record(*, sequence_completed_in_order,
                              complete_instruction_was_evaluated,
                              global_step, position_xyz):
    issued = bool(sequence_completed_in_order and complete_instruction_was_evaluated)
    ...
```

`issued`（是否真的发出了 STOP）需要**同时满足两个条件**：

1. `sequence_completed_in_order`：外层策略（模块 7）报告"整条子指令序列按顺序全部 `completed`"（`sequence_exploration_result.success` 为真）。
2. `complete_instruction_was_evaluated`：这一轮确实评测了**完整的**指令序列，不是只测了其中一段（`reference_index is None and len(sub_instructions) == len(all_sub_instructions)`）——如果只是诊断性地跑了指令的一部分（比如从某个中间参考点开始跑），哪怕这一段全部 `completed`，也不允许发 STOP，因为没有评测到"最后一句子指令"本身。

两个条件都满足，才在**当前位置**（`final_position`，不是目标位置、不是任何参考路径点）记一次 `action="STOP"`。这与 Habitat R2R 任务的语义完全对应：STOP 是任务专属的终止动作，一旦发出就不再有后续移动，成功与否就看发出 STOP 那一刻的位置。

`habitat_point_navigation.py` 里实际调用处（约第 1810-1823 行）把这两个布尔量算出来后传给 `build_stop_action_record`，再用第二个函数评分：

```python
def simulator_stop_success(*, stop_action_issued, goal_radius_hit):
    return bool(stop_action_issued and goal_radius_hit)
```

`goal_radius_hit` 是**独立计算**的：拿最终位置到 episode 目标位置的**测地距离**（`habitat_sim.ShortestPath`，沿导航网格走的真实最短路径，不是直线距离）跟 `success_radius`（取自 episode 数据里的 `goal.radius`，**默认 3.0 米**，`habitat_point_navigation.py:1808` 的 `float(goal.get("radius", 3.0))`）比较，小于等于就算命中。这两个量——STOP 有没有发、有没有落在半径内——**在代码里各自独立算，最后才 AND 起来**，任何一个不满足都不算 `simulator_reported_success`。

实际跑了 `tests/test_r2r_stop_success.py` 的 3 个用例，全部通过：

| 用例 | 含义 |
|---|---|
| `test_final_completed_stage_emits_stop_at_current_pose` | 序列全部按顺序完成、且评测的是完整指令 → 在当前位姿发 STOP |
| `test_partial_sequence_cannot_emit_stop` | 只完成部分序列或只评测了部分指令 → 不发 STOP |
| `test_stop_and_radius_are_jointly_required` | STOP 发了但不在半径内，或在半径内但没发 STOP，两种情况都不算成功 |

### 8.1.3 这不是唯一的"成功"字段——`trajectory.json` 里还有好几层

`habitat_point_navigation.py` 在 `r2r_metrics` 里一口气写了好几个相关但含义不同的字段（第 1888-1909 行附近），很容易搞混，这里按项目规则的口径捋清楚：

| 字段 | 含义 | 是否是最终通过标准 |
|---|---|---|
| `stop_action_issued` | STOP 有没有被发出 | 必要条件之一 |
| `goal_radius_hit_diagnostic` / `invalid_goal_radius_hit` | 最终位置是否在目标半径内（不管有没有发 STOP，纯几何诊断） | **不是**——仅几何命中不算数，这两个字段名字里特意带 `diagnostic`/`invalid` 提醒读者别拿它当成功标准 |
| `simulator_reported_success` | `simulator_stop_success()` 的结果（STOP + 半径同时满足） | **project_rulle.md 16.6 规定的唯一硬指标**：十 EP 同轮全部为 `true` 才算通过 |
| `system_provisional_goal_success` | 目前代码里直接等于 `simulator_reported_success` | 过渡态字段，见下条 |
| `instruction_validated_success` / `independent_semantic_verification_complete` | 写死为 `False` | **在线代码永远不会自己置真**；注释原话是"live navigator cannot certify its own completion decision, so task success fails closed here"——必须靠事后独立的语义核查（`verify_round_stage_completions.py`）才能把这个字段"升级"，在线系统自己不能给自己打勾 |

这也是为什么 project_rulle.md 16.6 只认 `simulator_reported_success`：它是唯一一个"在线代码自己算出来、且不能自我认证"的硬指标，其余字段要么是诊断量，要么明确设计成默认 `False`、留给事后审计去核实。

### 8.1.4 STOP 之外的"考官们"：审计脚本各查什么

STOP 判定本身很简单，但围绕它有一整套事后核查脚本，各自职责不同、**互不能替代**：

- **`audit_rgb_only_contract.py`**：扫 `trajectory.json` + `navigation_graph.json` 里每一个字段名，只要键名里出现 `position_xyz`/`base_yaw_rad`/`absolute_yaw_rad`/`depth_m`/`world_xyz`/`navmesh`/`geodesic`/`collision`/`moved_m`/`reference_path`/`goal_position` 这些"特权词"并且值不是 `None`，就记一条违规（`traveled_distance_m == 0.0` 单独豁免，见 5.1.8）。这个脚本查的是"有没有偷看不该看的东西"，跟成功与否无关。
- **`path_projection.py`**：只负责把隐藏的参考路径和 VLM 选的点画到图上给人看（青色路径线、红色叉是模型选的点），**画完的图从不会被喂回任何模型或排序逻辑**——文件开头的文档字符串直接写明"the reference path is never passed to a model or candidate ranker"。它是纯粹的事后可视化工具。
- **`audit_active_stop_round.py`**：**只认 `FIXED_EPISODES = (0, 3, 6, 9, 18, 27, 45, 126, 204, 219)` 这固定十条**的目录布局（`<round>/episode_XXXX/`），逐 EP 重新用参考轨迹打分并生成 `active_stop_audit.{json,md}`、`failure_cases.{json,md}`，同时会**覆盖写** `<round>/summary.json`。⚠️ **已知坑**：这个脚本不认识 `run_e2e_eval.sh` 产生的 `shard_N/episode_XXXX/` 分片布局，如果误跑在一个 e2e 分片轮次上，会因为找不到 `FIXED_EPISODES` 里的任何 episode 而"审出 0 条"，但仍然会把启动器合并好的 100-episode `summary.json` 整个覆盖掉（2026-09-11 在 `outputs/e2e_eval/20260911_001436_e2e_opennav` 上真的踩过这个坑，恢复方法是用 `merge_shards()` 重新从各分片 `summary.json` 拼回来）。**结论：这个脚本只能用在真正的固定十 EP 轮次上，绝不能跑在 e2e 分片轮次上。**
- **`verify_round_stage_completions.py`**：跟上面那个脚本是姊妹关系（复用它的 `_load_dataset`/`audit_episode`），但职责不同、**更安全**——它是"独立复核在线完成判定"的角色：对每一条被在线判定为 `completed`（或 `--include-unknown` 时也包含 `unknown`）的边，重新调用一次 DeepSeek（`VERIFY_SCHEMA` 要求返回 `semantic_completion_verified`/`ordered_stage_boundary_verified`/`confidence`/`reason`/`visual_evidence`），并且额外用参考轨迹加一道"≤30° 前向进度"几何门槛。它**只往每个 episode 目录写 `stage_completion_verification.json`，往轮次目录写 `stage_verification_summary.json`**，不会碰启动器的 `summary.json`，所以在 e2e 分片轮次上跑是安全的（已在 2026-09-11 改动后支持 `rgb_only_v1` 轮次）。注意它标榜"独立"，但复核用的仍然是同一个 DeepSeek 模型，所以更适合用来发现"在线判过于保守的 unknown"，而不能证明"在线判的 completed 里没有假阳性"。

### 8.1.5 各种情况处理一览

| 情况 | 结果 |
|---|---|
| 全部子指令按顺序 `completed`，评测的是完整指令，STOP 位置在半径内 | `simulator_reported_success=true`，唯一算通过的情况 |
| 全部子指令 `completed`，但 STOP 位置在半径外 | STOP 发了，`stop_action_issued=true`，但 `goal_radius_hit=false` → 不算成功 |
| 序列没走完（有 `unknown`/回溯失败/提前 `break`） | `sequence_completed_in_order=false` → 根本不发 STOP |
| 只评测了指令的一部分（诊断性跑法，`reference_index` 不为空或子指令数对不上） | 即便这部分全 `completed` 也不发 STOP（`complete_instruction_was_evaluated=false`） |
| 最终位置恰好离目标很近，但序列没走完/没发 STOP | `goal_radius_hit_diagnostic=true` 但明确**不算成功**，字段名本身就标了 `diagnostic` |
| 想给"完成判定"再加一层独立核实 | 跑 `verify_round_stage_completions.py`（成本：约每条被判边 1 次 DeepSeek 调用，~2.7 秒/次），不影响 `summary.json` |
| 想检查这一轮有没有 RGB-only 违规字段泄漏 | 跑 `audit_rgb_only_contract.py`，与成功率无关，是契约合规检查 |
| 需要固定十 EP 轮次的正式评分报告 | 跑 `audit_active_stop_round.py`，**只能**用在固定十 EP 布局上 |
| 误把 `audit_active_stop_round.py` 跑在 e2e 分片轮次上 | `summary.json` 被覆盖，需要用 `merge_shards()` 从分片摘要重建 |

### 8.1.6 下游谁在用 / 谁依赖这套口径

- `docs/curriculum_10ep_round_log.md`：每一轮必须依据 `simulator_reported_success` 追加一条记录，不能事后改口径（16.6 明文规定）。
- `evaluate_point_navigation.py`：单 episode/小批量评测也复用同一个 `simulator_stop_success()`，保证口径跟正式十 EP 轮次一致（第 87 行直接 import）。
- `validate_single_point_run.py`：读同一份 `trajectory.json` 做模块交接审计，会引用 `task_stop`/`r2r_metrics` 里的字段做诊断展示。
- 十 EP 冻结推进流程（project_rulle.md 16.2/16.6）：把 `simulator_reported_success` 当作唯一能让一版配置"冻结"的硬门槛，9/10、仅几何命中、仅游标完成、STOP 在半径外、人工接管、事后补写 STOP 全部不算数。

### 8.1.7 已知的边角与坑

- **`audit_active_stop_round.py` 覆盖 e2e 分片轮次 `summary.json`**：8.1.4 已详述，这是本节最重要的一条已知坑，已经真实踩过一次。
- **`instruction_validated_success` 永远是 `False`**：不熟悉这套口径的人第一次看 `trajectory.json` 容易把这个字段当成"最终是否成功"来读，其实它是设计上"fail-closed"的占位字段，必须靠独立事后核查才能升级，在线代码不会也不能自己置真。
- **`goal_radius_hit_diagnostic`/`invalid_goal_radius_hit` 容易被误当成功标准**：字段名里虽然带了 `diagnostic`/`invalid` 提示，但初次阅读容易忽略，一定要以 `simulator_reported_success` 为准。
- **`verify_round_stage_completions.py` 的"独立"是有限的**：复核用的模型和在线判定用的是同一个 DeepSeek 后端，2026-09-11 那一轮它与在线判定 38/38 一致（包括一个已知的假阳性 STOP 案例 id 469），说明它更擅长挑出"在线过于保守判成 unknown 的边"，而不能单独证明"在线判成 completed 的边里没有假阳性"。
- **`success_radius_m` 默认值 3.0 米来自代码兜底，不是每条 episode 数据都显式提供**：`goal.get("radius", 3.0)`，如果 episode JSON 没写 `radius` 字段就静默用 3 米，这属于静态阅读发现、未在真实数据集上逐条核实 R2R val_unseen 是否每条都显式提供该字段。
- **`audit_active_stop_round.py`/`verify_round_stage_completions.py` 内部更细的打分算法（比如具体怎么把参考轨迹和实际动作历史对齐算"≤30° 前向进度"）本节未通读全文源码，只读了与本节直接相关的入口和文档字符串，如需要精确复现这两个脚本的打分细节应再单独深入阅读。

---

# 现在做到哪一步了？

最近一轮 Round 021（2026-09-08，`docs/curriculum_10ep_round_log.md`）：固定十题里 **5/10 主动 STOP 成功**（EP0、45、126、204、219）；57 个选点里 66.7% 方向误差 ≤ 30°；物理到达判定的准确率/精确率/召回率都是 100%（也就是说模块 4 已经不是瓶颈）。失败的五题共同问题是"到了正确节点之后，语义终点附近没有合适的地面候选点"和"unknown 之后恢复时方向漂离已验证的走廊"，也就是模块 3 和模块 7 的问题。

---

# 文件速查

| 想看什么 | 去哪 |
|---|---|
| 总入口、仿真器、录像、打分 | `scripts/habitat_point_navigation.py` |
| 防作弊套子 | `scripts/rgb_only_runtime.py` |
| 拆指令 + 17 种类型 | `instruction_decomposer.py`、`instruction_taxonomy.py` |
| 地面分割 / 物体检测 | `ground_segmentation_backends.py`、`semantic_detector.py` |
| 选点 + 所有 VLM prompt | `point_selectors.py`、`vlm_harness.py` |
| 走路 + 到达判定 | `point_navigation_executor.py`、`image_goal_policy.py`、`track_cluster.py` |
| 记忆图 | `navigation_graph_memory.py` |
| 完成判定 | `instruction_completion_judge.py`、`instruction_completion_evidence.py` |
| 状态机 + 回溯 | `rgb_only_instruction_sequence.py`、`instruction_sequence_exploration.py`、`node_backtracking.py` |
| 实验规则（必读） | `project_rulle.md` |
| 每轮记录 | `docs/curriculum_10ep_round_log.md` |

以上是根据代码实际阅读整理的（五个模块分头读了源码并互相核对过关键数字），没有实际运行任何评测。如果你想深入某一个模块（比如想看某个版本 prompt 的原文，或者跟着一个真实 episode 的 `trajectory.json` 逐步对照），告诉我具体哪块，我可以再展开。

---

# 附录 A：判定模块崩溃 bug 与 Round 021 的 5/10 是怎么来的

> 用户当时发现：执行器首次报告到达后，`vlm_harness.judge_edge_instruction_completion_rgb_only` 调 `_contact_sheet(edge_keyframes)` 要求恰好 6 张图，而执行器默认只产出 5 张关键帧，任何 episode 都会崩；且 `HeuristicBackend` 对这个新 prompt 没有分支。以下是当时的分析与修法决定（修法已落地：改用 `_keyframe_storyboard`，`HeuristicBackend` 补分支，附回归测试）。

**结论先说：Round 021 的 5/10 根本没有经过这个会崩的函数。** 它跑的是 2026-09-08 的"旧契约"代码路径；你发现的 bug 在 2026-09-09 的 RGB-only 重构中新写的另一个函数里。两者的关系我逐条用代码证据讲清楚。

## 一、时间线：Round 021 在 RGB-only 硬约束之前

- `docs/curriculum_10ep_round_log.md:2045`：Round 021 日期 **2026-09-08**，是日志里最后一条记录，之后没有任何一轮。
- `project_rulle.md:3`：「第 0 节 在线导航的 RGB-only 硬约束（**2026-09-09**，优先级最高）」，第 19 行明确写着「当前有效执行配置固定为 `policy_input_contract=rgb_only_v1`……旧的 RGB-D、pose、navmesh、geodesic 执行分支只作为历史代码/离线基准保留，**不得从正式 CLI 选择，也不得计入当前结果**」。
- `habitat_point_navigation.py:528`：`--policy-input-contract choices=["rgb-only-v1"]`，现在 CLI 只剩这一个选项。
- git 历史是扁平导入（`9801651 Initial ...`），本机 `outputs/` 里也没有 Round 021 的产物目录，所以 Round 021 是原作者在 09-08 用重构前的代码跑的，本仓库快照是重构后的状态。

也就是说：**Round 021 = 旧契约 + 旧判定函数；你现在能跑的 = 新契约 + 新判定函数**。5/10 是历史成绩，按第 0 节的规定它已经"不得计入当前结果"。而在新契约下，本机从来没有一条 episode 完成过任何一条子指令。

## 二、两个判定函数是两套东西

`vlm_harness.py` 里有两个"边完成判定"：

| | 旧：`judge_edge_instruction_completion`（第 7677 行） | 新：`judge_edge_instruction_completion_rgb_only`（第 7553 行） |
|---|---|---|
| 谁调用 | `NodeTransitionInstructionCompletionJudge`（旧契约） | `RGBOnlyNodeTransitionInstructionCompletionJudge`（`instruction_completion_judge.py:878`，`rgb_only_v1` 唯一路径，`habitat_point_navigation.py:1859`） |
| 关键帧怎么拼 | `_keyframe_storyboard` / `_keyframe_strip`（第 2135、2150 行），**接受 1–N 张**，3 列排版 | `_contact_sheet`（第 2080 行），**硬要求恰好 6 张**，本来是给六视图环视用的 |
| prompt 版本 | 读 `instruction_completion_prompt_version`，V13–V24 全套规则 + `instruction_completion_judge.py` 里十来道确定性闸门 | **完全不读版本号**，一段固定的简化 prompt，只有一条"空动作边不算完成"的规则 |
| 输入 | 六/八视图、语义、`base_yaw_rad`、`position_xyz`、embedding、carryover 证据 | 六/八视图、关键帧、去敏后的动作序列、语义 |

所以 Round 021 日志里写的「冻结配置 V24」在旧函数里是真的生效的；而在现在的 `rgb_only_v1` 路径下，**V24 这个参数对判定是摆设**——新函数根本不看它。这是比关键帧数量更大的一个"隐形差异"，值得你知道。

新函数的 prompt 第 7629 行自己写着 "image 1 is the chronological edge **storyboard**"，但代码第 7606 行调的却是 `_contact_sheet`。这基本可以断定是重构时的复制粘贴错误：本意就是用 `_keyframe_storyboard`。5 张关键帧不是 bug，它是 README/CLAUDE.md 里冻结的口径（"5 张 keyframe"）；6 张才是错的那一边。

HeuristicBackend 这一半也同理：它靠 prompt 里的标记 `EDGE_INSTRUCTION_COMPLETION_JUDGMENT` 识别判定任务（`vlm_harness.py` 第 1069 行附近），新 prompt 没有这个标记，所以落空返回没有 `status` 的字典。

## 三、Round 021 的 5/10 到底是怎么做出来的

这才是你问题的核心。Round 021 之所以能成，是因为旧契约下**五个环节都在用现在被第 0 节明令禁止的特权信息**。逐个对应代码：

**1. 选点后的 navmesh 修复（模块 3）**
`point_selectors.py:1586`：`if (not rgb_only and ...)` 包着整段 `repair_selected_ground_point`（第 344 行）。它把 VLM 选的像素反投影到世界坐标（需要深度）、`sim.pathfinder.snap_point` 吸到可行走面、`find_path` 验证有测地路径，并按指令类型限制测地长度（EXIT_REGION 3 m、ENTER_REGION 6 m、TURN 8 m、默认 4 m，第 1626–1652 行）。VLM 选到墙后、桌面上、够不着的地方，都会被"修"到最近的可达点。日志第 1146 行「将 v31 的后选择地面/navmesh 修复扩展到非首段」说的就是它。Round 021「选点 ≤30° 38/57」这个数字是修复之后的。

**2. 执行器的测地端点 guard（模块 4）**
Round 021 日志：「冻结端点距离 guard 覆盖全部图像到达 proposal（包括导航点仍可见和零步 near-field）」。对应 `point_navigation_executor.py` 旧契约分支：`request.selected_point_navmesh_xyz`（第 794、1646、1861 行）+ `self.sim.pathfinder.find_path(endpoint_path)`（第 1657 行）——每一步都用**真实位置到目标点的测地距离**来否决或确认"到达"；走路也是 `pathfinder.try_step`（第 1933 行），连续转角 + 直接判碰撞。「物理到达混淆矩阵 TP=55、FP=0、FN=0」的 100% 准确率就是这样来的：有了测地距离当裁判，画面点簇消失只是提案，最终由几何拍板。新契约的 `rgb_only_dense_stop_v1`（第 482 行）只有"45 个点 ≥50% 消失连续 3 帧 + 至少前进 3 步 + 画面变化"，没有这个裁判。

**3. 回溯用坐标（模块 7）**
`node_backtracking.py` 的 `BACKTRACK_PLANNER_PROFILES`：`breadcrumb_endpoint_v4/v5` 用 `navmesh_first_route_waypoint`（第 36 行）取导航网格最短路的第一个航点当方向，近距离时直接把**存储的节点坐标**交给执行器；`NodeRevisitMatcher` 要求平面距离 ≤0.75 m **且**视觉相似度 ≥0.75（第 485 行）。新契约的 `RGBOnlyGraphBacktracker`（`rgb_only_instruction_sequence.py:101`）只剩视觉相似度一条，靠 VLM 指方向。（2026-09-11 起默认改为 `--backtrack-method action-reversal`：不再重新导航，而是把这一跳的命令串倒着回放、由 VLM 确认是否回到原节点，见 7.1.8。）

**4. 判定用姿态（模块 6）**
旧判定输入含 `previous_base_yaw_rad / current_base_yaw_rad / position_xyz`，`instruction_completion_evidence.summarize_visual_transition` 用真实朝向差算 `endpoint_heading_delta_deg`；`instruction_completion_judge.py:315-352` 的"反向车道否决"用 `cosine_with_source_incoming_lane < -0.20`，也是几何量。Round 021 的「八视图朝向校正」（V28）、「两参照物左右夹持」等闸门都挂在这条旧路径上。

**5. 事后隐藏 GT 审计（允许，但要分清）**
「独立逐段 RGB/action 核验」「隐藏 GT ≤30° 和正向进展门」是跑完之后用参考路径打分，不进决策。这部分在新旧契约下都合法，但它是**打分**，不是成功的来源。

把这五点合起来看：Round 021 的 5/10 是「VLM 给方向 + 几何兜底修正 + 几何裁判到达 + 坐标回溯 + 姿态辅助判定」的成绩。09-09 的重构把兜底和裁判全部拆掉了（这是正确的，规则要求如此），但重构后的新判定函数没有被端到端跑过一次——你发现的 bug 就是证据：只要执行器第一次报到达就崩，所以重构后从未有任何一条子指令在本机被判过 completed。**目前 `rgb_only_v1` 下的真实成功率是未知数，不是 5/10。**

## 四、修哪种、谁来修

我的决定：

- **修法采用 (a)，但具体是把 `_contact_sheet(edge_keyframes)` 换成已有的 `_keyframe_storyboard(edge_keyframes)`**（`vlm_harness.py:7606`）。理由：prompt 原文就叫它 storyboard，旧判定用的也是它，1–N 张都收，不改一个字的 prompt，不动 `edge_keyframe_count=5` 的冻结口径。不采用 (b)：把默认改成 6 是把错的那一边固定下来，而且 6 张关键帧被当成 3×2"六视图"排版，语义也是错的。
- **HeuristicBackend 补一个分支**：匹配 prompt 里已有的 `"RGB-only edge completion judge"` 字样（不需要改 prompt），返回和现有 `EDGE_INSTRUCTION_COMPLETION_JUDGMENT` 分支一致的 `status="completed", confidence=0.75` 加上 schema 要求的 `reason/visual_evidence/temporal_evidence`。用 completed 而不是 unknown，是因为你要的是无 key 冒烟能走到 STOP；新函数的 validator 本身已经会把"空动作边"降为 unknown，所以不会凭空造出到达。HeuristicBackend 只在测试里用，不影响正式结果。
- **配一个回归测试**：构造 5 张关键帧 + heuristic 后端，断言 `RGBOnlyNodeTransitionInstructionCompletionJudge.judge` 不抛异常且返回 `completed/unknown` 之一；顺手断言 1 张和 8 张关键帧也能过。
- **由我来改**。改动约十行、只碰 Rk-C 模块（`vlm_harness.py` 两处 + 一个新测试文件），不碰 prompt 文本和任何冻结常量，改完跑全部 280 个单元测试，再用 `--vlm-backend heuristic` 跑一条能到达的 episode 确认判定被真正触发。改完我会在结论里如实标注哪些是"实际运行通过"、哪些是"仅静态检查"。

另外要提醒一句：修好这个 bug 之后，`rgb_only_v1` 的判定仍然是那段简化 prompt，V24 的规则不会自动回来。如果你希望新契约下也用 V24 的结构化判定，那是另一个独立的 Rk-C 改动，需要把旧函数里不依赖 pose/position 的部分迁过来，应该单独立项、单独记轮次，不要和这个 bug 修复混在一起。

你回一句"去改"，我就按上面的方案动手。

---

# 附录 B：地面 mask 裁剪、节点间距与"无地面候选"的真正原因

> 用户当时的问题：这个实现是不是只打照片某个纵坐标以上的地面 mask，防止选点太近、节点太近？但有时 mask 太小一个点都打不了。

**结论先说：你的猜测一半对一半不对。** 代码确实在裁地面 mask 的行范围，而且底部那一刀的目的就是"别把点选得太近"；但它裁的不是"某个纵坐标以上"，而是一个**上下左右都收的内圈带**，并且这一刀**永远不会**把 mask 裁成空（裁空了会自动退回整张 mask）。你看到的"mask 太小、一个点都打不了"是另外三个原因造成的，我用本机 2026-09-10 那轮崩掉的 episode 6 给你对上号。

## 一、到底裁了哪一块

`scripts/point_selectors.py:113-122` `targetable_ground_mask`：

```python
valid[: int(0.42 * h)] = 0        # 上 42% 清零
valid[int(0.86 * h):] = 0         # 下 14% 清零
valid[:, : int(0.10 * w)] = 0     # 左 10%
valid[:, int(0.90 * w):] = 0      # 右 10%
if not valid.any(): valid = mask  # 裁空了就退回整张 mask
```

然后 `select_ground_point`（第 143 行）在这个带里挑点，偏好 **y ≈ 0.68h**、居中、离 mask 边缘远。

要理解这几条线的含义，得把像素换算成米。相机参数在 `habitat_point_navigation.py` `make_sim`：高 1.25 m、水平不俯仰、320×240、HFOV 90°，焦距 160 px，地平线正好在第 120 行（0.50h）。平地上 y 行对应的地面距离 = 1.25 × 160 / (y − 119.5)：

| 图像行 | 地面距离 | 在代码里的角色 |
|---|---|---|
| 0.42h（101 行） | **地平线以上** | 上截止线。平地上这里根本没有地面 |
| 0.50h（120 行） | 无穷远 | 地平线 |
| 0.52h | ≈38 m | VLM 锚点采样的"support band"下限（`vlm_harness.py:2236`） |
| 0.68h | ≈4.6 m | 选点偏好中心 |
| 0.72h | ≈3.7 m | STOP/PASS 形式的"近侧带"下限 |
| 0.86h（206 行） | **≈2.3 m** | 下截止线 |
| 1.00h（239 行） | **≈1.67 m** | 画面最底行：相机看不到比这更近的地面 |

所以三条线各干各的事：

- **上截止 0.42h 不是"防止选太远"**。它在地平线以上，平地上没有地面，只会切掉分割模型漏到墙面、泳池壁上的误判，以及下楼梯时高于地平线的台阶。对普通场景基本是空操作。
- **下截止 0.86h 才是你说的"防止选太近"**。它去掉 1.67–2.3 m 这一段最近的地面。注释写的理由是"avoid distorted near-camera rays / generic near-camera targets"：近处像素的方位角变化剧烈，一个像素偏差就是好几度，而且 VLM 很容易偷懒选脚下。
- **左右各 10%** 是去掉画面边缘的畸变射线。

## 二、这一刀和"节点太近"的关系

节点间距不是由这条线单独决定，而是它和执行器的到达判定合起来决定的：

1. 执行器把目标点放在 goal crop 的**底边**，stop 点簇就铺在目标点周围（`point_navigation_executor.py:648-650`，crop 底部 16%）。
2. 到达判定 = stop 点簇 ≥50% 跑出画面底边、连续 3 帧（`rgb_only_dense_stop_v1`，第 482 行）。点跑出画面底边的物理含义是"目标点已经比 1.67 m 更近"。
3. 再加 3 帧确认（每帧前进 0.25 m），实际停下时**离目标点还剩约 1–1.5 m**。

代入下截止线：目标点最近 2.3 m，减去 1.67 m 的盲区，只走 0.6 m ≈ 3 步就判到达。所以一条边最短大约 3–5 步（0.75–1.25 m），`minimum_forward_commands: 3` 是另一道底线。Episode 6 那 4 条边分别是 7、8、5、7 步，全是 1.25–2 m 的短边——这就是"节点之间太近"的来源：**不是 mask 裁太狠，而是 VLM 偏好选中近端 + 到达提前 1.67 m 触发**。若想拉开节点间距，该动的是选点偏好（0.68h）或到达判定，不是 0.86 这条线。

## 三、"一个点都打不了"的真正原因

先明确一点：`targetable_ground_mask` 第 120-121 行裁空会退回整张 mask，所以**这一刀本身永远不产生空 mask**。一个视图拿不到点（`point=None`），只有三个来源：

**1. dense-majority 本身就没投出地面。** 三个 ADE20K 模型 2/3 多数票（`ground_segmentation_backends.py:286`），只认 floor/rug/carpet/ground/stair 等词（第 113-117 行），水面、玻璃、被识别为 "mat"/"platform" 的地面都不算。非楼梯指令还要再减掉 stair 像素（`point_selectors.py:1177-1191`）。而且正式配置是 strict 模式：所有 `rgb_lower_floor_prior` 兜底都包在 `if not strict_ground` 里（第 1222-1240、1407-1441 行），最后还 `target_mask &= mask`（第 1447-1449 行，"语义只能减像素，不能加"）。所以 Grounded-SAM 模式能靠先验救回来的视图，strict 模式下就是空。

**2. 视图被方向锥整个排除。** 来路反向 50° 锥（`incoming_back`）、每个被 block 的方向 50° 锥（最多 5 个）、路线走廊。`select_ground_target` 第 2586-2677 行：先取"有点且未排除"的视图；没有就尝试只放开反向核心 25° 以外的切线射线、blocked 核心 20° 以外的切线射线；再没有就 `RuntimeError("No floor-bearing candidate can be sent to the VLM")`。

**3. 锚点级别过滤。** 关系走廊 <24 像素退回纯地面（`semantic_point_strategy.py:277`），`history_safe_anchors` 把 6 个锚点全砍掉时会在安全列上重采样（`vlm_harness.py:2482-2516`）。这一层基本都有兜底，很少是元凶。

**用 episode 6 对号入座**（`outputs/e2e_eval/20260910_224927_opennav10_w2/shard_0/episode_0006`）。指令是 "Go straight past the pool"，agent 站在一个 SPA 泳池边。崩溃时所在的 `node_0004` 六视图我拼出来看了：view 0/1/5 正对**水面**——三模型都不认为是地面，这是正确的；view 2/3/4 才有地板，但它们在 120°/180°/240° 的后半球，正是来路方向。此前四跳判定全是 `unknown`（VLM 每次都说"泳池还在前方视野里"），第三次选点时 VLM 已经写着 "View 2 is the only allowed front-hemisphere view"。到 node_0004 时，前方无地面 + 后方被来路锥/阻塞锥盖住 → 六个视图一个都不剩 → 第 2677 行抛错 → `rgb_only_instruction_sequence.run()` 第 275 行没接这个异常 → 进程崩、没有 `trajectory.json`。那一轮前 8 条里有 4 条（6、41、51、115）都是同一个栈。

注意这里"view 2/3/4 是被哪个锥排掉的"是我根据 VLM 留下的话推断的，没重跑分割器核实，其余都是读代码和现场产物得出的。

## 四、该怎么看这个问题

- 0.86 那条线不是病因，也别去放宽它；放宽只会让点更近、节点更密。
- 真正缺的是两件事，都在 Rk-A 范围内、应该单独立项：(1) 序列策略要像旧版那样把这个 `RuntimeError` 转成 `end_reason="vlm_selection_failed"`，让 episode 正常结束而不是进程崩（这条我之前记在 memory 里了）；(2) "六视图全部被排除"时需要一个 RGB-only 的合法出路，比如原地转向重新采样，或者把方向锥从"整视图排除"改成只排除具体射线——现在的切线放开逻辑只在 25°/20° 外才起作用，泳池这种"前方是真的没地面、唯一的地面都在来路方向"的情形它救不了。
- 另外 strict 模式下没有任何弱先验兜底是有意为之的设计（RGB-only 硬约束里"语义只减不加"），所以修法不能是"加一块假地面"，只能是策略层面处理"无地面"这个合法状态。
