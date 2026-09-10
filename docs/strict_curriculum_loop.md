# 十 EP 严格轮次推进记录规范

固定 EP：`0, 3, 6, 9, 18, 27, 45, 126, 204, 219`。

## 单子指令闭环记录

每个 `episode/sub_instruction/attempt` 建立一条记录，字段至少包括：

| 阶段 | 必须记录的证据 | 通过条件 |
|---|---|---|
| 人工基准 | 原始 instruction、拆解 JSON、六视图、人工候选视角/像素/地面理由、示范路径投影（仅诊断） | 人工目标可在当前 state 解释且落在地面/楼梯地面 |
| 系统选点 | VLM prompt/response、DINO+SAM 记录、允许视图、anchor、最终像素/视角、候选拒绝理由 | 点在合法 mask，视角和人工基准达到本轮角度阈值 |
| 物理导航 | crop、导航/停止点簇逐帧轨迹、动作、位姿、碰撞、到达 signal、终点测地距离（隐藏评分） | 执行器 signal 与真实到达一致，无提前丢点 |
| 人工终点复核 | 终点六视图、移动历史、人工判断 `best/needs_reselect`、理由 | `best` 才可进入节点写入 |
| 节点写入 | node/edge JSON、六视图、语义、embedding、action history、keyframes | node/edge 完整且与执行日志一致 |
| 系统判定 | judge 原始响应、结构化证据、最终 `completed/unknown`、置信度 | 与人工/示范事后审计一致 |
| unknown 分流 | 人工标签 `wrong/on_route`、外层 directive、下一次选点或真实回溯记录 | wrong 必须真实回溯成功后重选；on_route 必须继续当前子指令 |

人工基准和示范轨迹只能用于初始化和模型调用后的审计；任何 prompt、候选排序、
执行控制和 judge 输入都不得包含未来 waypoint、正确视角、path index 或示范距离。

## 轮次目录

```text
outputs/r2r_curriculum_10ep_YYYYMMDD/round_NNN_.../
  manifest.json
  process.log
  summary.json
  failure_cases.md
  trajectory.json
  exploration.mp4
  per_sub_instruction/*.json
  vlm_artifacts/
```

`manifest.json` 必须声明 `test_scope`、固定 EP、真实 state identity、当前只改模块、
父轮次、prompt/模型/阈值/seed、`device=cuda:0`、是否使用示范投影以及前缀回归结果。
未执行模块记为 `not_run`，不得按成功处理。只有当前子指令和所有冻结前缀均通过时，
才允许把轮次标记为 `stage_gate_passed=true`。

## 十 EP 同步运行硬规则

从本规则生效后，每一轮真实测试必须同时包含固定十个 EP：
`0, 3, 6, 9, 18, 27, 45, 126, 204, 219`。同一轮只能针对同一个阶段/子轮次
（选点、导航、判定或回溯）做通用优化，并为十个 EP 分别保存真实输入、结果和失败
证据；没有该阶段子指令的 EP 标记 `not_applicable`。不得先优化一个 EP 再把它的
结果当作十 EP 结论，也不得以单 EP 成功冻结阶段。单 EP 运行仅可作为诊断，不进入
成功率、阶段门槛、前缀冻结或最终端到端结论。任何改动必须先归纳十 EP 的共同错误
类型，再在同一十 EP 上重新验证，确保已通过前缀不退化。

## 当前推进状态

截至 Round 143，EP0/3/6/9 等首段已有历史候选，但 EP27 的最后
`STOP_WAIT` 仍未完成严格闭环，因此下一轮从该未冻结子指令继续，不得跳到后续阶段或
宣称十 EP 端到端完成。生产中间模型固定在 GPU0；CPU 只允许 I/O、统计和 checkpoint
反序列化，不允许模型推理回退。

## 2026-09-07 Active-STOP 重启门槛

本轮从固定十 EP 的真实起点重新建立基线，并继续按同一阶段的 selection、navigation、
judgment 子轮推进。最终验收不允许部分通过：必须在同一冻结配置下十个 EP 全部由在线
系统完成最后子指令后主动发出 `STOP`，并且十个 STOP pose 全部位于各自 Habitat goal
radius 内，即 `simulator_reported_success=10/10`。任何未 STOP 的几何命中或目标外 STOP
都按失败处理；中间轮次不能以单 EP 或部分 EP 的改善宣称最终通过。
