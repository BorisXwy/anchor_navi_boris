# 十 EP 组合式完成基准

## 目的

这个基准把“逐段完成”与“整个 EP 成功”明确连接起来：系统不能因为某一次
点导航到达、写入节点或接近 R2R 终点就算成功。只有每个适用子指令都完成同一
套闭环，系统才被允许宣称该 EP 完成。

固定测试集是 `0, 3, 6, 9, 18, 27, 45, 126, 204, 219`。冻结的机器可读基准由
`scripts/build_ten_ep_completion_benchmark.py` 生成，当前文件为：

`outputs/r2r_curriculum_10ep_20260904/benchmark/ten_ep_completion_benchmark_v1.json`

## 一个子指令的通过条件

对每个有实际子指令的阶段，必须依次通过：

1. **A — 选点**：人工在模型调用前冻结当前视角下的最佳合法地面/楼梯地面区域；
   系统点必须落在 Grounded-SAM 候选内、能投影到 navmesh，并满足当前严格的
   30 度方向门槛。人工基准不进入 VLM 输入。
2. **B — 物理到点**：`PointNavigationExecutor` 返回
   `point_navigation_arrived`，最终选定点测地距离不超过 `0.75m`，且人工根据
   RGB、动作历史和位姿确认不是提前丢簇或假到达。
3. **N — 节点/边 invariant**：写入位置、朝向、六视图、语义状态、视觉 embedding、
   incoming action history 和 keyframes。这个步骤是功能性完整性检查，不用节点写入
   代替选点或导航正确率。
4. **C — 子指令判定**：系统只能返回 `completed` 或 `unknown`。人工确认已完成时
   必须返回 `completed`；在路上、走错、停滞或证据不足均须为 `unknown`。
   `unknown` 的 `wrong/on_route` 细分只供外层探索策略使用。
5. **V — 独立完成核验**：只有 C 在到达后新建的节点/边上返回 `completed`，并且
   运行后的冻结人工语义审计根据节点全景、上一节点、action history、keyframes、
   实际轨迹和示范轨迹确认确实完成时，才计该段成功。缺失核验 artifact 默认失败；
   核验标签不能作为在线导航或判定模型的输入。

若出现 `unknown`，必须由系统继续选点或真实回溯；人工不能替系统接管。只有该
子指令最终得到 `completed` 才能推进下一个子指令。

## EP 级成功判定

对 episode `e`，定义：

```text
EP_success(e) =
  every_applicable_subinstruction_passes(A, B, N, C=completed)
  AND every_required_backtrack_passes
  AND no_unresolved_unknown
  AND no_manual_takeover
  AND final_habitat_goal_reached
  AND final_instruction_judged_completed
```

因此，单段成功不能直接推出 EP 成功；它只能成为后续阶段的真实起点。十 EP
基准成功定义为十个固定 EP 全部满足 `EP_success`。某轮没有对应子指令的 EP 必须
记录 `not_applicable`，不能静默从固定集合或最终分母移除。

进入 R2R goal radius 本身只保存为 `goal_radius_hit_diagnostic`。若完整指令链尚未
按序完成，则同时标记 `invalid_goal_radius_hit=true`，正式成功仍为 0；随机、越序、
走错或仅执行截断前缀造成的终点半径命中不得记录或汇报成“终点到达”。正式终点
成功字段统一为 `instruction_validated_r2r_success`。

## 人工基准与系统输入边界

人工基准用于冻结和事后审计，至少保存：原始 instruction、子指令、六/八视图、
人工目标视角/像素、目标地面理由、拒绝区域和置信度。系统运行前不能看到示范
未来路径、人工目标或隐藏 heading；示范轨迹只能用于运行后评分和失败分析。

每个阶段还必须保存 VLM 请求/响应、候选 mask、最终点、路径投影、crop、跟踪簇、
动作、位姿、节点/边、判定和视频。缺失任一适用阶段 artifact，就不能计为 EP 成功。

## 当前状态

基准 schema 和固定十 EP 已生成；当前优先使用 Round 70 已冻结的逐 EP
VLM/形式拆解（缺失时才回退到形式级 taxonomy scaffold）。第 0 段已经完成并冻结
全部四个门：R0-A 选点 10/10、R0-B 物理到点 10/10、R0-N 节点/边不变量 10/10、
R0-C completed/unknown 判定 10/10（两类 F1 均为 1.0）。对应冻结文件分别位于
`round_276`、`round_278`、`round_286` 和 `round_285`。

第 1 段及以后每个适用子指令的人工基准和真实运行 artifact 仍标记为 pending；下一轮
必须从第 0 段真实终点继续，先在十 EP 同步建立 R1-A 人工选点基准，再依序通过
R1-B/R1-N/R1-C，并在每轮检查已冻结前缀无回归。不能用第 0 段通过结果直接宣称任一
EP 的完整 instruction 或 Habitat 任务成功。
