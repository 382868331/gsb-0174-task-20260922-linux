# rt_analysis — 固定优先级任务的最坏响应时间界与优先级分配

单核固定优先级抢占式任务集的最坏响应时间（WCRT）界分析库，外加一个
**优先级分配入口**：当给定（输入）顺序不合适时，自动搜索一个能被分析
证明的优先级顺序。纯 Python 标准库实现，从零编写，不依赖任何第三方
同类实现。

## 环境

- Python 3.14.7（仅标准库；实际只用到 3.10+ 的特性）
- Linux 原生运行，安装 Python 后无需安装任何其他依赖
- 核心、测试、演示全部离线：不需要网络、账号、密钥、数据库或 Docker

## 接口一：固定顺序分析 `analyze`

```python
from rt_analysis import Task, CriticalSection, analyze, simulate

tasks = [
    Task("ctrl",  wcet=2, period=8,  deadline=7, jitter=1,
         critical_sections=(CriticalSection("bus", 1),)),
    Task("sense", wcet=3, period=12, deadline=12,
         critical_sections=(CriticalSection("bus", 2),)),
    Task("logger", wcet=4, period=20, deadline=20,
         critical_sections=(CriticalSection("bus", 3),)),
]
report = analyze(tasks)          # -> Report
report.guaranteed                # 整体是否可保证
report.utilization               # Fraction，精确有理数 sum(C/T)
report.tasks[i].response_time    # R_i（miss/不确定时为 None）
report.tasks[i].blocking         # BlockingInfo(amount, source_task, source_resource)
report.tasks[i].iterations       # 每轮 IterationRecord(value, interference, next_value)
report.tasks[i].status           # "schedulable" | "deadline_miss" | "iteration_budget_exhausted"
```

- **任务顺序即优先级**：`tasks[0]` 优先级最高。最多 **20** 个任务。
- `analyze(tasks, *, max_iterations=1000)`：`max_iterations` 是每任务
  不动点迭代的搜索预算。预算耗尽得到 `iteration_budget_exhausted`
  （不确定），与已证明的 `deadline_miss` 严格区分。
- `simulate(tasks, horizon)`：离散时间轴验证模拟器（固定优先级抢占 +
  优先级天花板协议，作业在 `k*T` 释放，临界区位于执行开头）。用于
  生成具体调度实例并核对观测响应时间不超过分析界；它是验证辅助，
  不构成保证的一部分。

## 接口二：优先级分配 `assign_priorities`（新入口）

```python
from rt_analysis import assign_priorities

result = assign_priorities(tasks)   # tasks 是无序任务集合，最多 8 个
result.status          # "assigned" | "no_feasible_order" | "incomplete"
result.order           # 找到的优先级顺序（最高在前）；未找到时为 None
result.report          # 该顺序的完整 analyze() 证据（逐轮迭代、阻塞来源）
result.placements      # 搜索轨迹：每次候选试放的 level/候选/结果
result.nodes_explored  # 候选评估次数
```

任务参数与阻塞模型完全沿用 `analyze` 的定义；本入口只搜索**顺序**。
搜索是确定性的，分两个阶段：

1. **从最低优先级向上贪心试放**（Audsley 式）。每一层按任务 ID
   （`name` 字典序）依次尝试未分配候选：把候选放在该层、其余未分配
   任务全部放在它上方，组成临时顺序后**完整重跑 `analyze`**——资源
   天花板、阻塞项、响应时间都按临时顺序重新计算，绝不沿用输入顺序
   的阻塞量。第一个在该层收敛于 `D-J` 之内的候选即被放置，搜索上移
   一层。
2. **有预算的回溯**。本模型的单任务判定只依赖更高/更低优先级任务的
   **集合**（干扰对高优先级集合求和；阻塞取决于低优先级集合以及资源
   是否有使用者在它之上），与集合内部顺序无关，因此支撑贪心的可交换
   条件成立；但实现不单纯依赖这一论证——贪心失败时会对同一放置
   测试做深度优先回溯，预算为 `node_budget`（默认 200000 次候选评估；
   8 个任务的完整搜索树不足 11 万次，故默认预算在接口上限内总能
   穷尽）。三种结局：
   - `assigned`：找到顺序，并用 `analyze` 对该顺序完整复核后返回证据；
   - `no_feasible_order`：搜索树**穷尽**仍无顺序可保证——这是该
     **分析模型下**的结论，不宣称现实调度绝对不可行；
   - `incomplete`：预算先耗尽，既未找到也未穷尽，不下结论。

限制与说明：

- 本入口最多 **8** 个任务（`MAX_ASSIGN_TASKS`）；`analyze` 的 20 任务
  上限不变。输入顺序在本入口中没有优先级含义。
- 利用率 `U > 1` 时不启动搜索直接返回 `no_feasible_order`（与顺序无关
  的精确必要条件）。
- 找到的“可保证顺序”含义与 `analyze` 相同：保守充分条件下的保证，
  不排除更精细分析能通过被本模型拒绝的实例。

## 模型（明示保守）

- 整数时间模型：`C >= 1`，`T >= 1`，`1 <= D <= T`，`J >= 0`，临界区
  长度 `>= 1` 且同一任务的临界区长度之和 `<= C`。
- 资源采用（立即）优先级天花板协议：资源天花板 = 使用该资源的最高
  优先级任务；临界区**禁止嵌套**（数据结构本身不表达嵌套）。
- 阻塞项 `B_i` = 低优先级任务中、天花板不低于 `i` 优先级的资源上的
  最大单个临界区；输出阻塞来源（任务与资源）。
- 迭代 `R = C_i + B_i + sum_h ceil((R + J_h) / T_h) * C_h`，从
  `R = C_i + B_i` 开始；候选值超过 `D_i - J_i` 判 miss，值稳定判可
  保证。相对名义请求时间的延迟界为 `R_i + J_i`。
- 利用率 `U = sum(C/T)`：`U > 1` 返回 `guaranteed=False`（**不是**
  格式错误）；`U <= 1` 只是必要条件，**不**被当作充分保证。

## 错误语义

- 非法输入抛出 `rt_analysis.ValidationError`（`ValueError` 子类），
  消息定位到任务下标、任务名与字段，不静默纠正。
- 数值输入拒绝 NaN/±Infinity；整数参数拒绝 bool；非整数值（如 2.5）
  拒绝；有限的整数值浮点（如 4.0）按数值等价接受。
- `analyze`/`simulate`/`assign_priorities` 是纯函数：校验失败时不留
  下任何部分变更；搜索预算耗尽（`incomplete`）与证明无解
  （`no_feasible_order`）严格分开。

## 误差口径

分析全程使用精确整数与 `Fraction` 有理数运算，**无浮点误差**；响应
时间界、阻塞项、干扰次数均为精确整数。利用率以 `Fraction` 返回，
仅 demo/摘要中的 `float(...)` 显示存在 <= 1e-4 的显示舍入，不参与
任何判定。边界（如 `R == D - J`、`U == 1`）按精确相等判定为可保证。

## 设计取舍与已知限制

- 分析是**保守充分**条件而非精确充要条件：可保证的结论可靠，判 miss
  的结论在该保守模型内成立，但不排除存在更精细分析可通过的实例。
- 模型不含上下文切换开销、缓存效应、多核、任务间任意依赖；仅覆盖
  周期/偶发 + 抖动 + 天花板协议子集，不宣称完整行业标准（如完整
  OSEK/AUTOSAR 语义）兼容。
- 模拟器采用一种具体实现（临界区位于执行开头、`k*T` 同步释放），
  用于“不超过界”的实例级核对，不枚举最坏情形。
- 上限 20 任务（`analyze`）与 8 任务（`assign_priorities`）是接口
  边界；演示与测试使用小规模确定性数据，未做上限规模的压力测试。
- `no_feasible_order` 仅表示**该分析模型下**穷尽搜索无可保证顺序，
  不是对现实调度可行性的绝对否定。

## 命令

```bash
./.venv/bin/python -m unittest discover -s tests -v   # 全部回归 + 新入口测试（约 0.1 s）
./.venv/bin/python demo.py                            # 固定顺序分析演示（约 0.05 s）
./.venv/bin/python demo_assign.py                     # 优先级分配演示（约 0.05 s，远低于 90 s）
```

## 文件

- `rt_analysis/core.py` — 数据结构、校验、`analyze`
- `rt_analysis/assign.py` — 优先级分配搜索 `assign_priorities`（新入口）
- `rt_analysis/simulate.py` — 离散时间轴 PCP 验证模拟器
- `tests/` — 校验、分析语义、模拟核对、优先级分配四组测试（期望值均
  手工推导；分配结果另用 `itertools` 全排列独立枚举交叉验证）
- `demo.py` — 固定顺序分析演示
- `demo_assign.py` — 优先级分配演示：低利用率因阻塞失败、换序后成功、
  穷尽搜索证明模型内无可行顺序
