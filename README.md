# rt_analysis — 固定优先级任务的最坏响应时间界与优先级分配

单核固定优先级抢占式任务集的最坏响应时间（WCRT）界分析库，外加一个
**优先级分配入口**：当给定优先级顺序无法保证时，自动搜索一个可保证
的顺序。纯 Python 标准库实现，从零编写，不依赖任何第三方同类实现。

## 环境

- Python 3.14.7（仅标准库；实际只用到 3.10+ 的特性）
- Linux 原生运行，安装 Python 后无需安装任何其他依赖
- 核心、测试、演示全部离线：不需要网络、账号、密钥、数据库或 Docker

## 接口

### 响应时间分析（既有入口）

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

### 优先级分配（新入口）

```python
from rt_analysis import assign_priorities

result = assign_priorities(tasks)   # tasks 顺序任意；任务 ID = 输入下标
result.status                       # "assigned" | "no_feasible_order" | "search_budget_exhausted"
result.order                        # 找到的可保证顺序（最高优先级在前）；未找到为 None
result.report                       # 对 result.order 重新 analyze 的完整逐轮证据
result.candidates_tested            # 实际执行的候选测试（搜索节点）数
```

- **最多 8 个任务**（`MAX_ASSIGN_TASKS`）；`analyze` 的 20 任务上限不变。
- 搜索是确定性的：从**最低优先级槽位向上**填充，每槽按任务 ID（输入
  下标）顺序尝试候选；候选测试时假定其余未分配任务都在它上方，并
  **用当前假设顺序重新计算资源天花板与阻塞项**——不沿用输入顺序的
  阻塞量。随后用同一不动点递推计算该候选的响应时间界。
- 该测试只依赖候选的"上方集合/下方集合"而非集合内部顺序，且任一任务
  被下移时干扰（≥ C_k）只会被换成不超过一个临界区（≤ C_k）的阻塞，
  因此通过槽位测试的候选固定在该槽不会破坏可行性——贪心选择是完备
  的。DFS 仍带回溯（有节点预算 `node_budget`，默认 200000，大于 8
  任务全空间的约 11 万次候选测试），用于完成无解的穷尽性证明；
  只有调用者缩小预算时才可能得到 `search_budget_exhausted`。
- 三种结果的语义严格区分：
  - `assigned`：找到顺序，并已用 `analyze` 端到端复核（`result.report`
    含逐轮响应时间证据）；
  - `no_feasible_order`：搜索空间穷尽，**该分析模型下**无可保证顺序
    ——这是模型范围内的结论（模型是保守充分条件），**不**宣称现实
    调度绝对不可行；
  - `search_budget_exhausted`：节点预算先耗尽，结论不确定。
- 利用率超过 1 与顺序无关，直接返回 `no_feasible_order`（0 次测试）。
- 测试中以独立的全排列枚举（对每个排列直接 `analyze`）交叉验证
  搜索找到的顺序与无解判定；演示 `demo_assign.py` 同样执行该枚举。

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
- `analyze`/`simulate` 是纯函数：校验失败时不留下任何部分变更。

## 误差口径

分析全程使用精确整数与 `Fraction` 有理数运算，**无浮点误差**；响应
时间界、阻塞项、干扰次数均为精确整数。利用率以 `Fraction` 返回，
仅 demo/摘要中的 `float(...)` 显示存在 <= 1e-4 的显示舍入，不参与
任何判定。边界（如 `R == D - J`、`U == 1`）按精确相等判定为可保证。

## 设计取舍与已知限制

- 分析是**保守充分**条件而非精确充要条件：可保证的结论可靠，判 miss
  的结论在该保守模型内成立，但不排除存在更精细分析可通过的实例。
- 模型不含上下文切换开销、缓存效应、多核、任务间任意依赖；仅覆盖
  本题契约的周期/偶发 + 抖动 + 天花板协议子集，不宣称完整行业标准
  （如完整 OSEK/AUTOSAR 语义）兼容。
- 模拟器采用一种具体实现（临界区位于执行开头、`k*T` 同步释放），
  用于“不超过界”的实例级核对，不枚举最坏情形。
- 上限 20 任务是分析接口边界；优先级分配入口因最坏情形为阶乘搜索，
  接口上限定为 8 任务（穷尽空间约 11 万次候选测试，在默认预算内）。
  演示与测试使用小规模确定性数据，未做上限规模的压力测试。
- `no_feasible_order` 仅表示在本保守分析模型下无优先级顺序可保证；
  更精细的分析模型或 EDF 等调度策略可能可行，本库不对此下结论。

## 命令

```bash
./.venv/bin/python -m unittest discover -s tests -v   # 全部测试（约 0.02 s）
./.venv/bin/python demo.py                            # 既有入口演示（约 0.05 s）
./.venv/bin/python demo_assign.py                     # 优先级分配演示（远低于 90 s）
```

## 文件

- `rt_analysis/core.py` — 数据结构、校验、`analyze`
- `rt_analysis/assign.py` — 优先级分配入口 `assign_priorities`
- `rt_analysis/simulate.py` — 离散时间轴 PCP 验证模拟器
- `tests/` — 校验、分析语义、模拟核对、优先级分配四组测试（期望值均手工推导，另含独立全排列枚举交叉验证）
- `demo.py` — 既有入口的固定输入演示
- `demo_assign.py` — 新入口演示：低利用率却因阻塞失败、换序后成功（含逐轮证据与模拟核对）、穷尽无解、预算耗尽
