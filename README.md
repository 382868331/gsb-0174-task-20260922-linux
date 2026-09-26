# rt_analysis — 固定优先级任务的最坏响应时间界（含周期性供给窗口）

单核固定优先级抢占式任务集的最坏响应时间（WCRT）界分析库。纯 Python
标准库实现，从零编写，不依赖任何第三方同类实现。

## 环境

- Python 3.14.7（仅标准库；实际只用到 3.10+ 的特性）
- Linux 原生运行，安装 Python 后无需安装任何其他依赖
- 核心、测试、演示全部离线：不需要网络、账号、密钥、数据库或 Docker

## 准备（隔离 venv）

```bash
python3 -m venv .venv          # 隔离环境；不安装任何第三方包
source .venv/bin/activate      # 或直接使用 .venv/bin/python
```

requirements.txt 为空（仅注释），无需 pip install。

## 接口

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
- `analyze(tasks, *, supply=None, max_iterations=1000)`：`max_iterations`
  是每任务不动点迭代的搜索预算。预算耗尽得到
  `iteration_budget_exhausted`（不确定），与已证明的 `deadline_miss`
  严格区分。
- `simulate(tasks, horizon)`：离散时间轴验证模拟器（固定优先级抢占 +
  优先级天花板协议，作业在 `k*T` 释放，临界区位于执行开头）。用于
  生成具体调度实例并核对观测响应时间不超过分析界；它是验证辅助，
  不构成保证的一部分。

## 周期性供给窗口（新增）

处理器只在每个周期 `P` 的最后 `Q` 个时间单位可执行（`0 < Q <= P`），
其余时间不可用：

```python
from rt_analysis import SupplyWindow, analyze, simulate, simulate_phases
from rt_analysis import supply_bound, supply_inverse

window = SupplyWindow(period=4, budget=2)   # 每 4 拍只有最后 2 拍可执行
report = analyze(tasks, supply=window)      # 响应时间界计入最坏到达相位的空档
report.tasks[i].iterations[j]               # IterationRecord 增加 demand / supply 字段
sim    = simulate(tasks, horizon, supply=window, phase=0)
sweep  = simulate_phases(tasks, horizon, supply=window)  # 枚举全部到达相位
sweep.worst_response()                      # 各任务跨相位最大观测响应时间
```

模型与推导：

- 窗口在第 `k` 个周期的 `[k*P + (P-Q), (k+1)*P)` 内供处理器。最坏到达
  相位下区间先经历最长 `P-Q` 的空档，故**最小供给函数**为
  `sbf(t) = (u // P) * Q + min(Q, u mod P)`，其中 `u = max(0, t-(P-Q))`。
  这是对相位取最小值的精确供给下界，**不是**平均速率 `t*Q/P`；因此
  分析不把执行时间简单除以 `Q/P`（那会忽略空档，偏于不安全）。
- 需求侧与基线相同：`demand(t) = C_i + B_i + ceil((t+J_h)/T_h)*C_h` 求和。
  迭代把需求经供给反函数映射回时间：`R' = supply_inverse(demand(R))`，
  `supply_inverse(x)` 是满足 `sbf(t) >= x` 的最小 `t`（闭式解）。每轮
  `IterationRecord` 记录 `value`（时间候选）、`demand`（执行需求）、
  `supply`（该区间保证供给）与 `next_value`；停止理由照旧区分
  `deadline_miss`（候选超过 `D_i - J_i`）与
  `iteration_budget_exhausted`（预算耗尽、不确定）。
- `Q == P` 时 `sbf(t) = t`、`supply_inverse(x) = x`，逐位退化回
  基线模型（测试逐轮比对两种调用结果一致）。
- `simulate(..., supply=window, phase=s)` 在窗口空档拍不调度；
  `simulate_phases` 枚举 `lcm(P, T_1..T_n)` 个相位（小整数周期下可行），
  作为实际排程参照核对分析界，包含 `Q=P` 退化、极短窗口与利用率低
  仍超时的用例（见 `tests/test_supply.py`）。

适用范围：只实现这一种供给模型（周期性最后-Q窗口），不实现完整实时
操作系统；阻塞项 `B_i` 沿用基线天花板协议上界，在窗口模型下仍为保守
充分条件。

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
- 上限 20 任务是接口边界；演示与测试使用小规模确定性数据，未做
  上限规模的压力测试。

## 命令

```bash
python -m unittest discover -s tests -v   # 必要测试（约 0.01 s）
python demo.py                            # 推导链 + 实际触发的拒绝边界（约 0.05 s）
python demo_iteration.py                  # 供给窗口：逐轮需求/供给/停止理由 + 相位枚举（约 0.05 s）
```

## 文件

- `rt_analysis/core.py` — 数据结构、校验、`analyze`、供给窗口最小供给函数
- `rt_analysis/simulate.py` — 离散时间轴 PCP 验证模拟器（含窗口与相位枚举）
- `tests/` — 校验、分析语义、模拟核对、供给窗口四组测试（期望值均手工推导）
- `demo.py` — 固定输入演示
- `demo_iteration.py` — 供给窗口固定输入演示
