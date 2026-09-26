# rt_analysis — 固定优先级任务的最坏响应时间界（含周期性供给窗口）

单核固定优先级抢占式任务集的最坏响应时间（WCRT）界分析库，并支持一种
明确的周期性处理器供给窗口模型（每周期仅最后 Q 个时间单位可用）。纯
Python 标准库实现，从零编写，不依赖任何第三方同类实现。

## 隔离 venv 准备（Linux 原生，无需联网装包）

```bash
python3.14 -m venv .venv          # 任何 3.10+ 均可；仅用到标准库
source .venv/bin/activate
python --version                 # 开发环境为 3.14.7
# requirements.txt 为空/仅标准库，不需要 pip install 任何第三方包
```

不使用 Docker、外部服务、账号或密钥；不激活 venv 时直接用系统
`python` 也可以运行。

## 环境

- Python 3.14.7（仅标准库；实际只用到 3.10+ 的特性）
- Linux 原生运行，安装 Python 后无需安装任何其他依赖
- 核心、测试、演示全部离线：不需要网络、账号、密钥、数据库或 Docker

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
report.tasks[i].iterations       # 每轮 IterationRecord(value, interference,
                                 #   next_value, demand, supply, stop_reason)
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

### 周期性供给窗口（本次唯一新增模型）

```python
from rt_analysis import SupplyWindow, analyze, minimum_supply, supply_time_for, simulate

supply = SupplyWindow(period=5, window=3)  # 每 5 个时间单位只在最后 3 个可用
report = analyze(tasks, supply=supply)
minimum_supply(t, supply)      # 任意连续 t 个槽内、最坏相位下保证的供给槽数
supply_time_for(demand, supply)  # 其精确整数逆函数
simulate(tasks, horizon, supply=supply, phases=range(supply.period))
```

- 可用槽为每个周期的最后 Q 个：`[kP + P - Q, (k+1)P)`，其余时间处理器
  对所有任务（含临界区）不可用；`0 < Q <= P`，`Q = P` 退化回原模型。
- **最小供给函数（按到达相位最坏化推导，整数槽精确成立）**：令
  `k = t // P`、`r = t mod P`，则
  `sbf(t) = k*Q + max(0, r - (P-Q))`。前 `P-Q` 个槽内保证供给为 0；
  这**不是** `t*Q/P`（后者忽略最坏到达相位，在窗口边界处会高估供给）。
  `Q = P` 时 `sbf(t) = t`。
- **逆函数**：`supply_time_for(d)` 是满足 `sbf(t) >= d` 的最小 t：
  `gap + floor((d-1)/Q)*P + (d mod Q 或 Q)`，其中 `gap = P-Q`。
- **与高优先级干扰联立**：每轮先算需求
  `demand(R) = C_i + B_i + sum_h ceil((R+J_h)/T_h) C_h`，再取
  `R' = supply_time_for(demand(R))`；从 `R = supply_time_for(C+B)` 起
  迭代到不动点（可保证）、`R > D_i - J_i`（已证明超时）或迭代预算耗尽
  （不确定）。每个 `IterationRecord` 同时返回本轮的 `demand`（需求）、
  `supply`（`sbf(R)` 实际保证供给）、`next_value` 与 `stop_reason`（仅
  最后一轮非空）。**超时与预算耗尽是两种不同状态**。
- 利用率仍按 `sum C/T` 计算（任务自身需求口径不变）；`U <= 1` 在供给
  受限时**不是**充分条件——利用率很低也可能因首个窗口开得太晚而超时，
  这由响应时间迭代判定，测试与 `demo_iteration.py` 均含该例。
- `simulate(..., supply=sw, phases=...)`：整数小周期下枚举给定到达相位
  （作业在 `phi + kT` 释放，供给窗口固定），作为实际排程参照；分析界
  可保证时必须不小于任何相位下观测到的最坏响应。

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
- 模拟器采用一种具体实现（临界区位于执行开头、`k*T`/`phi+k*T` 释放），
  用于“不超过界”的实例级核对；供给模型下可枚举所有整数到达相位，但
  模拟器本身仍是验证辅助，不构成保证的一部分。
- 供给窗口是**单一、明确**的周期性模型：所有任务共享同一处理器供给，
  窗口相对时刻 0 固定，不建模分层/分级供给、任务专属虚处理器、动态
  电压频率切换、上下文切换开销或服务器预算补充策略；本库**不**实现
  完整实时操作系统。
- 上限 20 任务是接口边界；演示与测试使用小规模确定性数据，未做
  上限规模的压力测试。

## 命令

```bash
python -m unittest discover -s tests -v   # 必要测试（约 0.01 s）
python demo.py                            # 推导链 + 实际触发的拒绝边界（约 0.05 s）
python demo_iteration.py                  # 供给窗口逐轮需求/供给 + 全相位
                                          # 模拟核对（约 0.1 s，远低于 90 s）
```

## 文件

- `rt_analysis/core.py` — 数据结构、校验、`analyze`
- `rt_analysis/simulate.py` — 离散时间轴 PCP 验证模拟器（含供给窗口、
  到达相位枚举）
- `tests/` — 校验、分析语义、模拟核对、供给窗口四组测试（期望值均手工
  推导；供给函数另用独立的全相位穷举核对）
- `demo.py` — 固定输入演示
- `demo_iteration.py` — 供给窗口离线演示（Q=P 退化、极短窗口、低利用率
  仍超时、逐轮需求/供给/停止理由）
