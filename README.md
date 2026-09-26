# rt_analysis — 固定优先级 WCRT 界 + 单处理器 EDF 需求界分析

本库包含两套相互独立、互不影响的单核实时分析：

1. **固定优先级 WCRT 界**（原有接口，完全保留）：`Task`/`analyze`/`simulate`；
2. **可抢占 EDF 需求界分析**（本轮新增）：`EdfTask`/`analyze_edf`/
   `simulate_edf`，任务模型为最小到达间隔为 T 的 sporadic 任务，
   `dbf(t) = sum max(0, floor((t-D)/T)+1) * C`。

## 环境与 venv

- Python **3.14.7**（仅标准库；代码只用到 3.10+ 特性），Linux 原生运行。
- 无需任何第三方依赖（`requirements.txt` 只有注释）。

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m unittest discover -s tests -v   # 全部测试（固定优先级 + EDF）
python demo.py                            # 固定优先级演示
python demo_iteration.py                  # EDF 正常/边界结果演示（< 90 s）
```

核心、测试、演示全部离线：不需要网络、账号、密钥、数据库、Docker 或外部服务。

## EDF 需求界分析（新增）

### 接口约定

```python
from fractions import Fraction
from rt_analysis import EdfTask, analyze_edf, simulate_edf

tasks = [
    EdfTask("ctrl",  wcet=Fraction(1, 2), period=2,  deadline=2),
    EdfTask("sense", wcet=2,              period=5,  deadline=5),
    EdfTask("log",   wcet=Fraction(3, 2), period=10, deadline=8),
]
report = analyze_edf(tasks)
report.status               # "schedulable" | "unschedulable" | "not_proven"
report.utilization          # Fraction：精确的 sum(C/T)
report.hyperperiod          # H = lcm(T)；预算超限时可能为 None
report.checkpoints_checked  # 实际检查的关键点数
report.checked_up_to        # 实际验证区间 (0, checked_up_to] 的右端
report.first_violation      # 最早违反点 t（无违反时 None）
report.violation_demands    # 违反点处各任务的 (name, jobs, demand)
```

- **任务模型**：单处理器、可抢占 EDF；sporadic 任务最小到达间隔
  `period`（T，正整数），相对截止期 `deadline`（D，正整数），
  `1 <= D <= T`；执行量 `wcet`（C）为**非负有理数**（`int`、
  `Fraction`，或值为整数的 `float`；非整数 `float` 拒绝以保证精确，
  分数请传 `fractions.Fraction`）。无锁、无自挂起。
- **判定**：利用率 `U = sum(C/T) > 1` 直接判不可行（返回
  `status="unschedulable"`，不抛异常）；否则取超周期
  `H = lcm(T_i)`，检查 `(0, H]` 内所有关键点
  `t = D_i + k*T_i`（去重、升序），`dbf(t) > t` 即在最早违反点判
  不可调度，并返回该点各任务需求与检查范围。
- **显式预算**：`analyze_edf(tasks, *, max_hyperperiod=1_000_000,
  max_checkpoints=100_000)`。若 H 或关键点总数超出预算，返回
  `status="not_proven"`（**未证明**）——这是“信息不足”，绝不能当成
  “不可调度”。
- **为什么 H 足够（不是随意截断）**：对任意 t，
  `dbf(t+H) = dbf(t) + H*U`（长度 H 的窗口中每任务恰好多贡献
  `H/T_i` 个作业）。当 `U <= 1` 时 `dbf(t+H) <= dbf(t) + H`，故
  `(0, H]` 上 `dbf(t) <= t` 可逐段归纳推出所有 `t + kH` 也满足；
  违反若存在，必在 `(0, H]` 内首次出现。此外 dbf 是阶梯函数，在相邻
  关键点之间恒定而供给 t 单调增长，所以只查 `D_i + kT_i` 跳跃点等价
  于查遍区间内所有实数 t。
- **EDF 模拟器** `simulate_edf(tasks, horizon)`：独立的事件驱动
  抢占式 EDF 模拟（同步释放短到达序列，最早绝对截止期优先，
  Fraction 精确时间），返回每作业释放/截止/完成时间及是否错过，是
  独立对照工具，不构成保证的一部分。

### 限制

- 仅覆盖无锁、无自挂起、D <= T 的 implicit/任意截止期 sporadic
  子集；不含开销、抖动、多核或任务依赖。
- 周期互质性差时 H 可能巨大；此时只给“未证明”，不做近似替代判定。

### EDF 测试

`tests/test_edf.py` 包含：输入校验；可调度集；U>1；**U 恰好为 1**
（截止期等于周期可调度、短截止期不可调度）；**零执行量任务**；
**低利用率短截止期失败**（`U=9/10` 但 `dbf(4)=5>4`，并由独立 EDF
  模拟观测到具体错过）；全部小整数任务集（C∈0..2、T,D∈1..3）上
  “逐整数 t 直接需求界枚举”与关键点压缩结果一致；可调度集同步
  EDF 模拟不错过；有理数执行量精确模拟；两类预算耗尽返回未证明。

---

# 固定优先级 WCRT 界（原有功能，接口保留）

单核固定优先级抢占式任务集的最坏响应时间（WCRT）界分析库。纯 Python
标准库实现，从零编写，不依赖任何第三方同类实现。

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
```

## 文件

- `rt_analysis/core.py` — 固定优先级：数据结构、校验、`analyze`
- `rt_analysis/simulate.py` — 固定优先级离散时间轴 PCP 验证模拟器
- `rt_analysis/edf.py` — EDF：校验、dbf、关键点压缩、`analyze_edf`
- `rt_analysis/edf_simulate.py` — EDF 独立事件驱动抢占模拟器
- `tests/` — 校验、固定优先级分析/模拟、EDF 需求界/模拟对照测试
- `demo.py` — 固定输入固定优先级演示
- `demo_iteration.py` — EDF 正常结果与各边界演示
