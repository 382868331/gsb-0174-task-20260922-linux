# rt_analysis — 固定优先级任务的最坏响应时间界

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

- `rt_analysis/core.py` — 数据结构、校验、`analyze`
- `rt_analysis/simulate.py` — 离散时间轴 PCP 验证模拟器
- `tests/` — 校验、分析语义、模拟核对三组测试（期望值均手工推导）
- `demo.py` — 固定输入演示


---

# EDF 需求界分析（本轮新增）

单核可抢占 EDF 的可调度性判定。任务为 sporadic 模型：最小到达间隔
T（正整数）、相对截止期 D（正整数，D <= T）、执行量 C（非负有理数，
接受 int / Fraction / 有限 float）。无锁、无自挂起。

## 接口

    from rt_analysis import EdfTask, analyze_edf, dbf, key_points, simulate_edf

    tasks = [EdfTask("fast", 2, 4, 2), EdfTask("slow", 2, 6, 3)]
    report = analyze_edf(tasks)        # -> EdfReport
    report.status                      # "schedulable" | "unschedulable" | "unproven"
    report.utilization                 # Fraction，精确 sum(C/T)
    report.hyperperiod                 # H = lcm(T)，预算内才给出
    report.checkpoints                 # 实际检查的 D + k*T 关键点
    report.checked_until               # 已验证区间 (0, H]
    report.violation                   # 最早违反点 Violation(time, demand, per_task)

- 判定条件：dbf(t) = sum(max(0, floor((t-D)/T)+1) * C)，dbf(t) <= t
  对所有 t > 0 成立且 U <= 1 时可调度。
- 为什么 H 足够而不是随意截断：dbf 只在 D_i + k*T_i 处跃升，两点之间
  不变而 t 持续增长，所以违反只会先出现在关键点；又 H 是每个 T_i 的
  倍数，故 dbf(t+H) = dbf(t) + U*H，当 U <= 1 且 (0, H] 内无违反时，
  任意 t' = t + m*H 满足 dbf(t') <= t + m*U*H <= t'。因此 (0, H] 是
  证明完备的有限界。
- U > 1 直接 unschedulable（过载，不做关键点检查）。
- 预算：analyze_edf(tasks, *, max_hyperperiod=1_000_000,
  max_checkpoints=100_000)。H 或关键点数超预算时返回 unproven
  （未证明），与已证明的 unschedulable 严格区分。
- simulate_edf(tasks, horizon, *, arrivals=None)：独立离散时间 EDF
  模拟器（抢占、按绝对截止期选作业，arrivals 可给满足最小到达间隔的
  显式释放序列，缺省 k*T），仅作对照验证，不构成证明的一部分；模拟器
  要求整数执行量。

## 误差口径与限制（EDF 部分）

- 全部判定用精确整数 / Fraction 运算，无浮点误差；U == 1 按精确相等
  判为可调度。
- 模型不含上下文切换开销、多核、锁与自挂起；C 为非负有理数，但模拟器
  只接受整数 C。
- 最多 20 个任务（沿用接口边界）；非法输入抛
  rt_analysis.ValidationError（可定位到任务与字段），analyze_edf 为
  纯函数。

## 命令

    python -m venv .venv && . .venv/bin/activate   # 准备 venv（可选，无第三方依赖）
    python -m unittest discover -s tests -v        # 全部测试（约 0.02 s）
    python demo.py                                 # 固定优先级演示（约 0.05 s）
    python demo_iteration.py                       # EDF 演示：正常+边界+对照（约 0.06 s）

## 文件（新增）

- rt_analysis/edf.py — EdfTask、dbf、key_points、analyze_edf
- rt_analysis/simulate.py — 追加 simulate_edf（独立 EDF 模拟器）
- tests/test_edf.py — 手工推导值、低利用率短截止期失败、U==1、零执行量、
  预算 unproven、暴力逐点核对关键点压缩、枚举短到达序列的模拟对照
- demo_iteration.py — EDF 演示入口

