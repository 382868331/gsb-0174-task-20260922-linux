# rt_analysis — 固定优先级任务的最坏响应时间界

单核固定优先级抢占式任务集的最坏响应时间（WCRT）界分析库。纯 Python
标准库实现，从零编写，不依赖任何第三方同类实现。

本库包含两套分析：

1. **普通固定优先级分析**（`analyze`）：含抖动与优先级天花板资源。
2. **混合关键性分析**（`analyze_mc`，本次迭代新增）：低/高两种执行模式、
   两套执行预算；高关键任务超出低预算触发模式切换，切换后低关键任务
   不再启动新作业。当前限定**无共享资源的单核固定优先级**子集。

## 环境

- Python 3.14.7（仅标准库；实际只用到 3.10+ 的特性）
- Linux 原生运行，安装 Python 后无需安装任何其他依赖
- 核心、测试、演示全部离线：不需要网络、账号、密钥、数据库或 Docker

## 接口

### 普通固定优先级分析

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
- `analyze(tasks, *, max_iterations=1000)`：每任务不动点迭代的搜索预算。
  预算耗尽得到 `iteration_budget_exhausted`（不确定），与已证明的
  `deadline_miss` 严格区分。
- `simulate(tasks, horizon)`：离散时间轴验证模拟器（固定优先级抢占 +
  优先级天花板协议，作业在 `k*T` 释放，临界区位于执行开头）。验证辅助，
  不构成保证的一部分。

### 混合关键性分析（LO/HI 两种模式、两套预算）

```python
from rt_analysis import MCTask, analyze_mc, simulate_mc, CRITICALITY_LO, CRITICALITY_HI

tasks = [
    MCTask("h1", wcet_lo=1, period=10, deadline=10,
           criticality=CRITICALITY_HI, wcet_hi=2),
    MCTask("L",  wcet_lo=1, period=10, deadline=10,
           criticality=CRITICALITY_LO),
    MCTask("h2", wcet_lo=2, period=20, deadline=20,
           criticality=CRITICALITY_HI, wcet_hi=3),
]
report = analyze_mc(tasks)       # -> MCReport
report.guaranteed                # 两种模式下整体是否可保证
report.utilization_lo            # Fraction: 所有任务按 C(LO)
report.utilization_hi            # Fraction: 仅高关键任务按 C(HI)
report.tasks[i].lo               # ModeReport(mode, bound, status, iterations, detail)
report.tasks[i].hi               # 高关键任务的 HI 模式 ModeReport；低关键任务为 None
report.tasks[i].response_time    # 低任务=R(LO)，高任务=max(R(LO),R(HI))；不可保证时 None
report.tasks[i].status           # 同普通分析的三种状态字符串
```

- `MCTask(name, wcet_lo, period, deadline, criticality, *, wcet_hi=None, jitter=0)`：
  - `criticality` 必须是 `"lo"` / `"hi"`（常量 `CRITICALITY_LO` /
    `CRITICALITY_HI`）。
  - 高关键任务必须给 `wcet_hi` 且 `wcet_hi >= wcet_lo`；低关键任务必须
    **不**给 `wcet_hi`（它们在 HI 模式没有预算，因为切换后不再运行）。
  - 其余整数契约与 `Task` 相同：`1 <= deadline <= period`、`wcet_lo >= 1`、
    `jitter >= 0`；**模型不含共享资源**，因此没有临界区字段。
- 任务顺序即固定优先级，两种模式下优先级相同；最多 **20** 个任务。
- `analyze_mc(tasks, *, max_iterations=1000)`：每个任务每种模式各有独立
  迭代预算；耗尽状态为 `iteration_budget_exhausted`（unknown），与
  `deadline_miss` 严格区分。
- 每个任务返回各模式的响应时间界（`ModeReport.bound`）或 unknown
  （状态非 schedulable 时 `bound is None`），以及每轮迭代记录
  `MCIterationRecord(round, value, hi_interference, lo_interference, next_value)`，
  干扰项为 `(任务名, 释放次数)` 序列。
- `simulate_mc(tasks, horizon, *, overruns=())`：整数时间轴验证模拟器，
  语义与分析模型一致：
  - 作业在 `k*T` 释放（抖动按 0 这一合法实现处理）；
  - `overruns` 为 `(任务名, 释放时刻)` 序列，命名的**高关键**作业执行
    需求取 `wcet_hi`；它首次需要超出 `wcet_lo` 的执行时系统切到 HI 模式；
    也可用 `OVERRUN_ALL`（即 `"all"`）让所有有预算余量的高关键作业超限；
  - 切换后低关键作业不再被调度：切换时已释放但未完成的低作业与切换后
    的释放都计入结果的 `abandoned`；`incomplete` 只表示因 horizon 结束而
    未完成（与 abandoned 区分）；`mode_switch` 给出切换时刻（未切换为
    `None`）。它同样只是验证辅助。

## 模型（明示保守）

### 普通分析

- 整数时间模型：`C >= 1`，`T >= 1`，`1 <= D <= T`，`J >= 0`，临界区
  长度 `>= 1` 且同一任务的临界区长度之和 `<= C`。
- 资源采用（立即）优先级天花板协议：资源天花板 = 使用该资源的最高
  优先级任务；临界区**禁止嵌套**。
- 阻塞项 `B_i` = 低优先级任务中、天花板不低于 `i` 优先级的资源上的
  最大单个临界区；输出阻塞来源（任务与资源）。
- 迭代 `R = C_i + B_i + sum_h ceil((R + J_h) / T_h) * C_h`，从
  `R = C_i + B_i` 开始；候选值超过 `D_i - J_i` 判 miss，值稳定判可
  保证。相对名义请求时间的延迟界为 `R_i + J_i`。
- 利用率 `U = sum(C/T)`：`U > 1` 返回 `guaranteed=False`（**不是**
  格式错误）；`U <= 1` 只是必要条件，**不**被当作充分保证。

### 混合关键性分析

- 两种执行模式、两套预算：LO 模式所有任务在 `wcet_lo` 内执行；某高关键
  作业需要超过其低预算的执行时进入 HI 模式，之后高关键作业可执行到
  `wcet_hi`，低关键任务不再启动新作业（已释放但未完成的作业被丢弃）。
- LO 模式界（每个任务，高优先级干扰一律按 C(LO)）：

  `R = C_i(LO) + sum_h ceil((R + J_h)/T_h) * C_h(LO)`
- HI 模式界（仅高关键任务；被分析任务取 C(HI)，每个高优先级任务在整个
  响应窗口内按其最大预算计干扰——高关键取 C(HI)、低关键取 C(LO)）：

  `R = C_i(HI) + sum_{h 高} ceil((R+J_h)/T_h)*C_h(HI) + sum_{l 低} ceil((R+J_l)/T_l)*C_l(LO)`
- 候选值超过 `D_i - J_i` 判 miss，值稳定判可保证。高关键任务的保证界取
  `max(R(LO), R(HI))`，因为它在两种模式下都必须满足截止期。
- 利用率分模式返回：`U(LO) = sum_i C_i(LO)/T_i`（所有任务），
  `U(HI) = sum_{i 高} C_i(HI)/T_i`（仅高关键任务）；任一大于 1 即
  `guaranteed=False`，但同样不是格式错误，小于等于 1 也不是充分条件。
- **为什么 HI 递推刻意保守**：切换后低关键作业确实不再启动，但把它们在
  整个窗口内照算（以及把切换前的高关键作业按 C(HI) 计）只会多算干扰，
  因此不动点对任意切换时刻都是可靠上界。一个更紧的"每个高优先级低关键
  任务只按其自身 R(LO) 计一次"的项在本契约下**不**安全：超限作业恰好
  执行满 C(LO) 的时刻若有低关键作业释放，该作业可以先抢占并把切换本身
  推后，从而第二次产生干扰（回归用例
  `TestConservatismRegression` 与枚举交叉检查记录了该反例）。利用切换
  时刻信息的更紧分析（AMC-max 一类）不在本次范围内。

## 错误语义

- 非法输入抛出 `rt_analysis.ValidationError`（`ValueError` 子类），
  消息定位到任务下标、任务名与字段，不静默纠正。
- 数值输入拒绝 NaN/±Infinity；整数参数拒绝 bool；非整数值（如 2.5）
  拒绝；有限的整数值浮点（如 4.0）按数值等价接受。
- `analyze` / `analyze_mc` / `simulate` / `simulate_mc` 都是纯函数：
  校验失败时不留下任何部分变更。
- `simulate_mc` 对 `overruns` 额外校验：任务必须存在且为高关键、释放
  时刻必须在 horizon 内且等于周期的整数倍、该任务必须有预算余量
  （`wcet_hi > wcet_lo`）。

## 误差口径

分析全程使用精确整数与 `Fraction` 有理数运算，**无浮点误差**；响应
时间界、干扰次数均为精确整数。利用率以 `Fraction` 返回，仅 demo/摘要中
的 `float(...)` 显示存在 <= 1e-4 的显示舍入，不参与任何判定。边界（如
`R == D - J`、`U == 1`）按精确相等判定为可保证。

## 设计取舍与已知限制

- 两套分析都是**保守充分**条件而非精确充要条件：可保证的结论可靠，判
  miss 的结论在该保守模型内成立，但不排除存在更精细分析可通过的实例；
  混合关键性 HI 界有意偏松（见上）。
- 混合关键性迭代当前只覆盖**无共享资源、单核、固定优先级、两级关键性、
  抖动取 0 实现**的子集；不包含资源上限/锁协议、遗留持锁工作、多关键级、
  多核、任务间任意依赖，也不做完整轨迹穷举；不宣称完整行业标准
  （如完整 OSEK/AUTOSAR/AMC 语义）兼容。
- 模拟器各自只实现一种具体确定性场景（普通模拟器：临界区位于执行开头、
  `k*T` 同步释放；MC 模拟器：指定作业超限、其余作业按低预算执行），
  用于"不超过界"的实例级核对，不枚举最坏情形。
- 上限 20 任务是接口边界；演示与测试使用小规模确定性数据，未做上限
  规模的压力测试；未单列大规模性能指标。

## 命令

```bash
python -m unittest discover -s tests -v   # 全部测试（约 0.03 s）
python demo.py                            # 普通 + 混合关键性推导链、边界与拒绝（约 0.05 s）
```

## 文件

- `rt_analysis/core.py` — 普通数据结构、校验、`analyze`
- `rt_analysis/simulate.py` — 普通离散时间轴 PCP 验证模拟器
- `rt_analysis/mixed.py` — 混合关键性数据结构、校验、`analyze_mc`
- `rt_analysis/mc_simulate.py` — 混合关键性整数时间轴验证模拟器
- `tests/` — 校验、普通分析、普通模拟、混合关键性分析、混合关键性模拟、
  独立小规模枚举交叉检查六组测试（期望值均手工推导或由独立参考调度器
  生成，不调用被测实现得到期望值）
- `demo.py` — 固定输入演示（普通 + 混合关键性）
