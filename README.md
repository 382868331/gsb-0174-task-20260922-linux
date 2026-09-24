# rt_analysis — 固定优先级任务的最坏响应时间界

单核固定优先级抢占式任务集的最坏响应时间（WCRT）界分析库。纯 Python
标准库实现，从零编写，不依赖任何第三方同类实现。

提供两套同构分析：

- **单关键性 + 共享资源**：`analyze` / `simulate`，优先级天花板协议（PCP）。
- **混合关键性双模式（LO/HI）**：`analyze_mc` / `simulate_mc`，两种执行
  模式与两套 WCET 预算、模式切换界；本次迭代新增，范围限定**单核、无共享
  资源**。

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

### 混合关键性双模式入口 `analyze_mc`

```python
from rt_analysis import Task, analyze_mc, CRITICALITY_HI, CRITICALITY_LO

tasks = [
    Task("h", wcet=1, period=6,  deadline=6,  criticality=CRITICALITY_HI, wcet_hi=2),
    Task("m", wcet=3, period=12, deadline=12, criticality=CRITICALITY_LO),
    Task("l", wcet=2, period=24, deadline=24, criticality=CRITICALITY_HI, wcet_hi=4),
]
mc = analyze_mc(tasks)          # -> MCReport
mc.guaranteed                   # LO/HI/AMC-rtb 全部界已知且在 D-J 内
mc.utilization_lo               # Fraction: sum C(LO)/T（所有任务）
mc.utilization_hi               # Fraction: sum C(HI)/T（仅 HI 任务）
mc.tasks[i].criticality         # "LO" | "HI"
mc.tasks[i].lo                  # ModeBound：LO 模式界（每个任务都有）
mc.tasks[i].hi                  # ModeBound：稳态 HI 界（仅 HI 任务，LO 为 None）
mc.tasks[i].amc_rtb             # ModeBound：模式切换界（仅 HI 任务，LO 为 None）
b.kind    # "LO" | "HI" | "AMC-rtb"
b.status  # "known" | "deadline_miss" | "unknown"
b.value   # known 时为精确整数界；其余为 None
b.limit   # D - J
b.iterations  # 每轮 IterationRecord
```

- `Task` 新增两个带默认值的字段：`criticality="LO"`、`wcet_hi=None`。
  旧的 `Task(...)` 位置/关键字调用与 `analyze` 行为完全不变；扩展后的
  任务也能照常传给 `analyze`（它只使用 LO 预算 `wcet`）。
- 约束：HI 任务必须给 `wcet_hi`，且 `wcet_hi >= wcet >= 1`；LO 任务
  不得给 `wcet_hi`（切换后不再调度）。至少有一个 HI 任务；任务集
  **不得含临界区**（本迭代范围无共享资源，违反即 `ValidationError`）。
- 三个界（均从 `R = 自身预算` 开始迭代，超过 `D-J` 判 miss，稳定判
  known，预算耗尽判 **unknown**）：
  - **LO**：所有任务按 LO 预算执行（切换前行为）。
  - **HI**：稳态高模式，只有高优先级 **HI** 任务按 `wcet_hi` 干扰。
  - **AMC-rtb（模式切换界）**：被分析作业按 HI 预算；高优先级 LO 任务
    按 LO 预算干扰、高优先级 HI 任务按 HI 预算干扰（即 AMC-rtb 递推
    `R = C_i(HI) + Σ_{h∈hp,LO} ceil((R+J_h)/T_h) C_h(LO)
                       + Σ_{h∈hp,HI} ceil((R+J_h)/T_h) C_h(HI)`）。
    LO 任务在切换后不启动新作业，故不给 HI/AMC 界（字段为 `None`）。
- `analyze_mc(tasks, *, max_iterations=1000)`：每个界独立计预算；
  **unknown（搜索预算耗尽）与已证明的 deadline_miss、已知界严格三分**，
  任一适用界为 miss/unknown 则该任务与整体 `guaranteed=False`。
- `simulate_mc(tasks, horizon, *, switch_time=None)`：双模式验证模拟器。
  `switch_time=None` 为纯 LO 运行；给定整数切换时刻 `1..horizon` 时，
  该时刻未完成的 HI 作业升级到 HI 预算（驱动切换的超时）、未完成 LO
  作业被丢弃，且该时刻起不再释放 LO 作业。返回 `MCSimResult`，含每个
  完成作业的 `response`/`switched` 标记、`dropped` 与 `incomplete` 列表。
  与 `simulate` 一样只生成实例核对，不构成保证、不枚举最坏情形。

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

### 混合关键性模型（LO/HI，本次迭代范围）

- 两种执行模式、HI 任务两套预算 `C(HI) >= C(LO) >= 1`；HI 作业执行
  超过 LO 预算即触发 LO→HI 切换。切换后 **LO 任务不再启动新作业**
  （切换瞬间未完成的 LO 作业被丢弃），HI 任务按 HI 预算继续。
- 范围**严格限定**：单核、固定优先级、隐式截止期 `D <= T`、抖动沿用
  同一整数模型、**无共享资源/无锁**。不含资源上限（priority-ceiling/
  SRP 等）锁协议、不含切换时遗留持锁工作、不做完整轨迹穷举；这些
  是后续迭代的事，本次不宣称覆盖。
- 模式切换界采用 **AMC-rtb**（response-time bound，非 AMC-max）：对
  切换时刻不做最大化，LO/HI 高优先级干扰在同一条递推里按各自预算
  各计一次，是经典、保守、可手算的界；`analyze_mc` 不实现 AMC-max。
- 双利用率：`U_LO = Σ C(LO)/T`（全部任务）、`U_HI = Σ_{HI} C(HI)/T`
  （仅 HI 任务）。任一 `> 1` 报 `guaranteed=False`（非格式错误）；
  两者 `<= 1` 同样只是必要条件。
- 三个界互相独立判定：LO 界已知而 HI 界 miss 的情况会如实分别报告。

## 错误语义

- 非法输入抛出 `rt_analysis.ValidationError`（`ValueError` 子类），
  消息定位到任务下标、任务名与字段，不静默纠正。
- 数值输入拒绝 NaN/±Infinity；整数参数拒绝 bool；非整数值（如 2.5）
  拒绝；有限的整数值浮点（如 4.0）按数值等价接受。
- 混合关键性新增校验：非法 `criticality` 取值、HI 任务缺 `wcet_hi`、
  LO 任务给 `wcet_hi`、`wcet_hi < wcet`、`analyze_mc` 收到含临界区的
  任务或全 LO 任务集，均抛定位明确的 `ValidationError`；`switch_time`
  非法（<1、bool、> horizon）同样拒绝。
- `analyze`/`analyze_mc`/`simulate`/`simulate_mc` 均为纯函数：校验
  失败时不留下任何部分变更（任务对象为不可变 dataclass）。

## 误差口径

分析全程使用精确整数与 `Fraction` 有理数运算，**无浮点误差**；响应
时间界、阻塞项、干扰次数均为精确整数。利用率以 `Fraction` 返回，
仅 demo/摘要中的 `float(...)` 显示存在 <= 1e-4 的显示舍入，不参与
任何判定。边界（如 `R == D - J`、`U == 1`）按精确相等判定为可保证；
混合关键性三个界同理，`unknown` 仅在整数不动点确实未在预算内收敛或
证伪时出现，不依赖任何浮点容差。

## 设计取舍与已知限制

- 分析是**保守充分**条件而非精确充要条件：可保证的结论可靠，判 miss
  的结论在该保守模型内成立，但不排除存在更精细分析可通过的实例。
- 单关键性模型不含上下文切换开销、缓存效应、多核、任务间任意依赖；
  仅覆盖本题契约的周期/偶发 + 抖动 + 天花板协议子集，不宣称完整
  行业标准（如完整 OSEK/AUTOSAR 语义）兼容。
- 混合关键性模型为 **AMC-rtb**：对切换时刻不做最大化，通常比
  AMC-max 更保守（界可能更大）；不含资源上限锁协议、切换时遗留
  持锁工作、协议容错重启语义，也不做完整切换轨迹穷举；无共享资源
  是硬范围（含临界区直接报错）。
- 两个模拟器都采用一种具体实现（临界区位于执行开头/无锁、`k*T`
  同步释放、显式单一切换时刻），用于“不超过界”的实例级核对，不
  枚举最坏情形；`simulate_mc` 不模拟抖动（与 `simulate` 一致）。
- 上限 20 任务是接口边界；演示与测试使用小规模确定性数据，未做
  上限规模的压力测试。

## 命令

```bash
python -m unittest discover -s tests -v   # 全部测试（75 个，约 0.01 s）
python demo.py                            # PCP 推导链 + 混合关键性 LO/HI/AMC-rtb、
                                          # 模式切换实例、边界与实际错误（约 0.05 s）
```

演示真实计算、无 sleep、无预录答案；混合关键性部分含：可保证的正常
结果与完整迭代链、一个具体切换调度（LO 作业被丢弃、HI 作业升级）、
恰好截止与超 1 两个边界，以及两类实际触发的 `ValidationError`。

## 文件

- `rt_analysis/core.py` — 数据结构、校验、`analyze`、`analyze_mc`（LO/HI/AMC-rtb）
- `rt_analysis/simulate.py` — 离散时间轴 PCP 模拟器 `simulate` 与双模式
  模式切换模拟器 `simulate_mc`
- `tests/` — 校验、分析语义、模拟核对与混合关键性四组测试
  （期望值均手工推导/手算时间轴，独立参考不调用被测实现取期望）
- `demo.py` — 固定输入演示（单关键性 + 混合关键性）
