# 固定优先级任务最坏响应时间界（WCRT RTA）

单核固定优先级抢占调度的最坏响应时间可调度性分析，纯 Python 标准库实现的可导入题库。

- **Python 版本**：Python 3.14.7（使用 `from __future__ import annotations`、dataclass、`fractions`；3.10+ 亦可运行，交付验证以 3.14.7 为准）
- **依赖**：无第三方依赖，仅标准库（`math`、`fractions`、`dataclasses`、`typing`）
- **离线**：不访问网络、账号、密钥、数据库、中间件、云 API 或 Docker；所有示例数据由仓库内确定性代码生成

## 运行命令

```bash
python -m unittest discover -s tests -v   # 必要测试（34 个）
python demo.py                            # 固定输入演示，约 0.05 秒
```

均在仓库根目录、Linux 原生执行，无需安装任何包。

## 公开接口

```python
from wcrt import Task, Resource, analyze

tasks = [
    Task(name, priority, c, t, d, j=0, sections=()),
    # priority：整数，数值越小优先级越高，同集内唯一
    # c/t/d：正整数；j：非负整数；契约 d <= t
    # sections：[(资源名, 临界区长度), ...]，扁平、禁止嵌套；各段长度之和 <= c
]
resources = [Resource(name, ceiling)]  # ceiling 用与 priority 相同的数值尺度

result = analyze(tasks, resources)     # resources 可省略
result.guaranteed                      # bool：U<=1 且每个任务迭代稳定且未越界
result.utilization_exact               # fractions.Fraction，精确利用率
result.utilization_over                # 精确 U > 1（不是格式错误）
for tr in result.tasks:                # 按优先级从高到低
    tr.status            # 'guaranteed' | 'miss'
    tr.response_time     # 稳定界（int）；miss 时为 None
    tr.deadline_limit    # D - J
    tr.blocking          # B_i
    tr.blocking_source   # (低优先级任务名, 资源名, 临界区长度) 或 None
    for it in tr.iterations:
        it.index, it.bound, it.interference, it.interference_workload, ...
```

数据结构为冻结 dataclass（`Task`/`Resource`/`Iteration`/`TaskResult`/`AnalysisResult`）。
`result.task(name)` 可按名取结果；`AnalysisResult.utilization`（float）仅供展示。

异常：

- `wcrt.TaskSetError`（`ValueError` 子类）：违反契约的输入，错误信息带 `tasks[i](name).field` 形式的可定位路径。
- `wcrt.BudgetExhaustedError`：迭代预算耗尽（见下），**与“证明 miss”严格区分**。
- `TypeError`：类型错误，包括 bool 冒充整数、float 冒充 int 等。

## 分析模型（明示的保守模型）

对任务 *i*，B_i 为低优先级任务中**可能阻塞 i 的最大单个临界区**，响应时间递归：

```
R0 = C_i + B_i
R_{k+1} = C_i + B_i + Σ_{h ∈ hp(i)} ceil((R_k + J_h) / T_h) · C_h
```

- R_{k+1} == R_k：迭代稳定，R 即最坏响应时间界；
- 任一轮 R_k > D_i − J_i：**证明错过截止期（miss）**，停止迭代；
- 整集可保证当且仅当精确利用率 U = Σ C/T ≤ 1 **且**每个任务都稳定不越界。
  U > 1 只令结果 `guaranteed=False`，不抛异常；U ≤ 1 也绝不当作充分保证。

资源采用**给定优先级天花板**：`Resource.ceiling` 由调用方给出，
`analyze` 校验它等于所有使用者中的最高优先级（`min(使用者优先级)`），不一致即拒绝；
悬空资源（声明但无人使用）同样拒绝。临界区扁平声明，**禁止嵌套**。

### 关键设计取舍

1. **天花板协议语义（ICPP/PCP）**：候选临界区资格只看天花板——
   只要 `ceiling(r) ≤ P_i`，即使 i 自身不使用 r，i 也可能在释放点被持低优先级锁的系统天花板挡住，
   因此计入 B_i。这是天花板协议相对优先级继承协议更保守、更简单的口径，
   不依赖具体持锁时刻列表。禁止嵌套保证 i 至多被一个临界区阻塞（B_i 为单段最大值）。
2. **抖动口径**：J 为释放抖动（到达后最晚 J 个时间单位释放）。干扰回溯窗口按
   `ceil((R+J_h)/T_h)` 计；任务自身的可用截止期缩短为 `D_i − J_i`。
3. **临界瞬时假设**：分析假设所有高优先级作业与 i 的临界释放点对齐（含抖动造成的一个积压作业），
   是标准的充分（可能悲观）上界；**界不保证紧**。
4. **整数时间**：C/T/D/J/临界区长度均为整数，递推全程整数运算；利用率用
   `fractions.Fraction` 精确比较。**判定路径不出现浮点、不使用任何容差。**
5. **每轮记录**：`iterations[0]` 是 R0 = C+B（干扰表为空）；稳定需要再多一轮相等值，
   故界序列包含重复的收敛值；`Iteration.interference` 给出每个高优先级任务当轮计入的作业数。
6. **miss 的 last_bound**：miss 时 `last_bound` 只是触发停止的越界值，不是“实际最坏响应时间”。

### 预算与无解的区分

未 miss 时 R 为单调递增整数且 R ≤ D−J，非收敛轮至少 +1，故收敛轮数天然有界；
另有硬保护常量 `wcrt.MAX_ITERATIONS_HARD`（默认 10^7）。耗尽时抛
`BudgetExhaustedError`（搜索预算问题），绝不会被报告成 miss（数学结论）。测试中通过临时压低该常量验证二者分开。

## 数值与错误语义

- 数值输入拒绝 `NaN`/`Infinity`（含负无穷），报 `ValueError` 并定位字段；
- 整数参数拒绝 `bool`（`True/False` 不静默当作 1/0）；拒绝 `2.0` 这类 float 冒充 int；
- 非法输入一律报错，不做静默纠正、不做隐式截断/取整；
- `analyze` 为纯函数，不修改输入、不持有全局可变状态，失败时不存在部分变更。

## 误差口径（README 明确）

- 分析输出的所有界、干扰次数、利用率判定均为**精确整数/有理数结果，误差为 0**；
- 唯一浮点是 `AnalysisResult.utilization`（float 展示值），仅用于打印，任何判定都不使用它；
- 测试断言全部为整数精确相等或精确不等，**不以宽容差掩盖边界错误**。

## 独立参考（不调用被测核心）

`tests/timeline.py` 是另一套独立实现：逐 tick 的离散时间轴 ICPP 调度器，
用普通 dict 描述任务，不导入 `wcrt`。它用于：

- 验证临界瞬时下调度实跑的实例响应**不超过**分析界（无资源场景下恰好相等）；
- 在对抗抖动相位下真实复现截止期错过；
- 固定种子的随机小规模交叉核对（补充证据）。

测试中的期望值主要来自写在断言旁的纸笔整数递推；时间轴只做上界/复现核对，
“往返一致”仅作补充，不作为唯一证据。

## 输入上限

- 任务数 ≤ 20，资源数 ≤ 20（接口边界；短演示不穷举上限）；
- 整数无上溢问题（Python 大整数）；但不切实际的巨值会使迭代轮数按 D−J 增长。

## 已知限制

- 仅单核、固定优先级、抢占式、周期/最小间隔零星任务；约束截止期 D ≤ T；
- 不支持截止期 D > T、非整数时间、自挂起、到达模型相关性、亲和性/多核、共享栈、任务间非资源依赖；
- 只实现本题契约的天花板协议子集，不声称兼容 POSIX/AUTOSAR 等行业标准，不模拟具体 RTOS；
- 分析是充分上界：可保证 ⇒ 安全；判 miss 是在该保守模型下证明越界（模型内精确）；
  不推断具体操作系统的实际执行轨迹；
- 临界区在任务执行程序内的位置不影响 B_i（取最大单段），测试用时间轴采用确定性铺放，仅为模拟需要。

## 文件

```
wcrt.py             核心库（analyze 及公开数据结构/异常）
demo.py             固定输入演示（正常推导链、抖动 miss、真实拒绝边界）
tests/test_wcrt.py  必要测试（手算期望值 + 时间轴核对 + 错误/预算语义）
tests/timeline.py   独立离散时间轴 ICPP 调度参考（不导入 wcrt）
requirements.txt    声明：仅标准库
```
