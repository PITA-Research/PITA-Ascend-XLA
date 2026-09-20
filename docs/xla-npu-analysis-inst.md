# XLA-NPU 问题分析 SOP

本文用于指导后续 XLA-NPU 后端问题排查，尤其适用于以下类型：

- CPU/NPU 结果不一致；
- NPU output buffer 被污染；
- while / conditional / copy / tuple 相关行为异常；
- ExecuteThunks 前后 buffer 值发生非预期变化；
- 疑似 buffer alias、调度顺序、in-place op 或 host readback 问题。

核心原则：

> 不要先猜 thunk 写错或 race。先用执行前后 buffer 快照定边界，再用 allocation
> watch 找第一个污染 thunk，最后回到 HLO 的 root / while / copy / schedule 结构
> 解释为什么该 buffer 会被合法但错误地复用或改写。

## 0. 测试和编译前置

排查 NPU JAX 问题时，必须先确认测试进程加载的是当前 xla-npu build 产物。
不要只设置 `JAX_PLATFORMS=npu`，否则可能看不到 NPU backend 或加载到错误插件。

MJX / JAX NPU 测试环境参考：

```bash
cd /data/CodeSpace/mjx_pita-main

source /usr/local/Ascend/cann/set_env.sh
source /data/xla-npu-master/build/xla_npu_env

export XLA_NPU_LOG_LEVEL="${XLA_NPU_LOG_LEVEL:-ERROR}"
export SOC_VERSION=Ascend910B1
export USE_OLD_CANN_STYLE=true
export PJRT_NAMES_AND_LIBRARY_PATHS=npu:/data/xla-npu-master/build/code/xla/bazel-bin/xla/xla_npu/pjrt/c/pjrt_c_api_npu_plugin.so
export JAX_PLATFORMS=npu,cpu
export USE_ORIGINAL_CHOLESKY="${USE_ORIGINAL_CHOLESKY:-1}"
export PYTHONPATH=src:$PYTHONPATH

# 定位编译/cache 相关问题时建议每轮使用独立目录，避免复用旧 auto-fuse artifact。
export NPU_AUTO_FUSE_COMPILE_ARTIFACT_ROOT_DIR=/tmp/npu_fuse_cache_<case_name>
mkdir -p "$NPU_AUTO_FUSE_COMPILE_ARTIFACT_ROOT_DIR"
```

`/data/CodeSpace/mjx_pita-main/scripts/train.sh` 已包含上述 NPU 环境设置。运行
MJX 训练时优先直接使用：

```bash
cd /data/CodeSpace/mjx_pita-main
bash scripts/train.sh
```

运行独立诊断脚本时，需要手动复用 `scripts/train.sh` 中的环境变量。例如：

```bash
cd /data/CodeSpace/mjx_pita-main
source /usr/local/Ascend/cann/set_env.sh
source /data/xla-npu-master/build/xla_npu_env
export XLA_NPU_LOG_LEVEL=ERROR
export SOC_VERSION=Ascend910B1
export USE_OLD_CANN_STYLE=true
export PJRT_NAMES_AND_LIBRARY_PATHS=npu:/data/xla-npu-master/build/code/xla/bazel-bin/xla/xla_npu/pjrt/c/pjrt_c_api_npu_plugin.so
export JAX_PLATFORMS=npu,cpu
export USE_ORIGINAL_CHOLESKY=1
export ALIGN_FORCE_RNG_PATCH=1
export PYTHONPATH=src:$PYTHONPATH
export NPU_AUTO_FUSE_COMPILE_ARTIFACT_ROOT_DIR=/tmp/npu_fuse_cache_cmd_diag
mkdir -p "$NPU_AUTO_FUSE_COMPILE_ARTIFACT_ROOT_DIR"

python tools/diagnose_command_resample_stages.py --num-envs 2 --start-step 0
```

修改 xla-npu 后，必须在测试前重新编译并安装插件。编译命令固定为：

```bash
cd /data/CodeSpace/xla-npu-master
bash build/build.sh install
```

不要额外加 `PARALLEL_JOBS=16` 等并行度环境变量，除非当前任务明确要求修改编译并行度。
测试时确认 `PJRT_NAMES_AND_LIBRARY_PATHS` 指向的
`pjrt_c_api_npu_plugin.so` 就是上述 build 产物。

## 1. 问题摘要

先把问题描述成可验证事实，不要直接写推测。

需要记录：

- CPU 期望值；
- NPU 实际值；
- 复现命令；
- 是否稳定复现；
- 影响哪些 output leaf；
- 是否和 batch size、num_envs、cache、auto-fuse artifact 相关；
- 是否只在 while / conditional / scan / reset 等控制流路径出现。

模板：

```text
CPU 期望：
NPU 实际：
复现命令：
是否稳定：
影响 output leaf：
相关 HLO module：
相关 cache/dump 目录：
```

示例：

```text
CPU: literal_zero / info_step = [0, 0]
NPU: literal_zero / info_step = [2, 2]
稳定复现
发生在 reset/counter/while 相关路径
```

## 2. 先定位污染边界

第一步目标是回答：问题发生在 Python、PJRT 输出组装、ExecuteThunks 内部，还是更早的编译阶段。

建议 debug 点：

```text
ExecuteThunks 前 dump output buffer
ExecuteThunks 后 dump output buffer
必要时按 ShapeIndex dump 指定 output leaf
```

判断标准：

```text
执行前已经错：
  优先查 buffer assignment、output alias、compile result、executable output info。

执行前正确，执行后错误：
  污染发生在 thunk 执行内部，继续查 thunk 顺序、allocation alias、in-place op、
  while/copy 调度。

执行后正确，host 侧看到错误：
  查 PJRT buffer 包装、tuple output assemble、D2H/readback。
```

结论模板：

```text
ExecuteThunks 前 output buffer：
ExecuteThunks 后 output buffer：
边界结论：
  [ ] 编译期已错
  [ ] ExecuteThunks 内污染
  [ ] host readback/output assemble 问题
```

## 3. Output Leaf 反查 Allocation

确认污染发生在 ExecuteThunks 内部后，下一步要把异常 output leaf 映射到具体 device allocation。

需要收集：

- ShapeIndex；
- OutputInfo；
- allocation index；
- offset；
- size；
- before/after 值。

记录模板：

```text
| ShapeIndex | Allocation | Offset | Size | Before | After | 备注 |
|---|---:|---:|---:|---|---|---|
| {0} | 237 | 0 | 8 | [0,0] | [2,2] | first polluted leaf |
```

如果多个 output leaf 同时异常，需要先判断：

- 是多个 leaf 各自被同一个 producer 污染；
- 还是第一个 leaf 被污染后，通过后续 copy 扩散到其它 leaf。

## 4. Thunk 级 Watch

对关键 allocation 做 thunk 级别 watch，用来回答：到底是哪一个 thunk 第一次写坏 buffer。

建议方式：

```text
对指定 allocation index：
  每个 thunk 执行前 dump
  每个 thunk 执行后 dump
  只打印发生变化的 allocation
```

记录模板：

```text
| Thunk | Allocation | Before | After | 判断 |
|---|---:|---|---|---|
| while.56 | 237 | [0,0] | [2,2] | 首次污染 |
| copy.207 | 238 | [0,0] | [2,2] | 复制已污染 source |
| copy.208 | 239 | [0,0] | [2,2] | 复制已污染 source |
```

判断标准：

```text
某个 while 后首次变化：
  查 while carry alias、while body 写入、root live-out 是否复用。

某个 copy 后变化：
  查 copy source 是否已经污染，copy 是否被调度到 mutation 之后。

某个 aclnn op 后变化：
  查 input/output 是否 alias，op 是否支持 in-place，thunk buffer slice 是否错。
```

## 5. 回到 HLO 和 Schedule

runtime watch 只能说明“谁写了”，HLO 和 schedule 才能解释“为什么允许这么写”。

重点比较以下阶段：

```text
NPU pipeline 后 HLO
NpuScheduleModule 后 HLO
AddSpecialCaseCopies 前 HLO
AddSpecialCaseCopies 后 HLO
scheduled-npu-module verifier 前后 HLO
buffer assignment / allocation dump
```

重点检查：

```text
root tuple 是否直接返回 constant/broadcast constant？
同一个 value 是否也进入 while init tuple？
while body 是否会更新这个 carry？
root output 和 while carry 是否可能共享 allocation？
default copy insertion 是否把 root 改写成 tuple/GTE/copy？
copy 是否依赖包含 while_result 的 tuple？
copy 是否被 schedule 到 mutation 之后？
```

典型危险模式：

```text
broadcast(constant 0)
  同时作为 root output
  同时作为 while init carry

while body 更新 carry:
  [0,0] -> [2,2]

root output 因为缺少 copy，复用了 while carry buffer。
```

多 output leaf 的扩散模式：

```text
tuple.22 = tuple(zero, zero, zero, zero, while_result)

root = tuple(
  gte(tuple.22, 0),
  copy(gte(tuple.22, 1)),
  copy(gte(tuple.22, 2)),
  copy(gte(tuple.22, 3)),
)
```

危险点：

```text
第 0 个 leaf 没有 copy，可能直接被 while carry 污染。
后续 leaf 虽然有 copy，但 copy source 依赖包含 while_result 的 tuple，
因此 copy 可能被调度在 while 后，复制到已污染值。
```

## 6. 根因分类

排查时必须区分以下几类问题，避免把 deterministic alias 问题误判成 race。

```text
race:
  多次运行不稳定，插同步/延迟后行为变化，读写顺序不确定。

buffer alias:
  稳定复现，同一 allocation 被两个语义上不该共享的值复用。

schedule/copy 顺序:
  copy/source 的依赖使 copy 被排到 mutation 之后，复制了污染值。

in-place op alias:
  某个 ACLNN op input/output alias 后结果异常，通常发生在具体 op 执行后。

output assemble/readback:
  ExecuteThunks 后 device buffer 正确，但 host/PJRT 侧看到错误。
```

结论模板：

```text
根因分类：
  [ ] race
  [ ] buffer alias
  [ ] schedule/copy 顺序
  [ ] in-place op alias
  [ ] output assemble/readback

最终判断：
证据：
```

## 7. 修复策略

优先选择窄范围 correctness copy 或 alias 修复，不要一开始扩大到全局禁 alias。

修复原则：

```text
只在确认的危险模式上插 copy。
不要全局禁止 alias，除非证明 alias policy 本身系统性错误。
不要大范围改 buffer assignment。
不要让 runtime thunk 长期规避编译期 alias 问题。
修复点要靠近语义缺口，而不是靠近症状。
```

修复设计模板：

```text
修复位置：
修复条件：
插入/改写动作：
为什么范围足够窄：
为什么 pass 顺序正确：
是否需要更新 schedule：
是否需要 DCE：
是否会影响其它 root output：
```

示例条件：

```text
entry root 是 tuple
root 顶层 operand 是 array
operand 是 constant 或 broadcast(constant) 派生
operand 通过 tuple/get-tuple-element/copy/bitcast 用户链到达 while
```

示例动作：

```text
在 root operand 位置插入 copy。
让 root live-out 拥有独立 buffer。
```

## 8. Pass 顺序检查

很多 NPU 后端问题不是“要不要插 copy”，而是“在哪个 pass 前后插 copy”。

必须回答：

```text
在哪个 pass 前插？
在哪个 pass 后插？
插完 schedule 是否有效？
后续 pass 会不会重写 root 结构？
后续 DCE 会不会删掉 copy？
后续 verifier 是否能接受？
```

推荐记录：

```text
当前 pass 顺序：
修复插入点：
插入前 HLO：
插入后 HLO：
后续 pass 改写：
最终 scheduled HLO：
```

示例正确顺序：

```text
NpuScheduleModule
RemoveUnnecessaryCopies
clear schedule
AddNpuRootCopiesForWhileInitConstants
AddSpecialCaseCopies
HloDCE
saved_schedule.Update()
set_schedule
scheduled-npu-module verifier
```

错误顺序示例：

```text
AddSpecialCaseCopies
AddNpuRootCopiesForWhileInitConstants
```

风险：

```text
默认 AddSpecialCaseCopies 之后 root 可能已经被改写成 tuple/GTE/copy。
此时再补 copy 可能已经晚了，copy 仍可能在 mutation 后复制污染值。
```

## 9. 验证策略

验证必须覆盖“单点修复”和“污染扩散”两类 case。

建议至少准备：

```text
最小单 leaf case：
  验证核心根因是否消除。

repeated leaf / many zero case：
  验证 tuple packing、GTE、copy insertion 不会扩散污染。

真实业务 case：
  验证完整业务路径没有遗漏。

全量回归：
  确认没有引入新失败。
```

验证记录模板：

```text
| Case | 目的 | 结果 | 备注 |
|---|---|---|---|
| minimal_zero | 单 leaf root/carry alias | OK | zero=[0,0] |
| many_zeros | repeated root operand 污染扩散 | OK | all zeros=[0,0] |
| full_reset | 真实业务路径 | OK | counters reset |
| op_e2e | 全量回归 | known failure only | 记录历史证据 |
```

如果全量回归存在失败，必须区分：

```text
是否本次新增失败？
是否已有历史记录？
失败 signature 是否一致？
是否与本次修改路径相关？
```

## 10. Debug 清理

定位结束后必须清理临时 debug。

检查项：

```text
rg 检查 debug env 名称
rg 检查 debug helper/class/function 名称
确认 runtime 临时文件无 diff
确认没有临时 dump 路径硬编码
确认没有临时日志级别提升
重新 install 编译
跑定向复现
跑全量回归
```

示例命令：

```bash
rg -n "XLA_NPU_DEBUG|NPU_OUTPUT_LEAF_DEBUG|NPU_WATCH_ALLOCATION|DebugWatch|Dump.*Debug" xla_npu || true
git diff --stat
git diff -- xla_npu/service/npu_executable.cc xla_npu/backend/runtime
```

## 11. 最终记录模板

每次问题修复后，建议在 `git_log_details/` 下补充记录。

模板：

```text
commit <TBD>
Author:
Date:

    fix: <one-line summary>

背景
----

问题现象
--------

定位过程
--------

根因一
------

根因二
------

解决方案
--------

为什么修复顺序正确
------------------

影响范围
--------

涉及文件
--------

验证结果
--------

未纳入本次修复的问题
--------------------

总结
----
```

## 12. 快速 Checklist

```text
[ ] CPU/NPU 差异已收敛成最小复现
[ ] ExecuteThunks 前后 buffer 已确认
[ ] output leaf 到 allocation 已映射
[ ] 第一个污染 thunk 已找到
[ ] HLO root / while init / copy / schedule 已串起来
[ ] 根因分类已明确，不把 alias 误判为 race
[ ] 修复范围足够窄
[ ] pass 顺序已验证
[ ] schedule 更新/DCE/verifier 已考虑
[ ] 单 leaf case 通过
[ ] many/repeated leaf case 通过
[ ] 真实业务 case 通过
[ ] 全量回归已跑或明确说明未跑原因
[ ] debug 代码已清理
[ ] git_log_details 已记录
```
