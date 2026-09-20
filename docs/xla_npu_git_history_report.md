# xla_npu Git 历史改动统计与重点代码审查报告

生成日期：2026-09-20（UTC）。统计快照：`23271f4d188fcd8f93a6871872b6483363b04fec`。本报告只读审查代码，未修改源文件、暂存或提交任何文件。

后续整改记录（2026-09-20）：F6、F7 已在工作区修复，尚未提交。已移除无效诊断变量及重复分支，将常态 while/conditional 和 carry 详情日志降为 DEBUG，并清理报告列出的 27 处尾随空白。`git diff --check` 与 `git diff --check d32d52c -- xla_npu/` 均通过；本次未运行编译或 NPU 回归。以下历史统计、审查位置及原始发现仍对应上述提交快照。

## 1. 结论

当前分支共 109 次提交，106 次涉及 `xla_npu/`。首次导入 410,634 行；后续累计新增 31,055 行、删除 17,357 行，累计改动 **48,412 行**，净增加 **13,698 行**。HEAD 下有 1,363 个文件、424,332 行；历史中共出现 1,365 个文件路径。

后续改动最多的目录是 `dependency`（27,059 行）、`backend`（12,816 行）、`service`（7,983 行）。但 `dependency` 主要受注释翻译影响：单次 `dbe12ec` 在整个统计范围内产生 28,206 行增删，占后续改动的 **58.26%**。排除这次提交作参考后，改动集中在 `backend`（11,976 行）和 `service`（7,467 行）。不能把累计行数直接解释为功能开发量。

文件层面，`npu_aclnn_thunk.cc` 累计改动 **9,359 行**、经历 70 次后续修改，占全部后续改动的 **19.33%**；现有文件长 6,297 行，是优先重构和补测试的对象。其次是 `npu_ir_emitter_unnested.cc`（1,267 行）和 `npu_thunk.cc`（884 行）。

审查发现：三角矩阵重写的匹配条件不等价、while carry 解析失败未传播错误，以及初始版本沿用至今的 NaN/Inf 静默改写；另有异步拷贝失败时的资源生命周期风险、遗留无效代码、过长函数和尾随空白。下面逐项提供位置、证据、影响、历史来源和建议。

## 2. 统计口径与边界

- 范围：当前 `HEAD` 可达的全部提交，从唯一根提交 `d32d52cd75da77d0a37981d84e353a1f9bc5b52e`（2026-05-15）到 `23271f4d188fcd8f93a6871872b6483363b04fec`（2026-09-20）；不是所有远端分支，也不推断更早的上游开发历史。仓库不是 shallow clone，当前历史无 merge commit。
- 路径范围严格为 `xla_npu/`，包含 vendored 依赖、头文件、BUILD、测试和 Python 文件；不包含仓库其他目录、未跟踪文件、暂存区或未提交修改。审查时 `xla_npu/` 工作区与 HEAD 一致。
- 主排名按“后续累计改动 = 根提交之后各次提交的新增行 + 删除行”降序，同值按路径排序。一行被多次修改会重复计数；替换一行通常计新增 1 + 删除 1。注释、空行均计入。
- 首次导入单列；“全历史累计”= 首次导入行数 + 后续累计改动。完整文件表还提供按全历史累计计算的名次，便于查看包含首次导入的结果。首次导入不代表本仓库作者编写了这些代码。
- “净增”= 后续新增 − 后续删除；“HEAD 行数”统计该提交中文件内容的逻辑行数；“次数”仅计根提交之后有文本增删的提交，不计首次导入。
- 使用 `--no-renames`，按历史路径归集，重命名会按旧路径删除、新路径新增处理，不追溯文件身份。已删除文件保留其历史数据。当前统计未出现二进制 numstat 项。
- “去翻译参考”仅从计数中扣除 `dbe12ec`，不声称剩余改动全部是功能改动，也不声称已逐行证明该提交全为注释。相邻 `f8da2fc` 没有涉及 `xla_npu/`。
- 日期只用于描述；遍历依据 Git 拓扑，不按作者日期筛选。历史中作者日期并非严格递增。

交叉校验：根提交到 HEAD 的单次端点 diff 为新增 28,031 / 删除 14,333，净增同为 13,698；它不等于逐提交累计的 48,412。全部 1,365 个路径均满足“首次行数 + 累计新增 − 累计删除 = HEAD 行数”（删除文件按 0）。

## 3. 一级目录排名

各行递归包含子目录，一级目录之间不重叠，可直接求和。

| 排名 | 目录 | 后续改动文件数 | 首次导入 | 后续新增 | 后续删除 | 后续累计 | 净增 | 全历史累计 | 去翻译参考 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `xla_npu/dependency` | 640 | 378130 | 13871 | 13188 | 27059 | 683 | 405189 | 695 |
| 2 | `xla_npu/backend` | 20 | 4674 | 9677 | 3139 | 12816 | 6538 | 17490 | 11976 |
| 3 | `xla_npu/service` | 64 | 15296 | 7203 | 780 | 7983 | 6423 | 23279 | 7467 |
| 4 | `xla_npu/pjrt` | 21 | 9836 | 141 | 141 | 282 | 0 | 10118 | 0 |
| 5 | `xla_npu/common` | 7 | 893 | 109 | 63 | 172 | 46 | 1065 | 56 |
| 6 | `xla_npu/stream_executor` | 5 | 1520 | 34 | 26 | 60 | 8 | 1580 | 12 |
| 7 | `xla_npu/client` | 2 | 285 | 20 | 20 | 40 | 0 | 325 | 0 |

## 4. 文件改动 Top 20

| 排名 | 文件 | 后续新增 | 后续删除 | 后续累计 | 次数 | HEAD 行数 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | [xla_npu/backend/runtime/npu_aclnn_thunk.cc](../xla_npu/backend/runtime/npu_aclnn_thunk.cc) | 6975 | 2384 | 9359 | 70 | 6297 |
| 2 | [xla_npu/service/npu_ir_emitter_unnested.cc](../xla_npu/service/npu_ir_emitter_unnested.cc) | 941 | 326 | 1267 | 24 | 1118 |
| 3 | [xla_npu/backend/runtime/npu_thunk.cc](../xla_npu/backend/runtime/npu_thunk.cc) | 752 | 132 | 884 | 29 | 1099 |
| 4 | `xla_npu/backend/runtime/npu_deferred_resource_manager.cc`（已删除） | 268 | 268 | 536 | 3 | 0 |
| 5 | [xla_npu/service/convert_to_afir/convert_to_afir.cc](../xla_npu/service/convert_to_afir/convert_to_afir.cc) | 466 | 57 | 523 | 10 | 2878 |
| 6 | [xla_npu/backend/runtime/npu_while_thunk.cc](../xla_npu/backend/runtime/npu_while_thunk.cc) | 403 | 115 | 518 | 6 | 288 |
| 7 | [xla_npu/dependency/interface/acl_nn_op_interface.h](../xla_npu/dependency/interface/acl_nn_op_interface.h) | 377 | 2 | 379 | 26 | 701 |
| 8 | [xla_npu/service/transform/npu_unsupported_fusion_defuser.cc](../xla_npu/service/transform/npu_unsupported_fusion_defuser.cc) | 337 | 34 | 371 | 12 | 303 |
| 9 | [xla_npu/service/transform/BUILD](../xla_npu/service/transform/BUILD) | 360 | 10 | 370 | 21 | 654 |
| 10 | [xla_npu/service/transform/npu_shift_left_decomposition.cc](../xla_npu/service/transform/npu_shift_left_decomposition.cc) | 298 | 43 | 341 | 3 | 255 |
| 11 | [xla_npu/service/transform/npu_dot_contracting_dims_normalizer.cc](../xla_npu/service/transform/npu_dot_contracting_dims_normalizer.cc) | 316 | 7 | 323 | 2 | 309 |
| 12 | [xla_npu/service/transform/npu_dot_contracting_dims_normalizer_test.cc](../xla_npu/service/transform/npu_dot_contracting_dims_normalizer_test.cc) | 303 | 0 | 303 | 2 | 303 |
| 13 | [xla_npu/dependency/external/afir/Ascend-MLIR/python/runtime/executor.py](../xla_npu/dependency/external/afir/Ascend-MLIR/python/runtime/executor.py) | 140 | 140 | 280 | 1 | 711 |
| 14 | [xla_npu/dependency/external/afir/Ascend-MLIR/python/runtime/ascgen.py](../xla_npu/dependency/external/afir/Ascend-MLIR/python/runtime/ascgen.py) | 138 | 122 | 260 | 2 | 741 |
| 15 | [xla_npu/service/transform/npu_tril_utils.h](../xla_npu/service/transform/npu_tril_utils.h) | 223 | 28 | 251 | 4 | 195 |
| 16 | [xla_npu/service/transform/npu_pipeline.cc](../xla_npu/service/transform/npu_pipeline.cc) | 203 | 42 | 245 | 19 | 222 |
| 17 | [xla_npu/service/transform/npu_fusion_block_level_rewriter.cc](../xla_npu/service/transform/npu_fusion_block_level_rewriter.cc) | 228 | 9 | 237 | 2 | 492 |
| 18 | [xla_npu/backend/runtime/npu_stream_ordered_device_memory_pool.cc](../xla_npu/backend/runtime/npu_stream_ordered_device_memory_pool.cc) | 235 | 0 | 235 | 1 | 235 |
| 19 | [xla_npu/service/transform/npu_unsigned_to_float_decomposition.cc](../xla_npu/service/transform/npu_unsigned_to_float_decomposition.cc) | 223 | 12 | 235 | 3 | 211 |
| 20 | [xla_npu/dependency/header/acl_nn_op/aclnn_scatter.h](../xla_npu/dependency/header/acl_nn_op/aclnn_scatter.h) | 113 | 113 | 226 | 1 | 210 |

前 20 个路径占后续累计改动的 35.41%。第 4 名 `npu_deferred_resource_manager.cc` 已在 `a074a5c` 删除，不能作为仍存在的生产文件提出整改；其历史添加和删除仍计入改动量。

## 5. 重点审查发现

审查依据：仓库 `.clang-format` 的 Google 基础风格、现有 `absl::Status` 错误传播方式、明确的资源生命周期，以及重写前后语义一致性。未发现 AGENTS.md 或独立的强制编码规范文件。本节的职责拆分和日志级别属于维护建议，不伪称违反仓库明文禁令。位置均针对上述 HEAD；链接指向相对路径，具体行号写在正文中。

### F1 · 高优先级：tril 匹配接受了不等价的方向和偏移

位置：[npu_tril_utils.h](../xla_npu/service/transform/npu_tril_utils.h) 第 160–179 行；调用处 [npu_tril_rewriter.cc](../xla_npu/service/transform/npu_tril_rewriter.cc) 第 42–59 行。后续新增问题，关键分支来自 `d5567e8`。

匹配器只接受 `GE`，但同时接受 `(lhs_dim=0, rhs_dim=1)` 和 `(lhs_dim=1, rhs_dim=0)`，后者仍返回 `tril`。`col >= row` 保留的是上三角，而 `aclnnTril` 保留下三角；仓库自带的 `aclnn_tril.h:29` 也明确写明 lower triangular。

可直接验证的反例：输入 `[[1,2],[3,4]]`，原始 `select(col >= row, input, 0)` 得 `[[1,2],[0,4]]`；匹配后的 `tril(k=0)` 得 `[[1,0],[3,4]]`。本次用纯 Python 枚举条件确认二者不同；这是语义反例，不是 NPU 端到端运行记录。

此外，正向 `row + lhs_k >= col + rhs_k` 应使用 `lhs_k - rhs_k`，当前仅取 `lhs_k`，也未验证 `rhs_k == 0`。建议收紧方向匹配并正确处理双侧偏移；不支持的模式返回不匹配。补充反向 GE、双侧偏移、嵌套 broadcast 的正反例测试。

### F2 · 高优先级：while carry 解析失败被日志和 continue 吞掉

位置：[npu_ir_emitter_unnested.cc](../xla_npu/service/npu_ir_emitter_unnested.cc) 第 639–669 行、692–701 行及 722–729 行。后续新增路径，`2140b59` 引入 continue，`9ddd728` 改进 buffer 查询但保留该行为。

源 buffer 查询失败时只打印 WARN 后跳过；目标 buffer 主查询和 fallback 都失败时，日志已明确提示 carry propagation may be broken，却仍 continue。标量 carry 查询失败也直接跳过复制。函数随后构造 thunk 并返回成功，没有证明被跳过的 buffer 已安全 alias。

影响：在查询失败的编译路径上，缺失必要 carry copy 的图仍可能被接受，错误从编译阶段延迟为运行时状态错误。这里确认的是错误被吞掉，未声称普通输入必然触发。建议传播 Status 并补充 instruction/ShapeIndex；仅在可证明不需要复制时跳过。增加失败路径测试。

### F3 · 高优先级，但属于初始遗留：初始化静默改写 NaN/Inf

位置：[npu_thunk.cc](../xla_npu/backend/runtime/npu_thunk.cc) 第 911–997 行，特别是 962–986 行；实际入口 [npu_aclnn_thunk.cc](../xla_npu/backend/runtime/npu_aclnn_thunk.cc) 第 829–835 行。

`InitializeSpecialValues` 对部分 4 字节浮点 buffer 将 `-inf/+inf` 改为有限极值，将 NaN 改为 0，再写回设备。除跳过入口参数外，没有用明确的常量来源证明这一修改合法，也没有按具体算子的语义限制；读写错误还可能被忽略并最终返回成功。改写会改变 is-finite、NaN 检测及依赖特殊值的计算含义。

`git blame` 确认核心替换逻辑已存在于根提交 `d32d52c`，不能归咎于后续 884 行改动；`6cbbe6d` 增加浮点类型过滤，缓解了整数误写，未消除浮点语义改变。建议取消通用静默清洗，针对不支持的算子做有边界的适配或明确报错，并验证 Inf/NaN 常量、参数和中间结果。未进行 NPU 运行验证。

### F4 · 中优先级风险：异步拷贝同步失败后仍释放 host buffer

位置：[npu_aclnn_thunk.cc](../xla_npu/backend/runtime/npu_aclnn_thunk.cc) 第 2269–2282 行；相关失败分支来自 `3b048f4`。

Iota 的 H2D 异步提交成功后，若 `BlockHostUntilDone()` 失败，代码立即 `aclrtFreeHost(host_buffer)`，再返回错误。此时尚未获得“设备已停止读取 staging buffer”的成功确认；同文件第 946–955 行的 metadata 清理遇到 drain 失败则明确保留资源以避免异步 use-after-free，两处策略不一致。

建议将 host buffer 纳入统一延迟回收，或在能够确认设备停止访问后释放。此项是失败路径生命周期风险，触发后的真实行为取决于 ACL 错误与设备状态；未通过错误注入验证，不报告为已复现的 UAF。

### F5 · 中优先级维护问题：核心执行函数过大

位置：[npu_aclnn_thunk.cc](../xla_npu/backend/runtime/npu_aclnn_thunk.cc) 第 838–6194 行附近的 `ExecuteOnStream`。

单文件 6,297 行，单个执行函数约 5,350 行，混合算子派发、形状检查、host fallback、临时设备内存、ACL descriptor 生命周期和日志。文件经历 70 次后续改动，局部修复很难覆盖全部提前返回和清理路径。建议按 gather/scatter、矩阵运算、转换与基础算子拆分 helper，并统一资源持有器及错误路径测试。这是有数据支持的维护建议，不是仓库明文规定的函数长度违规。

### F6 · 低优先级：调试遗留变量、重复分支和常态 WARN

位置：[npu_aclnn_thunk.cc](../xla_npu/backend/runtime/npu_aclnn_thunk.cc) 第 141–220 行中，`source` 和 `fallback_reason` 多次赋值，但函数最终只返回 tensor，未消费这些字符串；第 6219–6228 行 if/else 两边完全相同。前者变量在 `6fdaf9b` 引入，当前快照仍残留，后者也没有必要保留分支。

另见 [npu_while_thunk.cc](../xla_npu/backend/runtime/npu_while_thunk.cc) 第 53–70 行，正常的“需要 carry copy / 不需要 copy”两个分支都用 WARN；[npu_ir_emitter_unnested.cc](../xla_npu/service/npu_ir_emitter_unnested.cc) 第 556、595 行也对常态编译过程打印 WARN。建议清理不再使用的诊断构造和重复分支，常态详情降为 DEBUG/VLOG，保留真正异常的 WARN/ERROR。

### F7 · 低优先级、机械检查已确认：尾随空白

执行 `git diff --check d32d52c HEAD -- xla_npu` 检出 **27 处、12 个文件**的尾随空白，包括：

- `npu_aclnn_thunk.cc:1082,4723`。
- `npu_thunk.cc:633`。
- `convert_to_afir.cc:133,2180`。
- 其余 22 处位于依赖头文件。

这些是端点 diff 中新增/修改行的检查结果，不代表所有历史提交的违规总数。建议只清理相关行，并把 `git diff --check` 放入提交前检查。环境没有可用的 `clang-format` 命令，本次未伪称通过完整格式检查；`.clang-format` 明确禁用注释重排和 include 排序，因此没有把长注释或 include 顺序单独认定为违规。

## 6. 审查覆盖与验证限制

本次对 Top 20 路径做重点抽查：阅读关键函数、相关调用点、初始版本差异及必要的 blame；不是逐行证明所有算子语义正确。审查按当前代码提出仍适用的问题，历史删除文件单独标注。

| 主排名 | 文件 | 检查内容/结论 |
| --- | --- | --- |
| 1 | [xla_npu/backend/runtime/npu_aclnn_thunk.cc](../xla_npu/backend/runtime/npu_aclnn_thunk.cc) | 派发、资源清理、Iota 和转换路径；F4/F5/F6/F7。 |
| 2 | [xla_npu/service/npu_ir_emitter_unnested.cc](../xla_npu/service/npu_ir_emitter_unnested.cc) | while carry 生成、tril 调用；F2/F6。 |
| 3 | [xla_npu/backend/runtime/npu_thunk.cc](../xla_npu/backend/runtime/npu_thunk.cc) | 算子元数据、特殊值初始化；F3/F7。 |
| 4 | `xla_npu/backend/runtime/npu_deferred_resource_manager.cc`（已删除） | 查证添加/删除历史；HEAD 已删除，未作为当前缺陷。 |
| 5 | [xla_npu/service/convert_to_afir/convert_to_afir.cc](../xla_npu/service/convert_to_afir/convert_to_afir.cc) | 转换状态返回、reshape/broadcast 辅助代码、module 验证及 context 生命周期入口；F7；其他路径未作正确性证明。 |
| 6 | [xla_npu/backend/runtime/npu_while_thunk.cc](../xla_npu/backend/runtime/npu_while_thunk.cc) | 循环体/条件/复制的调度与错误传播；F6。 |
| 7 | [xla_npu/dependency/interface/acl_nn_op_interface.h](../xla_npu/dependency/interface/acl_nn_op_interface.h) | ACLNN 声明宏和新增接口组织；未确认新的独立问题。 |
| 8 | [xla_npu/service/transform/npu_unsupported_fusion_defuser.cc](../xla_npu/service/transform/npu_unsupported_fusion_defuser.cc) | wrapper、hero 和 unsupported 判定；未确认新的独立问题。 |
| 9 | [xla_npu/service/transform/BUILD](../xla_npu/service/transform/BUILD) | 抽查新增 pass、测试 target 和依赖声明；未运行 Bazel 构建。 |
| 10 | [xla_npu/service/transform/npu_shift_left_decomposition.cc](../xla_npu/service/transform/npu_shift_left_decomposition.cc) | 常量/动态两条 lowering 路径；动态移位极值语义需进一步设备测试，未作为确定缺陷。 |
| 11 | [xla_npu/service/transform/npu_dot_contracting_dims_normalizer.cc](../xla_npu/service/transform/npu_dot_contracting_dims_normalizer.cc) | 维度分类、布局与常量物化；未确认新的独立问题。 |
| 12 | [xla_npu/service/transform/npu_dot_contracting_dims_normalizer_test.cc](../xla_npu/service/transform/npu_dot_contracting_dims_normalizer_test.cc) | 检查规范布局、不改写和 tuple/layout 测试存在；本次未执行这些测试。 |
| 13 | [xla_npu/dependency/external/afir/Ascend-MLIR/python/runtime/executor.py](../xla_npu/dependency/external/afir/Ascend-MLIR/python/runtime/executor.py) | 核对单次翻译历史并抽查运行库加载；未将原有注释块当作新引入问题。 |
| 14 | [xla_npu/dependency/external/afir/Ascend-MLIR/python/runtime/ascgen.py](../xla_npu/dependency/external/afir/Ascend-MLIR/python/runtime/ascgen.py) | 抽查失败图保存、异常传播和编译调用；未确认新的独立问题。 |
| 15 | [xla_npu/service/transform/npu_tril_utils.h](../xla_npu/service/transform/npu_tril_utils.h) | 方向、偏移与调用链；F1。 |
| 16 | [xla_npu/service/transform/npu_pipeline.cc](../xla_npu/service/transform/npu_pipeline.cc) | 检查 pass 接入和执行顺序；未确认新的独立问题。 |
| 17 | [xla_npu/service/transform/npu_fusion_block_level_rewriter.cc](../xla_npu/service/transform/npu_fusion_block_level_rewriter.cc) | 抽查 fusion 选择和改写入口；未确认新的独立问题。 |
| 18 | [xla_npu/backend/runtime/npu_stream_ordered_device_memory_pool.cc](../xla_npu/backend/runtime/npu_stream_ordered_device_memory_pool.cc) | 单 stream 约束、drain、reset 和退休资源；压力条件未做运行验证。 |
| 19 | [xla_npu/service/transform/npu_unsigned_to_float_decomposition.cc](../xla_npu/service/transform/npu_unsigned_to_float_decomposition.cc) | 抽查 U32→F32 分解及执行入口；未进行数值回归。 |
| 20 | [xla_npu/dependency/header/acl_nn_op/aclnn_scatter.h](../xla_npu/dependency/header/acl_nn_op/aclnn_scatter.h) | 核对历史：根提交导入，后续仅 dbe12ec；未把其文本 churn 解释成功能改写。 |

已完成的验证：逐提交统计、全文件行数守恒校验、端点 diff 交叉校验、目录汇总校验、当前工作区一致性检查、27 处空白诊断，以及 F1 的 2×2 条件反例。

未执行：全量编译、NPU 回归、ACL 错误注入、clang-format。任务是统计和审查报告，本次未修改实现；上述静态结论不应当作硬件复现结果。“未确认问题”也不等于对应文件已通过完整审计。未跟踪的 `tests/regression/` 不属于提交快照，未拿它作为已提交测试覆盖的依据。

建议顺序：先补 F1 的等价性负例、F2 的错误传播，再为 F3 做历史兼容性评估和特殊值测试；F4 用错误注入确认并统一生命周期；随后拆分大函数、清理无效代码及格式。不要把首次导入体积或大规模翻译行数作为整改优先级的唯一依据。

## 7. 全部目录降序排名（130 个）

目录均递归汇总其后代文件，因此父子目录重复覆盖；**本表不能跨行求总和**。要核对总量，请使用第 3 节互不重叠的一级目录。包含只有初始导入、后续改动为 0 的目录。

| 排名 | 目录 | 历史文件数 | 首次导入 | 后续新增 | 后续删除 | 后续累计 | 净增 | 全历史累计 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `xla_npu/dependency` | 1123 | 378130 | 13871 | 13188 | 27059 | 683 | 405189 |
| 2 | `xla_npu/dependency/header` | 879 | 76547 | 12662 | 12662 | 25324 | 0 | 101871 |
| 3 | `xla_npu/dependency/header/acl_nn_op` | 741 | 46102 | 11870 | 11870 | 23740 | 0 | 69842 |
| 4 | `xla_npu/backend` | 29 | 4674 | 9677 | 3139 | 12816 | 6538 | 17490 |
| 5 | `xla_npu/backend/runtime` | 24 | 4192 | 9559 | 3100 | 12659 | 6459 | 16851 |
| 6 | `xla_npu/service` | 111 | 15296 | 7203 | 780 | 7983 | 6423 | 23279 |
| 7 | `xla_npu/service/transform` | 59 | 3203 | 5401 | 333 | 5734 | 5068 | 8937 |
| 8 | `xla_npu/dependency/header/metadef` | 86 | 13883 | 761 | 761 | 1522 | 0 | 15405 |
| 9 | `xla_npu/dependency/header/metadef/exe_graph` | 24 | 4199 | 697 | 697 | 1394 | 0 | 5593 |
| 10 | `xla_npu/dependency/header/metadef/exe_graph/runtime` | 23 | 4121 | 677 | 677 | 1354 | 0 | 5475 |
| 11 | `xla_npu/dependency/external` | 223 | 299147 | 533 | 491 | 1024 | 42 | 300171 |
| 12 | `xla_npu/dependency/external/afir` | 222 | 298858 | 532 | 490 | 1022 | 42 | 299880 |
| 13 | `xla_npu/dependency/external/afir/Ascend-MLIR` | 193 | 18315 | 531 | 490 | 1021 | 41 | 19336 |
| 14 | `xla_npu/dependency/external/afir/Ascend-MLIR/python` | 5 | 2376 | 466 | 450 | 916 | 16 | 3292 |
| 15 | `xla_npu/dependency/external/afir/Ascend-MLIR/python/runtime` | 5 | 2376 | 466 | 450 | 916 | 16 | 3292 |
| 16 | `xla_npu/service/convert_to_afir` | 10 | 4865 | 596 | 72 | 668 | 524 | 5533 |
| 17 | `xla_npu/dependency/interface` | 9 | 750 | 644 | 3 | 647 | 641 | 1397 |
| 18 | `xla_npu/pjrt` | 74 | 9836 | 141 | 141 | 282 | 0 | 10118 |
| 19 | `xla_npu/pjrt/air` | 58 | 8971 | 137 | 137 | 274 | 0 | 9245 |
| 20 | `xla_npu/pjrt/air/mair` | 34 | 5027 | 133 | 133 | 266 | 0 | 5293 |
| 21 | `xla_npu/common` | 9 | 893 | 109 | 63 | 172 | 46 | 1065 |
| 22 | `xla_npu/backend/codegen` | 5 | 482 | 118 | 39 | 157 | 79 | 639 |
| 23 | `xla_npu/pjrt/air/mair/test` | 4 | 533 | 42 | 42 | 84 | 0 | 617 |
| 24 | `xla_npu/dependency/external/afir/Ascend-MLIR/lib` | 14 | 3333 | 53 | 28 | 81 | 25 | 3414 |
| 25 | `xla_npu/dependency/header/metadef/graph` | 16 | 3443 | 35 | 35 | 70 | 0 | 3513 |
| 26 | `xla_npu/pjrt/air/mair/transforms` | 5 | 1162 | 33 | 33 | 66 | 0 | 1228 |
| 27 | `xla_npu/dependency/tool` | 7 | 698 | 30 | 30 | 60 | 0 | 758 |
| 28 | `xla_npu/stream_executor` | 14 | 1520 | 34 | 26 | 60 | 8 | 1580 |
| 29 | `xla_npu/dependency/header/metadef/register` | 27 | 3767 | 29 | 29 | 58 | 0 | 3825 |
| 30 | `xla_npu/pjrt/air/mair/translations` | 11 | 1337 | 26 | 26 | 52 | 0 | 1389 |
| 31 | `xla_npu/dependency/external/afir/Ascend-MLIR/lib/Dialect` | 12 | 2019 | 23 | 24 | 47 | -1 | 2066 |
| 32 | `xla_npu/dependency/external/afir/Ascend-MLIR/lib/Dialect/AFIR` | 12 | 2019 | 23 | 24 | 47 | -1 | 2066 |
| 33 | `xla_npu/dependency/external/afir/Ascend-MLIR/lib/Dialect/AFIR/Ops` | 8 | 1377 | 23 | 24 | 47 | -1 | 1424 |
| 34 | `xla_npu/dependency/external/afir/Ascend-MLIR/lib/Dialect/AFIR/Ops/Basic` | 5 | 1003 | 23 | 24 | 47 | -1 | 1050 |
| 35 | `xla_npu/client` | 5 | 285 | 20 | 20 | 40 | 0 | 325 |
| 36 | `xla_npu/dependency/header/metadef/exe_graph/lowering` | 1 | 78 | 20 | 20 | 40 | 0 | 118 |
| 37 | `xla_npu/service/kernel` | 5 | 314 | 19 | 19 | 38 | 0 | 352 |
| 38 | `xla_npu/dependency/header/acl_nn` | 34 | 7103 | 18 | 18 | 36 | 0 | 7139 |
| 39 | `xla_npu/dependency/external/afir/Ascend-MLIR/lib/Conversion` | 1 | 1251 | 30 | 4 | 34 | 26 | 1285 |
| 40 | `xla_npu/dependency/external/afir/Ascend-MLIR/lib/Conversion/AFIRToASCIRText` | 1 | 1251 | 30 | 4 | 34 | 26 | 1285 |
| 41 | `xla_npu/dependency/header/acl_nn/opdev` | 31 | 6845 | 16 | 16 | 32 | 0 | 6877 |
| 42 | `xla_npu/dependency/external/afir/Ascend-MLIR/include` | 34 | 2451 | 12 | 12 | 24 | 0 | 2475 |
| 43 | `xla_npu/dependency/external/afir/Ascend-MLIR/include/Dialect` | 22 | 1677 | 12 | 12 | 24 | 0 | 1701 |
| 44 | `xla_npu/dependency/external/afir/Ascend-MLIR/include/Dialect/AFIR` | 22 | 1677 | 12 | 12 | 24 | 0 | 1701 |
| 45 | `xla_npu/dependency/header/ge` | 10 | 2296 | 11 | 11 | 22 | 0 | 2318 |
| 46 | `xla_npu/dependency/header/acl_nn/opdev/aicpu` | 5 | 1169 | 8 | 8 | 16 | 0 | 1185 |
| 47 | `xla_npu/dependency/header/metadef/graph/utils` | 2 | 138 | 5 | 5 | 10 | 0 | 148 |
| 48 | `xla_npu/pjrt/air/npu` | 19 | 3485 | 4 | 4 | 8 | 0 | 3493 |
| 49 | `xla_npu/dependency/external/afir/Ascend-MLIR/include/Dialect/AFIR/Core` | 3 | 390 | 3 | 3 | 6 | 0 | 396 |
| 50 | `xla_npu/dependency/header/acl` | 8 | 7163 | 2 | 2 | 4 | 0 | 7167 |
| 51 | `xla_npu/dependency/stub` | 5 | 988 | 2 | 2 | 4 | 0 | 992 |
| 52 | `xla_npu/pjrt/air/mair/ir` | 6 | 948 | 2 | 2 | 4 | 0 | 952 |
| 53 | `xla_npu/pjrt/c` | 6 | 278 | 2 | 2 | 4 | 0 | 282 |
| 54 | `xla_npu/pjrt/npu` | 3 | 128 | 2 | 2 | 4 | 0 | 132 |
| 55 | `xla_npu/dependency/external/npuir` | 1 | 289 | 1 | 1 | 2 | 0 | 291 |
| 56 | `xla_npu/pjrt/air/npu/service` | 5 | 492 | 1 | 1 | 2 | 0 | 494 |
| 57 | `xla_npu/dependency/external/afir/Ascend-MLIR/externals` | 140 | 10155 | 0 | 0 | 0 | 0 | 10155 |
| 58 | `xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc` | 140 | 10155 | 0 | 0 | 0 | 0 | 10155 |
| 59 | `xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include` | 134 | 9594 | 0 | 0 | 0 | 0 | 9594 |
| 60 | `xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir` | 133 | 9585 | 0 | 0 | 0 | 0 | 9585 |
| 61 | `xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/API` | 2 | 409 | 0 | 0 | 0 | 0 | 409 |
| 62 | `xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect` | 85 | 6479 | 0 | 0 | 0 | 0 | 6479 |
| 63 | `xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc` | 72 | 5834 | 0 | 0 | 0 | 0 | 5834 |
| 64 | `xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR` | 66 | 5554 | 0 | 0 | 0 | 0 | 5554 |
| 65 | `xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Adv` | 7 | 652 | 0 | 0 | 0 | 0 | 652 |
| 66 | `xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic` | 43 | 2443 | 0 | 0 | 0 | 0 | 2443 |
| 67 | `xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Core` | 5 | 1108 | 0 | 0 | 0 | 0 | 1108 |
| 68 | `xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Fwk` | 4 | 447 | 0 | 0 | 0 | 0 | 447 |
| 69 | `xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/Transforms` | 3 | 174 | 0 | 0 | 0 | 0 | 174 |
| 70 | `xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/Utils` | 2 | 96 | 0 | 0 | 0 | 0 | 96 |
| 71 | `xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/EmitAsc` | 8 | 410 | 0 | 0 | 0 | 0 | 410 |
| 72 | `xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/EmitAsc/IR` | 6 | 370 | 0 | 0 | 0 | 0 | 370 |
| 73 | `xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/EmitAsc/Utils` | 1 | 31 | 0 | 0 | 0 | 0 | 31 |
| 74 | `xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Utils` | 4 | 225 | 0 | 0 | 0 | 0 | 225 |
| 75 | `xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target` | 45 | 2687 | 0 | 0 | 0 | 0 | 2687 |
| 76 | `xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc` | 45 | 2687 | 0 | 0 | 0 | 0 | 2687 |
| 77 | `xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Adv` | 6 | 277 | 0 | 0 | 0 | 0 | 277 |
| 78 | `xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Basic` | 21 | 1177 | 0 | 0 | 0 | 0 | 1177 |
| 79 | `xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Core` | 3 | 116 | 0 | 0 | 0 | 0 | 116 |
| 80 | `xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/External` | 6 | 370 | 0 | 0 | 0 | 0 | 370 |
| 81 | `xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Fwk` | 2 | 87 | 0 | 0 | 0 | 0 | 87 |
| 82 | `xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/lib` | 6 | 561 | 0 | 0 | 0 | 0 | 561 |
| 83 | `xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/lib/Dialect` | 6 | 561 | 0 | 0 | 0 | 0 | 561 |
| 84 | `xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/lib/Dialect/Asc` | 6 | 561 | 0 | 0 | 0 | 0 | 561 |
| 85 | `xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/lib/Dialect/Asc/IR` | 6 | 561 | 0 | 0 | 0 | 0 | 561 |
| 86 | `xla_npu/dependency/external/afir/Ascend-MLIR/include/CAPI` | 2 | 135 | 0 | 0 | 0 | 0 | 135 |
| 87 | `xla_npu/dependency/external/afir/Ascend-MLIR/include/Conversion` | 8 | 396 | 0 | 0 | 0 | 0 | 396 |
| 88 | `xla_npu/dependency/external/afir/Ascend-MLIR/include/Conversion/AFIRToASCIR` | 4 | 222 | 0 | 0 | 0 | 0 | 222 |
| 89 | `xla_npu/dependency/external/afir/Ascend-MLIR/include/Conversion/AFIRToASCIR/Math` | 1 | 45 | 0 | 0 | 0 | 0 | 45 |
| 90 | `xla_npu/dependency/external/afir/Ascend-MLIR/include/Conversion/AFIRToASCIRText` | 1 | 37 | 0 | 0 | 0 | 0 | 37 |
| 91 | `xla_npu/dependency/external/afir/Ascend-MLIR/include/Dialect/AFIR/Ops` | 7 | 442 | 0 | 0 | 0 | 0 | 442 |
| 92 | `xla_npu/dependency/external/afir/Ascend-MLIR/include/Dialect/AFIR/Ops/Adv` | 2 | 124 | 0 | 0 | 0 | 0 | 124 |
| 93 | `xla_npu/dependency/external/afir/Ascend-MLIR/include/Dialect/AFIR/Ops/Basic` | 5 | 318 | 0 | 0 | 0 | 0 | 318 |
| 94 | `xla_npu/dependency/external/afir/Ascend-MLIR/include/Dialect/AFIR/TransformOps` | 3 | 119 | 0 | 0 | 0 | 0 | 119 |
| 95 | `xla_npu/dependency/external/afir/Ascend-MLIR/include/Dialect/AFIR/Transforms` | 3 | 115 | 0 | 0 | 0 | 0 | 115 |
| 96 | `xla_npu/dependency/external/afir/Ascend-MLIR/include/Utils` | 2 | 243 | 0 | 0 | 0 | 0 | 243 |
| 97 | `xla_npu/dependency/external/afir/Ascend-MLIR/lib/Dialect/AFIR/Ops/Adv` | 3 | 374 | 0 | 0 | 0 | 0 | 374 |
| 98 | `xla_npu/dependency/external/afir/Ascend-MLIR/lib/Dialect/AFIR/Transforms` | 3 | 540 | 0 | 0 | 0 | 0 | 540 |
| 99 | `xla_npu/dependency/external/afir/Ascend-MLIR/lib/Utils` | 1 | 63 | 0 | 0 | 0 | 0 | 63 |
| 100 | `xla_npu/dependency/external/afir/generated` | 28 | 280366 | 0 | 0 | 0 | 0 | 280366 |
| 101 | `xla_npu/dependency/external/afir/generated/externals` | 16 | 227548 | 0 | 0 | 0 | 0 | 227548 |
| 102 | `xla_npu/dependency/external/afir/generated/externals/pyasc` | 16 | 227548 | 0 | 0 | 0 | 0 | 227548 |
| 103 | `xla_npu/dependency/external/afir/generated/externals/pyasc/include` | 16 | 227548 | 0 | 0 | 0 | 0 | 227548 |
| 104 | `xla_npu/dependency/external/afir/generated/externals/pyasc/include/ascir` | 16 | 227548 | 0 | 0 | 0 | 0 | 227548 |
| 105 | `xla_npu/dependency/external/afir/generated/externals/pyasc/include/ascir/API` | 2 | 454 | 0 | 0 | 0 | 0 | 454 |
| 106 | `xla_npu/dependency/external/afir/generated/externals/pyasc/include/ascir/Dialect` | 14 | 227094 | 0 | 0 | 0 | 0 | 227094 |
| 107 | `xla_npu/dependency/external/afir/generated/externals/pyasc/include/ascir/Dialect/Asc` | 14 | 227094 | 0 | 0 | 0 | 0 | 227094 |
| 108 | `xla_npu/dependency/external/afir/generated/externals/pyasc/include/ascir/Dialect/Asc/IR` | 14 | 227094 | 0 | 0 | 0 | 0 | 227094 |
| 109 | `xla_npu/dependency/external/afir/generated/include` | 12 | 52818 | 0 | 0 | 0 | 0 | 52818 |
| 110 | `xla_npu/dependency/external/afir/generated/include/Conversion` | 1 | 224 | 0 | 0 | 0 | 0 | 224 |
| 111 | `xla_npu/dependency/external/afir/generated/include/Dialect` | 11 | 52594 | 0 | 0 | 0 | 0 | 52594 |
| 112 | `xla_npu/dependency/external/afir/generated/include/Dialect/AFIR` | 11 | 52594 | 0 | 0 | 0 | 0 | 52594 |
| 113 | `xla_npu/dependency/external/afir/generated/include/Dialect/AFIR/TransformOps` | 2 | 729 | 0 | 0 | 0 | 0 | 729 |
| 114 | `xla_npu/dependency/external/afir/generated/include/Dialect/AFIR/Transforms` | 1 | 296 | 0 | 0 | 0 | 0 | 296 |
| 115 | `xla_npu/dependency/header/acl/error_codes` | 2 | 261 | 0 | 0 | 0 | 0 | 261 |
| 116 | `xla_npu/dependency/header/acl_nn/opdev/op_common` | 2 | 250 | 0 | 0 | 0 | 0 | 250 |
| 117 | `xla_npu/dependency/header/acl_nn/opdev/register` | 1 | 59 | 0 | 0 | 0 | 0 | 59 |
| 118 | `xla_npu/dependency/header/metadef/flow_graph` | 4 | 322 | 0 | 0 | 0 | 0 | 322 |
| 119 | `xla_npu/dependency/header/metadef/ge` | 2 | 128 | 0 | 0 | 0 | 0 | 128 |
| 120 | `xla_npu/dependency/header/metadef/ge/framework` | 1 | 47 | 0 | 0 | 0 | 0 | 47 |
| 121 | `xla_npu/dependency/header/metadef/ge/framework/common` | 1 | 47 | 0 | 0 | 0 | 0 | 47 |
| 122 | `xla_npu/dependency/header/metadef/ge_common` | 3 | 980 | 0 | 0 | 0 | 0 | 980 |
| 123 | `xla_npu/dependency/header/metadef/hcom` | 1 | 74 | 0 | 0 | 0 | 0 | 74 |
| 124 | `xla_npu/dependency/header/metadef/op_common` | 4 | 327 | 0 | 0 | 0 | 0 | 327 |
| 125 | `xla_npu/dependency/header/metadef/platform` | 3 | 514 | 0 | 0 | 0 | 0 | 514 |
| 126 | `xla_npu/dependency/header/metadef/register/scope` | 1 | 478 | 0 | 0 | 0 | 0 | 478 |
| 127 | `xla_npu/dependency/header/metadef/utils` | 1 | 111 | 0 | 0 | 0 | 0 | 111 |
| 128 | `xla_npu/pjrt/air/npu/ge` | 2 | 78 | 0 | 0 | 0 | 0 | 78 |
| 129 | `xla_npu/pjrt/plugin` | 7 | 459 | 0 | 0 | 0 | 0 | 459 |
| 130 | `xla_npu/service/convert_to_npuir` | 3 | 102 | 0 | 0 | 0 | 0 | 102 |

## 8. 全部文件降序排名（1,365 个历史路径）

前 759 个路径有后续文本修改，剩余 606 个仅有首次导入。主排序按后续累计；“全史名次”按包含首次导入的全历史累计降序、同值按路径排序。全历史累计可由“首次 + 新增 + 删除”得到。已删除文件的 HEAD 行数为 0。

### 8.1 后续有改动的文件

| 后续名次 | 全史名次 | 文件 | 首次 | 后续新增 | 后续删除 | 后续累计 | 净增 | 次数 | HEAD 行数 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 5 | [xla_npu/backend/runtime/npu_aclnn_thunk.cc](../xla_npu/backend/runtime/npu_aclnn_thunk.cc) | 1706 | 6975 | 2384 | 9359 | 4591 | 70 | 6297 |
| 2 | 11 | [xla_npu/service/npu_ir_emitter_unnested.cc](../xla_npu/service/npu_ir_emitter_unnested.cc) | 503 | 941 | 326 | 1267 | 615 | 24 | 1118 |
| 3 | 15 | [xla_npu/backend/runtime/npu_thunk.cc](../xla_npu/backend/runtime/npu_thunk.cc) | 479 | 752 | 132 | 884 | 620 | 29 | 1099 |
| 4 | 43 | `xla_npu/backend/runtime/npu_deferred_resource_manager.cc`（已删除） | 0 | 268 | 268 | 536 | 0 | 3 | 0 |
| 5 | 8 | [xla_npu/service/convert_to_afir/convert_to_afir.cc](../xla_npu/service/convert_to_afir/convert_to_afir.cc) | 2469 | 466 | 57 | 523 | 409 | 10 | 2878 |
| 6 | 46 | [xla_npu/backend/runtime/npu_while_thunk.cc](../xla_npu/backend/runtime/npu_while_thunk.cc) | 0 | 403 | 115 | 518 | 288 | 6 | 288 |
| 7 | 34 | [xla_npu/dependency/interface/acl_nn_op_interface.h](../xla_npu/dependency/interface/acl_nn_op_interface.h) | 326 | 377 | 2 | 379 | 375 | 26 | 701 |
| 8 | 76 | [xla_npu/service/transform/npu_unsupported_fusion_defuser.cc](../xla_npu/service/transform/npu_unsupported_fusion_defuser.cc) | 0 | 337 | 34 | 371 | 303 | 12 | 303 |
| 9 | 36 | [xla_npu/service/transform/BUILD](../xla_npu/service/transform/BUILD) | 304 | 360 | 10 | 370 | 350 | 21 | 654 |
| 10 | 84 | [xla_npu/service/transform/npu_shift_left_decomposition.cc](../xla_npu/service/transform/npu_shift_left_decomposition.cc) | 0 | 298 | 43 | 341 | 255 | 3 | 255 |
| 11 | 93 | [xla_npu/service/transform/npu_dot_contracting_dims_normalizer.cc](../xla_npu/service/transform/npu_dot_contracting_dims_normalizer.cc) | 0 | 316 | 7 | 323 | 309 | 2 | 309 |
| 12 | 100 | [xla_npu/service/transform/npu_dot_contracting_dims_normalizer_test.cc](../xla_npu/service/transform/npu_dot_contracting_dims_normalizer_test.cc) | 0 | 303 | 0 | 303 | 303 | 2 | 303 |
| 13 | 21 | [xla_npu/dependency/external/afir/Ascend-MLIR/python/runtime/executor.py](../xla_npu/dependency/external/afir/Ascend-MLIR/python/runtime/executor.py) | 711 | 140 | 140 | 280 | 0 | 1 | 711 |
| 14 | 22 | [xla_npu/dependency/external/afir/Ascend-MLIR/python/runtime/ascgen.py](../xla_npu/dependency/external/afir/Ascend-MLIR/python/runtime/ascgen.py) | 725 | 138 | 122 | 260 | 16 | 2 | 741 |
| 15 | 139 | [xla_npu/service/transform/npu_tril_utils.h](../xla_npu/service/transform/npu_tril_utils.h) | 0 | 223 | 28 | 251 | 195 | 4 | 195 |
| 16 | 98 | [xla_npu/service/transform/npu_pipeline.cc](../xla_npu/service/transform/npu_pipeline.cc) | 61 | 203 | 42 | 245 | 161 | 19 | 222 |
| 17 | 48 | [xla_npu/service/transform/npu_fusion_block_level_rewriter.cc](../xla_npu/service/transform/npu_fusion_block_level_rewriter.cc) | 273 | 228 | 9 | 237 | 219 | 2 | 492 |
| 18 | 146 | [xla_npu/backend/runtime/npu_stream_ordered_device_memory_pool.cc](../xla_npu/backend/runtime/npu_stream_ordered_device_memory_pool.cc) | 0 | 235 | 0 | 235 | 235 | 1 | 235 |
| 19 | 147 | [xla_npu/service/transform/npu_unsigned_to_float_decomposition.cc](../xla_npu/service/transform/npu_unsigned_to_float_decomposition.cc) | 0 | 223 | 12 | 235 | 211 | 3 | 211 |
| 20 | 62 | [xla_npu/dependency/header/acl_nn_op/aclnn_scatter.h](../xla_npu/dependency/header/acl_nn_op/aclnn_scatter.h) | 210 | 113 | 113 | 226 | 0 | 1 | 210 |
| 21 | 161 | [xla_npu/backend/runtime/npu_conditional_thunk.cc](../xla_npu/backend/runtime/npu_conditional_thunk.cc) | 0 | 205 | 14 | 219 | 191 | 3 | 191 |
| 22 | 65 | [xla_npu/dependency/header/acl_nn_op/aclnn_repeat_interleave.h](../xla_npu/dependency/header/acl_nn_op/aclnn_repeat_interleave.h) | 212 | 103 | 103 | 206 | 0 | 1 | 212 |
| 23 | 186 | [xla_npu/service/transform/npu_scatter_window_expander.cc](../xla_npu/service/transform/npu_scatter_window_expander.cc) | 0 | 205 | 0 | 205 | 205 | 1 | 205 |
| 24 | 187 | [xla_npu/service/transform/npu_variadic_reduce_expander.cc](../xla_npu/service/transform/npu_variadic_reduce_expander.cc) | 0 | 204 | 0 | 204 | 204 | 1 | 204 |
| 25 | 69 | [xla_npu/dependency/header/acl_nn_op/aclnn_remainder.h](../xla_npu/dependency/header/acl_nn_op/aclnn_remainder.h) | 218 | 98 | 98 | 196 | 0 | 1 | 218 |
| 26 | 51 | [xla_npu/service/transform/npu_priority_fusion.cc](../xla_npu/service/transform/npu_priority_fusion.cc) | 298 | 166 | 30 | 196 | 136 | 5 | 434 |
| 27 | 35 | [xla_npu/backend/runtime/npu_thunk.h](../xla_npu/backend/runtime/npu_thunk.h) | 494 | 153 | 30 | 183 | 123 | 23 | 617 |
| 28 | 58 | [xla_npu/dependency/header/acl_nn_op/aclnn_bernoulli.h](../xla_npu/dependency/header/acl_nn_op/aclnn_bernoulli.h) | 271 | 90 | 90 | 180 | 0 | 1 | 271 |
| 29 | 80 | [xla_npu/dependency/header/acl_nn_op/aclnn_convolution.h](../xla_npu/dependency/header/acl_nn_op/aclnn_convolution.h) | 173 | 90 | 90 | 180 | 0 | 1 | 173 |
| 30 | 246 | [xla_npu/service/transform/npu_dot_fp32_left_order_rewriter.cc](../xla_npu/service/transform/npu_dot_fp32_left_order_rewriter.cc) | 0 | 175 | 0 | 175 | 175 | 1 | 175 |
| 31 | 284 | `xla_npu/backend/runtime/npu_deferred_resource_manager.h`（已删除） | 0 | 82 | 82 | 164 | 0 | 3 | 0 |
| 32 | 26 | [xla_npu/service/npu_compiler.cc](../xla_npu/service/npu_compiler.cc) | 676 | 158 | 1 | 159 | 157 | 3 | 833 |
| 33 | 52 | [xla_npu/dependency/header/metadef/exe_graph/runtime/kernel_context.h](../xla_npu/dependency/header/metadef/exe_graph/runtime/kernel_context.h) | 333 | 79 | 79 | 158 | 0 | 1 | 333 |
| 34 | 310 | [xla_npu/service/transform/npu_variadic_reduce_expander_test.cc](../xla_npu/service/transform/npu_variadic_reduce_expander_test.cc) | 0 | 154 | 0 | 154 | 154 | 1 | 154 |
| 35 | 71 | [xla_npu/backend/codegen/npu_custom.cc](../xla_npu/backend/codegen/npu_custom.cc) | 247 | 114 | 36 | 150 | 78 | 5 | 325 |
| 36 | 47 | [xla_npu/dependency/external/afir/Ascend-MLIR/python/runtime/utils.py](../xla_npu/dependency/external/afir/Ascend-MLIR/python/runtime/utils.py) | 367 | 74 | 74 | 148 | 0 | 1 | 367 |
| 37 | 337 | [xla_npu/service/transform/npu_fma_addcmul_rewriter.cc](../xla_npu/service/transform/npu_fma_addcmul_rewriter.cc) | 0 | 140 | 7 | 147 | 133 | 2 | 133 |
| 38 | 359 | [xla_npu/service/transform/npu_scatter_window_expander_test.cc](../xla_npu/service/transform/npu_scatter_window_expander_test.cc) | 0 | 143 | 0 | 143 | 143 | 1 | 143 |
| 39 | 375 | [xla_npu/service/transform/npu_scalar_fma_compensation.cc](../xla_npu/service/transform/npu_scalar_fma_compensation.cc) | 0 | 138 | 0 | 138 | 138 | 1 | 138 |
| 40 | 55 | [xla_npu/dependency/header/metadef/exe_graph/runtime/tensor.h](../xla_npu/dependency/header/metadef/exe_graph/runtime/tensor.h) | 328 | 68 | 68 | 136 | 0 | 1 | 328 |
| 41 | 224 | [xla_npu/service/transform/npu_fusion_whitelist.h](../xla_npu/service/transform/npu_fusion_whitelist.h) | 51 | 115 | 19 | 134 | 96 | 31 | 147 |
| 42 | 53 | [xla_npu/dependency/external/afir/Ascend-MLIR/python/runtime/compiler.py](../xla_npu/dependency/external/afir/Ascend-MLIR/python/runtime/compiler.py) | 355 | 66 | 66 | 132 | 0 | 1 | 355 |
| 43 | 78 | [xla_npu/dependency/header/acl_nn_op/aclnn_clamp.h](../xla_npu/dependency/header/acl_nn_op/aclnn_clamp.h) | 231 | 66 | 66 | 132 | 0 | 1 | 231 |
| 44 | 127 | [xla_npu/dependency/header/acl_nn_op/aclnn_convolution_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_convolution_backward.h) | 132 | 66 | 66 | 132 | 0 | 1 | 132 |
| 45 | 56 | [xla_npu/dependency/header/metadef/exe_graph/runtime/compute_node_info.h](../xla_npu/dependency/header/metadef/exe_graph/runtime/compute_node_info.h) | 332 | 65 | 65 | 130 | 0 | 1 | 332 |
| 46 | 45 | [xla_npu/dependency/header/metadef/exe_graph/runtime/tiling_context.h](../xla_npu/dependency/header/metadef/exe_graph/runtime/tiling_context.h) | 395 | 64 | 64 | 128 | 0 | 1 | 395 |
| 47 | 144 | [xla_npu/dependency/interface/acl_nn_op_interface.cc](../xla_npu/dependency/interface/acl_nn_op_interface.cc) | 112 | 128 | 0 | 128 | 128 | 25 | 240 |
| 48 | 417 | [xla_npu/service/transform/npu_fma_addcmul_rewriter_test.cc](../xla_npu/service/transform/npu_fma_addcmul_rewriter_test.cc) | 0 | 127 | 0 | 127 | 127 | 2 | 127 |
| 49 | 141 | [xla_npu/dependency/header/acl_nn_op/aclnn_moe_distribute_combine_add_rms_norm_v2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_moe_distribute_combine_add_rms_norm_v2.h) | 121 | 63 | 63 | 126 | 0 | 1 | 121 |
| 50 | 109 | [xla_npu/dependency/header/acl_nn_op/aclnn_round.h](../xla_npu/dependency/header/acl_nn_op/aclnn_round.h) | 159 | 63 | 63 | 126 | 0 | 1 | 159 |
| 51 | 126 | [xla_npu/dependency/header/acl_nn_op/aclnn_median.h](../xla_npu/dependency/header/acl_nn_op/aclnn_median.h) | 141 | 62 | 62 | 124 | 0 | 1 | 141 |
| 52 | 118 | [xla_npu/dependency/header/acl_nn_op/aclnn_mul.h](../xla_npu/dependency/header/acl_nn_op/aclnn_mul.h) | 152 | 62 | 62 | 124 | 0 | 1 | 152 |
| 53 | 448 | [xla_npu/service/transform/npu_scalar_fma_compensation_test.cc](../xla_npu/service/transform/npu_scalar_fma_compensation_test.cc) | 0 | 122 | 0 | 122 | 122 | 1 | 122 |
| 54 | 81 | [xla_npu/dependency/header/acl_nn_op/aclnn_bitwise_and_tensor.h](../xla_npu/dependency/header/acl_nn_op/aclnn_bitwise_and_tensor.h) | 231 | 60 | 60 | 120 | 0 | 1 | 231 |
| 55 | 455 | [xla_npu/service/transform/npu_entry_param_copy_insertion.cc](../xla_npu/service/transform/npu_entry_param_copy_insertion.cc) | 0 | 118 | 1 | 119 | 117 | 2 | 117 |
| 56 | 90 | [xla_npu/dependency/header/acl_nn_op/aclnn_bitwise_and_scalar.h](../xla_npu/dependency/header/acl_nn_op/aclnn_bitwise_and_scalar.h) | 210 | 58 | 58 | 116 | 0 | 1 | 210 |
| 57 | 57 | [xla_npu/common/npu_tool.h](../xla_npu/common/npu_tool.h) | 346 | 77 | 35 | 112 | 42 | 3 | 388 |
| 58 | 539 | [xla_npu/backend/runtime/npu_while_thunk.h](../xla_npu/backend/runtime/npu_while_thunk.h) | 0 | 100 | 9 | 109 | 91 | 3 | 91 |
| 59 | 565 | [xla_npu/service/transform/npu_while_body_copy_insertion.cc](../xla_npu/service/transform/npu_while_body_copy_insertion.cc) | 0 | 100 | 7 | 107 | 93 | 2 | 93 |
| 60 | 99 | [xla_npu/dependency/header/acl_nn_op/aclnn_addcdiv.h](../xla_npu/dependency/header/acl_nn_op/aclnn_addcdiv.h) | 197 | 53 | 53 | 106 | 0 | 1 | 197 |
| 61 | 151 | [xla_npu/dependency/header/acl_nn_op/aclnn_addbmm.h](../xla_npu/dependency/header/acl_nn_op/aclnn_addbmm.h) | 125 | 52 | 52 | 104 | 0 | 1 | 125 |
| 62 | 125 | [xla_npu/dependency/header/acl_nn_op/aclnn_addcmul.h](../xla_npu/dependency/header/acl_nn_op/aclnn_addcmul.h) | 161 | 52 | 52 | 104 | 0 | 1 | 161 |
| 63 | 180 | [xla_npu/dependency/header/acl_nn_op/aclnn_moe_distribute_combine_add_rms_norm.h](../xla_npu/dependency/header/acl_nn_op/aclnn_moe_distribute_combine_add_rms_norm.h) | 105 | 52 | 52 | 104 | 0 | 1 | 105 |
| 64 | 171 | [xla_npu/dependency/header/acl_nn_op/aclnn_max_pool2d_with_indices_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_max_pool2d_with_indices_backward.h) | 111 | 51 | 51 | 102 | 0 | 1 | 111 |
| 65 | 625 | [xla_npu/backend/runtime/npu_stream_ordered_device_memory_pool.h](../xla_npu/backend/runtime/npu_stream_ordered_device_memory_pool.h) | 0 | 100 | 0 | 100 | 100 | 1 | 100 |
| 66 | 158 | [xla_npu/dependency/header/acl_nn_op/aclnn_geglu_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_geglu_backward.h) | 125 | 49 | 49 | 98 | 0 | 1 | 125 |
| 67 | 95 | [xla_npu/dependency/external/afir/Ascend-MLIR/python/runtime/__init__.py](../xla_npu/dependency/external/afir/Ascend-MLIR/python/runtime/__init__.py) | 218 | 48 | 48 | 96 | 0 | 1 | 218 |
| 68 | 164 | [xla_npu/dependency/header/acl_nn_op/aclnn_baddbmm.h](../xla_npu/dependency/header/acl_nn_op/aclnn_baddbmm.h) | 121 | 48 | 48 | 96 | 0 | 1 | 121 |
| 69 | 178 | [xla_npu/dependency/header/acl_nn_op/aclnn_add_layer_norm_quant.h](../xla_npu/dependency/header/acl_nn_op/aclnn_add_layer_norm_quant.h) | 116 | 47 | 47 | 94 | 0 | 1 | 116 |
| 70 | 143 | [xla_npu/dependency/header/acl_nn_op/aclnn_lt_tensor.h](../xla_npu/dependency/header/acl_nn_op/aclnn_lt_tensor.h) | 149 | 47 | 47 | 94 | 0 | 1 | 149 |
| 71 | 196 | [xla_npu/dependency/header/acl_nn_op/aclnn_moe_distribute_combine_v3.h](../xla_npu/dependency/header/acl_nn_op/aclnn_moe_distribute_combine_v3.h) | 105 | 47 | 47 | 94 | 0 | 1 | 105 |
| 72 | 17 | [xla_npu/service/convert_to_afir/ascir_backend.cc](../xla_npu/service/convert_to_afir/ascir_backend.cc) | 1075 | 79 | 14 | 93 | 65 | 3 | 1140 |
| 73 | 136 | [xla_npu/backend/runtime/npu_workspace_memory_pool.h](../xla_npu/backend/runtime/npu_workspace_memory_pool.h) | 162 | 51 | 41 | 92 | 10 | 2 | 172 |
| 74 | 197 | [xla_npu/dependency/header/acl_nn_op/aclnn_max_pool2d_with_indices.h](../xla_npu/dependency/header/acl_nn_op/aclnn_max_pool2d_with_indices.h) | 104 | 46 | 46 | 92 | 0 | 1 | 104 |
| 75 | 155 | [xla_npu/dependency/header/acl_nn_op/aclnn_relu.h](../xla_npu/dependency/header/acl_nn_op/aclnn_relu.h) | 136 | 45 | 45 | 90 | 0 | 1 | 136 |
| 76 | 150 | [xla_npu/dependency/header/acl_nn_op/aclnn_tanh.h](../xla_npu/dependency/header/acl_nn_op/aclnn_tanh.h) | 141 | 45 | 45 | 90 | 0 | 1 | 141 |
| 77 | 709 | [xla_npu/service/transform/npu_unsigned_to_float_decomposition.h](../xla_npu/service/transform/npu_unsigned_to_float_decomposition.h) | 0 | 75 | 15 | 90 | 60 | 2 | 60 |
| 78 | 153 | [xla_npu/dependency/header/acl_nn_op/aclnn_cosh.h](../xla_npu/dependency/header/acl_nn_op/aclnn_cosh.h) | 140 | 44 | 44 | 88 | 0 | 1 | 140 |
| 79 | 213 | [xla_npu/dependency/header/acl_nn_op/aclnn_pad2d_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_pad2d_backward.h) | 102 | 43 | 43 | 86 | 0 | 1 | 102 |
| 80 | 148 | [xla_npu/dependency/header/acl_nn_op/aclnn_uniform.h](../xla_npu/dependency/header/acl_nn_op/aclnn_uniform.h) | 148 | 43 | 43 | 86 | 0 | 1 | 148 |
| 81 | 749 | [xla_npu/backend/runtime/npu_conditional_thunk.h](../xla_npu/backend/runtime/npu_conditional_thunk.h) | 0 | 78 | 6 | 84 | 72 | 2 | 72 |
| 82 | 165 | [xla_npu/dependency/header/acl_nn_op/aclnn_add.h](../xla_npu/dependency/header/acl_nn_op/aclnn_add.h) | 132 | 42 | 42 | 84 | 0 | 1 | 132 |
| 83 | 166 | [xla_npu/dependency/header/acl_nn_op/aclnn_bitwise_xor_tensor.h](../xla_npu/dependency/header/acl_nn_op/aclnn_bitwise_xor_tensor.h) | 132 | 42 | 42 | 84 | 0 | 1 | 132 |
| 84 | 203 | [xla_npu/dependency/header/acl_nn_op/aclnn_geglu.h](../xla_npu/dependency/header/acl_nn_op/aclnn_geglu.h) | 108 | 42 | 42 | 84 | 0 | 1 | 108 |
| 85 | 234 | [xla_npu/dependency/header/acl_nn_op/aclnn_moe_distribute_dispatch_v3.h](../xla_npu/dependency/header/acl_nn_op/aclnn_moe_distribute_dispatch_v3.h) | 94 | 42 | 42 | 84 | 0 | 1 | 94 |
| 86 | 217 | [xla_npu/dependency/header/acl_nn_op/aclnn_rsub.h](../xla_npu/dependency/header/acl_nn_op/aclnn_rsub.h) | 103 | 42 | 42 | 84 | 0 | 1 | 103 |
| 87 | 167 | [xla_npu/dependency/header/acl_nn_op/aclnn_sub.h](../xla_npu/dependency/header/acl_nn_op/aclnn_sub.h) | 131 | 42 | 42 | 84 | 0 | 1 | 131 |
| 88 | 163 | [xla_npu/dependency/header/acl_nn_op/aclnn_tan.h](../xla_npu/dependency/header/acl_nn_op/aclnn_tan.h) | 134 | 42 | 42 | 84 | 0 | 1 | 134 |
| 89 | 50 | [xla_npu/pjrt/air/mair/test/test_mair_gen.cc](../xla_npu/pjrt/air/mair/test/test_mair_gen.cc) | 413 | 42 | 42 | 84 | 0 | 1 | 413 |
| 90 | 221 | [xla_npu/dependency/header/acl_nn_op/aclnn_add_rms_norm_quant_v2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_add_rms_norm_quant_v2.h) | 103 | 41 | 41 | 82 | 0 | 1 | 103 |
| 91 | 222 | [xla_npu/dependency/header/acl_nn_op/aclnn_atan2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_atan2.h) | 103 | 41 | 41 | 82 | 0 | 1 | 103 |
| 92 | 179 | [xla_npu/dependency/header/acl_nn_op/aclnn_fmod_scalar.h](../xla_npu/dependency/header/acl_nn_op/aclnn_fmod_scalar.h) | 127 | 41 | 41 | 82 | 0 | 1 | 127 |
| 93 | 182 | [xla_npu/dependency/header/acl_nn_op/aclnn_fmod_tensor.h](../xla_npu/dependency/header/acl_nn_op/aclnn_fmod_tensor.h) | 126 | 41 | 41 | 82 | 0 | 1 | 126 |
| 94 | 159 | [xla_npu/dependency/header/acl_nn_op/aclnn_hardtanh.h](../xla_npu/dependency/header/acl_nn_op/aclnn_hardtanh.h) | 138 | 41 | 41 | 82 | 0 | 1 | 138 |
| 95 | 218 | [xla_npu/dependency/header/acl_nn_op/aclnn_searchsorted.h](../xla_npu/dependency/header/acl_nn_op/aclnn_searchsorted.h) | 105 | 41 | 41 | 82 | 0 | 1 | 105 |
| 96 | 766 | [xla_npu/service/transform/npu_shift_left_decomposition.h](../xla_npu/service/transform/npu_shift_left_decomposition.h) | 0 | 66 | 15 | 81 | 51 | 3 | 51 |
| 97 | 181 | [xla_npu/dependency/header/acl_nn_op/aclnn_bitwise_xor_scalar.h](../xla_npu/dependency/header/acl_nn_op/aclnn_bitwise_xor_scalar.h) | 128 | 40 | 40 | 80 | 0 | 1 | 128 |
| 98 | 199 | [xla_npu/dependency/header/acl_nn_op/aclnn_eq_tensor.h](../xla_npu/dependency/header/acl_nn_op/aclnn_eq_tensor.h) | 114 | 40 | 40 | 80 | 0 | 1 | 114 |
| 99 | 183 | [xla_npu/dependency/header/acl_nn_op/aclnn_bitwise_or_scalar.h](../xla_npu/dependency/header/acl_nn_op/aclnn_bitwise_or_scalar.h) | 129 | 39 | 39 | 78 | 0 | 1 | 129 |
| 100 | 185 | [xla_npu/dependency/header/acl_nn_op/aclnn_bitwise_or_tensor.h](../xla_npu/dependency/header/acl_nn_op/aclnn_bitwise_or_tensor.h) | 127 | 39 | 39 | 78 | 0 | 1 | 127 |
| 101 | 202 | [xla_npu/dependency/header/acl_nn_op/aclnn_elu.h](../xla_npu/dependency/header/acl_nn_op/aclnn_elu.h) | 114 | 39 | 39 | 78 | 0 | 1 | 114 |
| 102 | 174 | [xla_npu/dependency/header/acl_nn_op/aclnn_lt_scalar.h](../xla_npu/dependency/header/acl_nn_op/aclnn_lt_scalar.h) | 134 | 39 | 39 | 78 | 0 | 1 | 134 |
| 103 | 271 | [xla_npu/dependency/header/acl_nn_op/aclnn_moe_distribute_combine.h](../xla_npu/dependency/header/acl_nn_op/aclnn_moe_distribute_combine.h) | 90 | 39 | 39 | 78 | 0 | 1 | 90 |
| 104 | 262 | [xla_npu/dependency/header/acl_nn_op/aclnn_moe_distribute_combine_v2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_moe_distribute_combine_v2.h) | 92 | 39 | 39 | 78 | 0 | 1 | 92 |
| 105 | 264 | [xla_npu/dependency/header/acl_nn_op/aclnn_quant_convolution.h](../xla_npu/dependency/header/acl_nn_op/aclnn_quant_convolution.h) | 92 | 39 | 39 | 78 | 0 | 1 | 92 |
| 106 | 106 | [xla_npu/dependency/header/metadef/exe_graph/runtime/op_execute_context.h](../xla_npu/dependency/header/metadef/exe_graph/runtime/op_execute_context.h) | 213 | 39 | 39 | 78 | 0 | 1 | 213 |
| 107 | 775 | [xla_npu/service/transform/npu_tril_rewriter.cc](../xla_npu/service/transform/npu_tril_rewriter.cc) | 0 | 75 | 2 | 77 | 73 | 2 | 73 |
| 108 | 267 | [xla_npu/dependency/header/acl_nn_op/aclnn_all_to_all_all_gather_batch_matmul.h](../xla_npu/dependency/header/acl_nn_op/aclnn_all_to_all_all_gather_batch_matmul.h) | 92 | 38 | 38 | 76 | 0 | 1 | 92 |
| 109 | 193 | [xla_npu/dependency/header/acl_nn_op/aclnn_logical_and.h](../xla_npu/dependency/header/acl_nn_op/aclnn_logical_and.h) | 124 | 38 | 38 | 76 | 0 | 1 | 124 |
| 110 | 288 | [xla_npu/dependency/header/acl_nn_op/aclnn_moe_distribute_dispatch.h](../xla_npu/dependency/header/acl_nn_op/aclnn_moe_distribute_dispatch.h) | 87 | 38 | 38 | 76 | 0 | 1 | 87 |
| 111 | 283 | [xla_npu/dependency/header/acl_nn_op/aclnn_moe_distribute_dispatch_v2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_moe_distribute_dispatch_v2.h) | 89 | 38 | 38 | 76 | 0 | 1 | 89 |
| 112 | 239 | [xla_npu/dependency/header/acl_nn_op/aclnn_random.h](../xla_npu/dependency/header/acl_nn_op/aclnn_random.h) | 101 | 38 | 38 | 76 | 0 | 1 | 101 |
| 113 | 781 | [xla_npu/service/transform/npu_dot_contracting_dims_normalizer.h](../xla_npu/service/transform/npu_dot_contracting_dims_normalizer.h) | 0 | 74 | 2 | 76 | 72 | 2 | 72 |
| 114 | 257 | [xla_npu/dependency/header/acl_nn_op/aclnn_add_rms_norm_dynamic_quant_v2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_add_rms_norm_dynamic_quant_v2.h) | 97 | 37 | 37 | 74 | 0 | 1 | 97 |
| 115 | 258 | [xla_npu/dependency/header/acl_nn_op/aclnn_add_rms_norm_quant.h](../xla_npu/dependency/header/acl_nn_op/aclnn_add_rms_norm_quant.h) | 97 | 37 | 37 | 74 | 0 | 1 | 97 |
| 116 | 270 | [xla_npu/dependency/header/acl_nn_op/aclnn_ge_tensor.h](../xla_npu/dependency/header/acl_nn_op/aclnn_ge_tensor.h) | 94 | 37 | 37 | 74 | 0 | 1 | 94 |
| 117 | 204 | [xla_npu/dependency/header/acl_nn_op/aclnn_gt_tensor.h](../xla_npu/dependency/header/acl_nn_op/aclnn_gt_tensor.h) | 118 | 37 | 37 | 74 | 0 | 1 | 118 |
| 118 | 154 | [xla_npu/dependency/header/acl_nn_op/aclnn_hardsigmoid.h](../xla_npu/dependency/header/acl_nn_op/aclnn_hardsigmoid.h) | 153 | 37 | 37 | 74 | 0 | 1 | 153 |
| 119 | 230 | [xla_npu/dependency/header/acl_nn_op/aclnn_index_copy.h](../xla_npu/dependency/header/acl_nn_op/aclnn_index_copy.h) | 105 | 37 | 37 | 74 | 0 | 1 | 105 |
| 120 | 216 | [xla_npu/dependency/header/acl_nn_op/aclnn_logical_or.h](../xla_npu/dependency/header/acl_nn_op/aclnn_logical_or.h) | 113 | 37 | 37 | 74 | 0 | 1 | 113 |
| 121 | 194 | [xla_npu/dependency/header/acl_nn_op/aclnn_sqrt.h](../xla_npu/dependency/header/acl_nn_op/aclnn_sqrt.h) | 126 | 37 | 37 | 74 | 0 | 1 | 126 |
| 122 | 110 | [xla_npu/backend/runtime/BUILD](../xla_npu/backend/runtime/BUILD) | 212 | 66 | 6 | 72 | 60 | 7 | 272 |
| 123 | 243 | [xla_npu/dependency/header/acl_nn_op/aclnn_add_relu.h](../xla_npu/dependency/header/acl_nn_op/aclnn_add_relu.h) | 103 | 36 | 36 | 72 | 0 | 1 | 103 |
| 124 | 249 | [xla_npu/dependency/header/acl_nn_op/aclnn_asin.h](../xla_npu/dependency/header/acl_nn_op/aclnn_asin.h) | 101 | 36 | 36 | 72 | 0 | 1 | 101 |
| 125 | 209 | [xla_npu/dependency/header/acl_nn_op/aclnn_erfc.h](../xla_npu/dependency/header/acl_nn_op/aclnn_erfc.h) | 119 | 36 | 36 | 72 | 0 | 1 | 119 |
| 126 | 250 | [xla_npu/dependency/header/acl_nn_op/aclnn_gather_nd.h](../xla_npu/dependency/header/acl_nn_op/aclnn_gather_nd.h) | 101 | 36 | 36 | 72 | 0 | 1 | 101 |
| 127 | 152 | [xla_npu/dependency/header/acl_nn_op/aclnn_hardswish.h](../xla_npu/dependency/header/acl_nn_op/aclnn_hardswish.h) | 157 | 36 | 36 | 72 | 0 | 1 | 157 |
| 128 | 228 | [xla_npu/dependency/header/acl_nn_op/aclnn_lerp_tensor.h](../xla_npu/dependency/header/acl_nn_op/aclnn_lerp_tensor.h) | 109 | 36 | 36 | 72 | 0 | 1 | 109 |
| 129 | 263 | [xla_npu/dependency/header/acl_nn_op/aclnn_prod.h](../xla_npu/dependency/header/acl_nn_op/aclnn_prod.h) | 98 | 36 | 36 | 72 | 0 | 1 | 98 |
| 130 | 279 | [xla_npu/dependency/header/acl_nn_op/aclnn_renorm.h](../xla_npu/dependency/header/acl_nn_op/aclnn_renorm.h) | 94 | 36 | 36 | 72 | 0 | 1 | 94 |
| 131 | 293 | [xla_npu/dependency/header/acl_nn_op/aclnn_threshold.h](../xla_npu/dependency/header/acl_nn_op/aclnn_threshold.h) | 89 | 36 | 36 | 72 | 0 | 1 | 89 |
| 132 | 253 | [xla_npu/dependency/header/acl_nn_op/aclnn_acos.h](../xla_npu/dependency/header/acl_nn_op/aclnn_acos.h) | 102 | 35 | 35 | 70 | 0 | 1 | 102 |
| 133 | 286 | [xla_npu/dependency/header/acl_nn_op/aclnn_add_rms_norm_dynamic_quant.h](../xla_npu/dependency/header/acl_nn_op/aclnn_add_rms_norm_dynamic_quant.h) | 93 | 35 | 35 | 70 | 0 | 1 | 93 |
| 134 | 261 | [xla_npu/dependency/header/acl_nn_op/aclnn_atan.h](../xla_npu/dependency/header/acl_nn_op/aclnn_atan.h) | 100 | 35 | 35 | 70 | 0 | 1 | 100 |
| 135 | 315 | [xla_npu/dependency/header/acl_nn_op/aclnn_batch_matmul_reduce_scatter_all_to_all.h](../xla_npu/dependency/header/acl_nn_op/aclnn_batch_matmul_reduce_scatter_all_to_all.h) | 82 | 35 | 35 | 70 | 0 | 1 | 82 |
| 136 | 254 | [xla_npu/dependency/header/acl_nn_op/aclnn_cos.h](../xla_npu/dependency/header/acl_nn_op/aclnn_cos.h) | 102 | 35 | 35 | 70 | 0 | 1 | 102 |
| 137 | 215 | [xla_npu/dependency/header/acl_nn_op/aclnn_erf.h](../xla_npu/dependency/header/acl_nn_op/aclnn_erf.h) | 117 | 35 | 35 | 70 | 0 | 1 | 117 |
| 138 | 244 | [xla_npu/dependency/header/acl_nn_op/aclnn_ffn_v2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_ffn_v2.h) | 105 | 35 | 35 | 70 | 0 | 1 | 105 |
| 139 | 233 | [xla_npu/dependency/header/acl_nn_op/aclnn_ffn_v3.h](../xla_npu/dependency/header/acl_nn_op/aclnn_ffn_v3.h) | 108 | 35 | 35 | 70 | 0 | 1 | 108 |
| 140 | 290 | [xla_npu/dependency/header/acl_nn_op/aclnn_ge_scalar.h](../xla_npu/dependency/header/acl_nn_op/aclnn_ge_scalar.h) | 92 | 35 | 35 | 70 | 0 | 1 | 92 |
| 141 | 297 | [xla_npu/dependency/header/acl_nn_op/aclnn_grouped_mat_mul_allto_allv.h](../xla_npu/dependency/header/acl_nn_op/aclnn_grouped_mat_mul_allto_allv.h) | 89 | 35 | 35 | 70 | 0 | 1 | 89 |
| 142 | 241 | [xla_npu/dependency/header/acl_nn_op/aclnn_lerp_scalar.h](../xla_npu/dependency/header/acl_nn_op/aclnn_lerp_scalar.h) | 106 | 35 | 35 | 70 | 0 | 1 | 106 |
| 143 | 275 | [xla_npu/dependency/header/acl_nn_op/aclnn_normal.h](../xla_npu/dependency/header/acl_nn_op/aclnn_normal.h) | 97 | 35 | 35 | 70 | 0 | 1 | 97 |
| 144 | 251 | [xla_npu/dependency/header/acl_nn_op/aclnn_pow.h](../xla_npu/dependency/header/acl_nn_op/aclnn_pow.h) | 103 | 35 | 35 | 70 | 0 | 1 | 103 |
| 145 | 259 | [xla_npu/dependency/header/acl_nn_op/aclnn_triu.h](../xla_npu/dependency/header/acl_nn_op/aclnn_triu.h) | 101 | 35 | 35 | 70 | 0 | 1 | 101 |
| 146 | 107 | [xla_npu/dependency/header/metadef/exe_graph/runtime/shape.h](../xla_npu/dependency/header/metadef/exe_graph/runtime/shape.h) | 220 | 35 | 35 | 70 | 0 | 1 | 220 |
| 147 | 281 | [xla_npu/dependency/header/acl_nn_op/aclnn_asinh.h](../xla_npu/dependency/header/acl_nn_op/aclnn_asinh.h) | 97 | 34 | 34 | 68 | 0 | 1 | 97 |
| 148 | 282 | [xla_npu/dependency/header/acl_nn_op/aclnn_atanh.h](../xla_npu/dependency/header/acl_nn_op/aclnn_atanh.h) | 97 | 34 | 34 | 68 | 0 | 1 | 97 |
| 149 | 255 | [xla_npu/dependency/header/acl_nn_op/aclnn_ffn.h](../xla_npu/dependency/header/acl_nn_op/aclnn_ffn.h) | 104 | 34 | 34 | 68 | 0 | 1 | 104 |
| 150 | 291 | [xla_npu/dependency/header/acl_nn_op/aclnn_grouped_bias_add_grad.h](../xla_npu/dependency/header/acl_nn_op/aclnn_grouped_bias_add_grad.h) | 93 | 34 | 34 | 68 | 0 | 1 | 93 |
| 151 | 303 | [xla_npu/dependency/header/acl_nn_op/aclnn_le_scalar.h](../xla_npu/dependency/header/acl_nn_op/aclnn_le_scalar.h) | 90 | 34 | 34 | 68 | 0 | 1 | 90 |
| 152 | 298 | [xla_npu/dependency/header/acl_nn_op/aclnn_le_tensor.h](../xla_npu/dependency/header/acl_nn_op/aclnn_le_tensor.h) | 91 | 34 | 34 | 68 | 0 | 1 | 91 |
| 153 | 227 | [xla_npu/dependency/header/acl_nn_op/aclnn_log10.h](../xla_npu/dependency/header/acl_nn_op/aclnn_log10.h) | 114 | 34 | 34 | 68 | 0 | 1 | 114 |
| 154 | 304 | [xla_npu/dependency/header/acl_nn_op/aclnn_ne_tensor.h](../xla_npu/dependency/header/acl_nn_op/aclnn_ne_tensor.h) | 89 | 34 | 34 | 68 | 0 | 1 | 89 |
| 155 | 225 | [xla_npu/dependency/header/acl_nn_op/aclnn_neg.h](../xla_npu/dependency/header/acl_nn_op/aclnn_neg.h) | 116 | 34 | 34 | 68 | 0 | 1 | 116 |
| 156 | 248 | [xla_npu/dependency/header/acl_nn_op/aclnn_norm_rope_concat.h](../xla_npu/dependency/header/acl_nn_op/aclnn_norm_rope_concat.h) | 106 | 34 | 34 | 68 | 0 | 1 | 106 |
| 157 | 115 | [xla_npu/dependency/header/metadef/exe_graph/runtime/extended_kernel_context.h](../xla_npu/dependency/header/metadef/exe_graph/runtime/extended_kernel_context.h) | 212 | 34 | 34 | 68 | 0 | 1 | 212 |
| 158 | 811 | [xla_npu/service/transform/npu_unsupported_fusion_defuser.h](../xla_npu/service/transform/npu_unsupported_fusion_defuser.h) | 0 | 58 | 10 | 68 | 48 | 6 | 48 |
| 159 | 818 | [xla_npu/service/transform/npu_right_triangular_solve_expander_test.cc](../xla_npu/service/transform/npu_right_triangular_solve_expander_test.cc) | 0 | 67 | 0 | 67 | 67 | 1 | 67 |
| 160 | 268 | [xla_npu/dependency/header/acl_nn_op/aclnn_celu.h](../xla_npu/dependency/header/acl_nn_op/aclnn_celu.h) | 102 | 33 | 33 | 66 | 0 | 1 | 102 |
| 161 | 273 | [xla_npu/dependency/header/acl_nn_op/aclnn_ctc_loss.h](../xla_npu/dependency/header/acl_nn_op/aclnn_ctc_loss.h) | 101 | 33 | 33 | 66 | 0 | 1 | 101 |
| 162 | 238 | [xla_npu/dependency/header/acl_nn_op/aclnn_eq_scalar.h](../xla_npu/dependency/header/acl_nn_op/aclnn_eq_scalar.h) | 111 | 33 | 33 | 66 | 0 | 1 | 111 |
| 163 | 285 | [xla_npu/dependency/header/acl_nn_op/aclnn_frac.h](../xla_npu/dependency/header/acl_nn_op/aclnn_frac.h) | 98 | 33 | 33 | 66 | 0 | 1 | 98 |
| 164 | 274 | [xla_npu/dependency/header/acl_nn_op/aclnn_gather_v3.h](../xla_npu/dependency/header/acl_nn_op/aclnn_gather_v3.h) | 101 | 33 | 33 | 66 | 0 | 1 | 101 |
| 165 | 184 | [xla_npu/dependency/header/metadef/exe_graph/runtime/runtime_attrs.h](../xla_npu/dependency/header/metadef/exe_graph/runtime/runtime_attrs.h) | 141 | 33 | 33 | 66 | 0 | 1 | 141 |
| 166 | 828 | [xla_npu/service/transform/npu_right_triangular_solve_expander.cc](../xla_npu/service/transform/npu_right_triangular_solve_expander.cc) | 0 | 66 | 0 | 66 | 66 | 1 | 66 |
| 167 | 829 | [xla_npu/service/transform/npu_scatter_copy_insertion.cc](../xla_npu/service/transform/npu_scatter_copy_insertion.cc) | 0 | 66 | 0 | 66 | 66 | 1 | 66 |
| 168 | 308 | [xla_npu/dependency/header/acl_nn_op/aclnn_ctc_loss_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_ctc_loss_backward.h) | 90 | 32 | 32 | 64 | 0 | 1 | 90 |
| 169 | 323 | [xla_npu/dependency/header/acl_nn_op/aclnn_cumprod.h](../xla_npu/dependency/header/acl_nn_op/aclnn_cumprod.h) | 87 | 32 | 32 | 64 | 0 | 1 | 87 |
| 170 | 277 | [xla_npu/dependency/header/acl_nn_op/aclnn_erfinv.h](../xla_npu/dependency/header/acl_nn_op/aclnn_erfinv.h) | 102 | 32 | 32 | 64 | 0 | 1 | 102 |
| 171 | 269 | [xla_npu/dependency/header/acl_nn_op/aclnn_expm1.h](../xla_npu/dependency/header/acl_nn_op/aclnn_expm1.h) | 104 | 32 | 32 | 64 | 0 | 1 | 104 |
| 172 | 242 | [xla_npu/dependency/header/acl_nn_op/aclnn_log2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_log2.h) | 112 | 32 | 32 | 64 | 0 | 1 | 112 |
| 173 | 335 | [xla_npu/dependency/header/acl_nn_op/aclnn_moe_fused_topk.h](../xla_npu/dependency/header/acl_nn_op/aclnn_moe_fused_topk.h) | 83 | 32 | 32 | 64 | 0 | 1 | 83 |
| 174 | 329 | [xla_npu/dependency/header/acl_nn_op/aclnn_nan_to_num.h](../xla_npu/dependency/header/acl_nn_op/aclnn_nan_to_num.h) | 85 | 32 | 32 | 64 | 0 | 1 | 85 |
| 175 | 318 | [xla_npu/dependency/header/acl_nn_op/aclnn_ne_scalar.h](../xla_npu/dependency/header/acl_nn_op/aclnn_ne_scalar.h) | 88 | 32 | 32 | 64 | 0 | 1 | 88 |
| 176 | 265 | [xla_npu/dependency/header/acl_nn_op/aclnn_swi_glu_quant_v2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_swi_glu_quant_v2.h) | 105 | 32 | 32 | 64 | 0 | 1 | 105 |
| 177 | 20 | [xla_npu/pjrt/air/mair/transforms/mair_passes.cc](../xla_npu/pjrt/air/mair/transforms/mair_passes.cc) | 1038 | 32 | 32 | 64 | 0 | 1 | 1038 |
| 178 | 307 | [xla_npu/dependency/header/acl_nn_op/aclnn_acosh.h](../xla_npu/dependency/header/acl_nn_op/aclnn_acosh.h) | 92 | 31 | 31 | 62 | 0 | 1 | 92 |
| 179 | 326 | [xla_npu/dependency/header/acl_nn_op/aclnn_allto_allv_grouped_mat_mul.h](../xla_npu/dependency/header/acl_nn_op/aclnn_allto_allv_grouped_mat_mul.h) | 87 | 31 | 31 | 62 | 0 | 1 | 87 |
| 180 | 316 | [xla_npu/dependency/header/acl_nn_op/aclnn_group_norm_silu.h](../xla_npu/dependency/header/acl_nn_op/aclnn_group_norm_silu.h) | 90 | 31 | 31 | 62 | 0 | 1 | 90 |
| 181 | 341 | [xla_npu/dependency/header/acl_nn_op/aclnn_grouped_matmul_finalize_routing_weight_nz_v2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_grouped_matmul_finalize_routing_weight_nz_v2.h) | 83 | 31 | 31 | 62 | 0 | 1 | 83 |
| 182 | 292 | [xla_npu/dependency/header/acl_nn_op/aclnn_grouped_matmul_v4.h](../xla_npu/dependency/header/acl_nn_op/aclnn_grouped_matmul_v4.h) | 99 | 31 | 31 | 62 | 0 | 1 | 99 |
| 183 | 338 | [xla_npu/dependency/header/acl_nn_op/aclnn_grouped_matmul_weight_nz.h](../xla_npu/dependency/header/acl_nn_op/aclnn_grouped_matmul_weight_nz.h) | 84 | 31 | 31 | 62 | 0 | 1 | 84 |
| 184 | 317 | [xla_npu/dependency/header/acl_nn_op/aclnn_gt_scalar.h](../xla_npu/dependency/header/acl_nn_op/aclnn_gt_scalar.h) | 90 | 31 | 31 | 62 | 0 | 1 | 90 |
| 185 | 278 | [xla_npu/dependency/header/acl_nn_op/aclnn_log.h](../xla_npu/dependency/header/acl_nn_op/aclnn_log.h) | 104 | 31 | 31 | 62 | 0 | 1 | 104 |
| 186 | 289 | [xla_npu/dependency/header/acl_nn_op/aclnn_tril.h](../xla_npu/dependency/header/acl_nn_op/aclnn_tril.h) | 101 | 31 | 31 | 62 | 0 | 1 | 101 |
| 187 | 91 | [xla_npu/dependency/header/metadef/exe_graph/runtime/tensor_data.h](../xla_npu/dependency/header/metadef/exe_graph/runtime/tensor_data.h) | 263 | 31 | 31 | 62 | 0 | 1 | 263 |
| 188 | 121 | [xla_npu/backend/runtime/npu_workspace_memory_pool.cc](../xla_npu/backend/runtime/npu_workspace_memory_pool.cc) | 206 | 58 | 3 | 61 | 55 | 1 | 261 |
| 189 | 214 | [xla_npu/dependency/header/acl_nn_op/aclnn_embedding_renorm.h](../xla_npu/dependency/header/acl_nn_op/aclnn_embedding_renorm.h) | 127 | 30 | 30 | 60 | 0 | 1 | 127 |
| 190 | 295 | [xla_npu/dependency/header/acl_nn_op/aclnn_exp.h](../xla_npu/dependency/header/acl_nn_op/aclnn_exp.h) | 100 | 30 | 30 | 60 | 0 | 1 | 100 |
| 191 | 363 | [xla_npu/dependency/header/acl_nn_op/aclnn_grouped_matmul_finalize_routing_v3.h](../xla_npu/dependency/header/acl_nn_op/aclnn_grouped_matmul_finalize_routing_v3.h) | 81 | 30 | 30 | 60 | 0 | 1 | 81 |
| 192 | 356 | [xla_npu/dependency/header/acl_nn_op/aclnn_grouped_matmul_v5.h](../xla_npu/dependency/header/acl_nn_op/aclnn_grouped_matmul_v5.h) | 83 | 30 | 30 | 60 | 0 | 1 | 83 |
| 193 | 287 | [xla_npu/dependency/header/acl_nn_op/aclnn_hardtanh_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_hardtanh_backward.h) | 103 | 30 | 30 | 60 | 0 | 1 | 103 |
| 194 | 348 | [xla_npu/dependency/header/acl_nn_op/aclnn_log1p.h](../xla_npu/dependency/header/acl_nn_op/aclnn_log1p.h) | 84 | 30 | 30 | 60 | 0 | 1 | 84 |
| 195 | 299 | [xla_npu/dependency/header/acl_nn_op/aclnn_logical_not.h](../xla_npu/dependency/header/acl_nn_op/aclnn_logical_not.h) | 99 | 30 | 30 | 60 | 0 | 1 | 99 |
| 196 | 378 | [xla_npu/dependency/header/acl_nn_op/aclnn_quant_scatter_v2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_quant_scatter_v2.h) | 77 | 30 | 30 | 60 | 0 | 1 | 77 |
| 197 | 349 | [xla_npu/dependency/header/acl_nn_op/aclnn_selu.h](../xla_npu/dependency/header/acl_nn_op/aclnn_selu.h) | 84 | 30 | 30 | 60 | 0 | 1 | 84 |
| 198 | 342 | [xla_npu/dependency/header/acl_nn_op/aclnn_sigmoid.h](../xla_npu/dependency/header/acl_nn_op/aclnn_sigmoid.h) | 85 | 30 | 30 | 60 | 0 | 1 | 85 |
| 199 | 343 | [xla_npu/dependency/header/acl_nn_op/aclnn_sin.h](../xla_npu/dependency/header/acl_nn_op/aclnn_sin.h) | 85 | 30 | 30 | 60 | 0 | 1 | 85 |
| 200 | 350 | [xla_npu/dependency/header/acl_nn_op/aclnn_sinc.h](../xla_npu/dependency/header/acl_nn_op/aclnn_sinc.h) | 84 | 30 | 30 | 60 | 0 | 1 | 84 |
| 201 | 351 | [xla_npu/dependency/header/acl_nn_op/aclnn_sinh.h](../xla_npu/dependency/header/acl_nn_op/aclnn_sinh.h) | 84 | 30 | 30 | 60 | 0 | 1 | 84 |
| 202 | 296 | [xla_npu/dependency/header/acl_nn_op/aclnn_swi_glu_quant.h](../xla_npu/dependency/header/acl_nn_op/aclnn_swi_glu_quant.h) | 100 | 30 | 30 | 60 | 0 | 1 | 100 |
| 203 | 135 | [xla_npu/dependency/header/metadef/exe_graph/runtime/expand_dims_type.h](../xla_npu/dependency/header/metadef/exe_graph/runtime/expand_dims_type.h) | 195 | 30 | 30 | 60 | 0 | 1 | 195 |
| 204 | 327 | [xla_npu/dependency/header/acl_nn_op/aclnn_ceil.h](../xla_npu/dependency/header/acl_nn_op/aclnn_ceil.h) | 91 | 29 | 29 | 58 | 0 | 1 | 91 |
| 205 | 377 | [xla_npu/dependency/header/acl_nn_op/aclnn_grouped_matmul_finalize_routing_v2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_grouped_matmul_finalize_routing_v2.h) | 79 | 29 | 29 | 58 | 0 | 1 | 79 |
| 206 | 328 | [xla_npu/dependency/header/acl_nn_op/aclnn_leaky_relu.h](../xla_npu/dependency/header/acl_nn_op/aclnn_leaky_relu.h) | 91 | 29 | 29 | 58 | 0 | 1 | 91 |
| 207 | 383 | [xla_npu/dependency/header/acl_nn_op/aclnn_quant_scatter.h](../xla_npu/dependency/header/acl_nn_op/aclnn_quant_scatter.h) | 76 | 29 | 29 | 58 | 0 | 1 | 76 |
| 208 | 365 | [xla_npu/dependency/header/acl_nn_op/aclnn_trunc.h](../xla_npu/dependency/header/acl_nn_op/aclnn_trunc.h) | 83 | 29 | 29 | 58 | 0 | 1 | 83 |
| 209 | 371 | [xla_npu/dependency/header/acl_nn_op/aclnn_exp2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_exp2.h) | 83 | 28 | 28 | 56 | 0 | 1 | 83 |
| 210 | 325 | [xla_npu/dependency/header/acl_nn_op/aclnn_exp_segsum_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_exp_segsum_backward.h) | 94 | 28 | 28 | 56 | 0 | 1 | 94 |
| 211 | 340 | [xla_npu/dependency/header/acl_nn_op/aclnn_floor.h](../xla_npu/dependency/header/acl_nn_op/aclnn_floor.h) | 89 | 28 | 28 | 56 | 0 | 1 | 89 |
| 212 | 113 | [xla_npu/dependency/header/metadef/exe_graph/runtime/tiling_data.h](../xla_npu/dependency/header/metadef/exe_graph/runtime/tiling_data.h) | 227 | 28 | 28 | 56 | 0 | 1 | 227 |
| 213 | 885 | [xla_npu/dependency/interface/acl_custom_op_interface.h](../xla_npu/dependency/interface/acl_custom_op_interface.h) | 0 | 55 | 0 | 55 | 55 | 2 | 55 |
| 214 | 367 | [xla_npu/dependency/header/acl_nn_op/aclnn_elu_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_elu_backward.h) | 86 | 27 | 27 | 54 | 0 | 1 | 86 |
| 215 | 369 | [xla_npu/dependency/header/acl_nn_op/aclnn_mish.h](../xla_npu/dependency/header/acl_nn_op/aclnn_mish.h) | 86 | 27 | 27 | 54 | 0 | 1 | 86 |
| 216 | 382 | [xla_npu/dependency/header/acl_nn_op/aclnn_quant_matmul_all_reduce_v4.h](../xla_npu/dependency/header/acl_nn_op/aclnn_quant_matmul_all_reduce_v4.h) | 80 | 27 | 27 | 54 | 0 | 1 | 80 |
| 217 | 357 | [xla_npu/dependency/header/acl_nn_op/aclnn_segsum.h](../xla_npu/dependency/header/acl_nn_op/aclnn_segsum.h) | 89 | 27 | 27 | 54 | 0 | 1 | 89 |
| 218 | 402 | [xla_npu/dependency/header/acl_nn_op/aclnn_silent_check_v2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_silent_check_v2.h) | 75 | 27 | 27 | 54 | 0 | 1 | 75 |
| 219 | 388 | [xla_npu/dependency/header/acl_nn_op/aclnn_weight_quant_matmul_all_reduce_add_rms_norm.h](../xla_npu/dependency/header/acl_nn_op/aclnn_weight_quant_matmul_all_reduce_add_rms_norm.h) | 79 | 27 | 27 | 54 | 0 | 1 | 79 |
| 220 | 896 | [xla_npu/service/transform/npu_entry_param_copy_insertion.h](../xla_npu/service/transform/npu_entry_param_copy_insertion.h) | 0 | 53 | 0 | 53 | 53 | 1 | 53 |
| 221 | 897 | [xla_npu/service/transform/npu_scatter_copy_insertion.h](../xla_npu/service/transform/npu_scatter_copy_insertion.h) | 0 | 53 | 0 | 53 | 53 | 1 | 53 |
| 222 | 384 | [xla_npu/dependency/header/acl_nn_op/acl_stft.h](../xla_npu/dependency/header/acl_nn_op/acl_stft.h) | 81 | 26 | 26 | 52 | 0 | 1 | 81 |
| 223 | 404 | [xla_npu/dependency/header/acl_nn_op/aclnn_all_gather_matmul_v2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_all_gather_matmul_v2.h) | 76 | 26 | 26 | 52 | 0 | 1 | 76 |
| 224 | 420 | [xla_npu/dependency/header/acl_nn_op/aclnn_bincount.h](../xla_npu/dependency/header/acl_nn_op/aclnn_bincount.h) | 74 | 26 | 26 | 52 | 0 | 1 | 74 |
| 225 | 347 | [xla_npu/dependency/header/acl_nn_op/aclnn_fake_quant_per_tensor_affine_cachemask.h](../xla_npu/dependency/header/acl_nn_op/aclnn_fake_quant_per_tensor_affine_cachemask.h) | 92 | 26 | 26 | 52 | 0 | 1 | 92 |
| 226 | 423 | [xla_npu/dependency/header/acl_nn_op/aclnn_grouped_matmul_finalize_routing.h](../xla_npu/dependency/header/acl_nn_op/aclnn_grouped_matmul_finalize_routing.h) | 74 | 26 | 26 | 52 | 0 | 1 | 74 |
| 227 | 431 | [xla_npu/dependency/header/acl_nn_op/aclnn_grouped_matmul_finalize_routing_weight_nz.h](../xla_npu/dependency/header/acl_nn_op/aclnn_grouped_matmul_finalize_routing_weight_nz.h) | 73 | 26 | 26 | 52 | 0 | 1 | 73 |
| 228 | 424 | [xla_npu/dependency/header/acl_nn_op/aclnn_grouped_matmul_swiglu_quant_v2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_grouped_matmul_swiglu_quant_v2.h) | 74 | 26 | 26 | 52 | 0 | 1 | 74 |
| 229 | 407 | [xla_npu/dependency/header/acl_nn_op/aclnn_inplace_weight_quant_matmul_all_reduce_add_rms_norm.h](../xla_npu/dependency/header/acl_nn_op/aclnn_inplace_weight_quant_matmul_all_reduce_add_rms_norm.h) | 76 | 26 | 26 | 52 | 0 | 1 | 76 |
| 230 | 413 | [xla_npu/dependency/header/acl_nn_op/aclnn_instance_norm.h](../xla_npu/dependency/header/acl_nn_op/aclnn_instance_norm.h) | 75 | 26 | 26 | 52 | 0 | 1 | 75 |
| 231 | 396 | [xla_npu/dependency/header/acl_nn_op/aclnn_matmul_reduce_scatter_v2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_matmul_reduce_scatter_v2.h) | 78 | 26 | 26 | 52 | 0 | 1 | 78 |
| 232 | 372 | [xla_npu/dependency/header/acl_nn_op/aclnn_multinomial.h](../xla_npu/dependency/header/acl_nn_op/aclnn_multinomial.h) | 87 | 26 | 26 | 52 | 0 | 1 | 87 |
| 233 | 437 | [xla_npu/dependency/header/acl_nn_op/aclnn_nll_loss2d_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_nll_loss2d_backward.h) | 72 | 26 | 26 | 52 | 0 | 1 | 72 |
| 234 | 397 | [xla_npu/dependency/header/acl_nn_op/aclnn_quant_matmul_reduce_sum.h](../xla_npu/dependency/header/acl_nn_op/aclnn_quant_matmul_reduce_sum.h) | 78 | 26 | 26 | 52 | 0 | 1 | 78 |
| 235 | 333 | [xla_npu/dependency/header/acl_nn_op/aclnn_range.h](../xla_npu/dependency/header/acl_nn_op/aclnn_range.h) | 96 | 26 | 26 | 52 | 0 | 1 | 96 |
| 236 | 119 | [xla_npu/dependency/header/metadef/exe_graph/runtime/continuous_vector.h](../xla_npu/dependency/header/metadef/exe_graph/runtime/continuous_vector.h) | 223 | 26 | 26 | 52 | 0 | 1 | 223 |
| 237 | 260 | [xla_npu/dependency/header/metadef/exe_graph/runtime/storage_format.h](../xla_npu/dependency/header/metadef/exe_graph/runtime/storage_format.h) | 119 | 26 | 26 | 52 | 0 | 1 | 119 |
| 238 | 102 | [xla_npu/dependency/header/metadef/graph/tensor.h](../xla_npu/dependency/header/metadef/graph/tensor.h) | 246 | 26 | 26 | 52 | 0 | 1 | 246 |
| 239 | 429 | [xla_npu/dependency/header/acl_nn_op/aclnn_avgpool3d_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_avgpool3d_backward.h) | 75 | 25 | 25 | 50 | 0 | 1 | 75 |
| 240 | 346 | [xla_npu/dependency/header/acl_nn_op/aclnn_cumsum.h](../xla_npu/dependency/header/acl_nn_op/aclnn_cumsum.h) | 94 | 25 | 25 | 50 | 0 | 1 | 94 |
| 241 | 324 | [xla_npu/dependency/header/acl_nn_op/aclnn_grid_sampler2d_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_grid_sampler2d_backward.h) | 101 | 25 | 25 | 50 | 0 | 1 | 101 |
| 242 | 425 | [xla_npu/dependency/header/acl_nn_op/aclnn_grouped_matmul_swiglu_quant_weight_nz_v2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_grouped_matmul_swiglu_quant_weight_nz_v2.h) | 76 | 25 | 25 | 50 | 0 | 1 | 76 |
| 243 | 426 | [xla_npu/dependency/header/acl_nn_op/aclnn_quant_matmul_all_reduce_add_rms_norm.h](../xla_npu/dependency/header/acl_nn_op/aclnn_quant_matmul_all_reduce_add_rms_norm.h) | 76 | 25 | 25 | 50 | 0 | 1 | 76 |
| 244 | 409 | [xla_npu/dependency/header/acl_nn_op/aclnn_quant_matmul_all_reduce_v3.h](../xla_npu/dependency/header/acl_nn_op/aclnn_quant_matmul_all_reduce_v3.h) | 78 | 25 | 25 | 50 | 0 | 1 | 78 |
| 245 | 468 | [xla_npu/dependency/header/acl_nn_op/aclnn_silent_check.h](../xla_npu/dependency/header/acl_nn_op/aclnn_silent_check.h) | 67 | 25 | 25 | 50 | 0 | 1 | 67 |
| 246 | 427 | [xla_npu/dependency/header/acl_nn_op/aclnn_trans_convolution_weight.h](../xla_npu/dependency/header/acl_nn_op/aclnn_trans_convolution_weight.h) | 76 | 25 | 25 | 50 | 0 | 1 | 76 |
| 247 | 391 | [xla_npu/dependency/header/acl_nn_op/aclnn_trans_matmul_weight.h](../xla_npu/dependency/header/acl_nn_op/aclnn_trans_matmul_weight.h) | 82 | 25 | 25 | 50 | 0 | 1 | 82 |
| 248 | 387 | [xla_npu/dependency/header/acl_nn_op/aclnn_unique_consecutive.h](../xla_npu/dependency/header/acl_nn_op/aclnn_unique_consecutive.h) | 83 | 25 | 25 | 50 | 0 | 1 | 83 |
| 249 | 305 | [xla_npu/dependency/header/metadef/exe_graph/runtime/infer_shape_range_context.h](../xla_npu/dependency/header/metadef/exe_graph/runtime/infer_shape_range_context.h) | 106 | 25 | 25 | 50 | 0 | 1 | 106 |
| 250 | 944 | [xla_npu/service/transform/npu_while_body_copy_insertion.h](../xla_npu/service/transform/npu_while_body_copy_insertion.h) | 0 | 47 | 2 | 49 | 45 | 2 | 45 |
| 251 | 190 | [xla_npu/dependency/header/acl_nn_op/aclnn_div.h](../xla_npu/dependency/header/acl_nn_op/aclnn_div.h) | 153 | 24 | 24 | 48 | 0 | 1 | 153 |
| 252 | 332 | [xla_npu/dependency/header/acl_nn_op/aclnn_dropout_gen_mask.h](../xla_npu/dependency/header/acl_nn_op/aclnn_dropout_gen_mask.h) | 100 | 24 | 24 | 48 | 0 | 1 | 100 |
| 253 | 421 | [xla_npu/dependency/header/acl_nn_op/aclnn_gather_v2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_gather_v2.h) | 78 | 24 | 24 | 48 | 0 | 1 | 78 |
| 254 | 362 | [xla_npu/dependency/header/acl_nn_op/aclnn_grid_sampler2d.h](../xla_npu/dependency/header/acl_nn_op/aclnn_grid_sampler2d.h) | 93 | 24 | 24 | 48 | 0 | 1 | 93 |
| 255 | 312 | [xla_npu/dependency/header/acl_nn_op/aclnn_hardswish_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_hardswish_backward.h) | 105 | 24 | 24 | 48 | 0 | 1 | 105 |
| 256 | 309 | [xla_npu/dependency/header/acl_nn_op/aclnn_hardswish_backward_v2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_hardswish_backward_v2.h) | 106 | 24 | 24 | 48 | 0 | 1 | 106 |
| 257 | 444 | [xla_npu/dependency/header/acl_nn_op/aclnn_inplace_quant_matmul_all_reduce_add_rms_norm.h](../xla_npu/dependency/header/acl_nn_op/aclnn_inplace_quant_matmul_all_reduce_add_rms_norm.h) | 74 | 24 | 24 | 48 | 0 | 1 | 74 |
| 258 | 445 | [xla_npu/dependency/header/acl_nn_op/aclnn_matmul_all_reduce_add_rms_norm.h](../xla_npu/dependency/header/acl_nn_op/aclnn_matmul_all_reduce_add_rms_norm.h) | 74 | 24 | 24 | 48 | 0 | 1 | 74 |
| 259 | 467 | [xla_npu/dependency/header/acl_nn_op/aclnn_obfuscation_setup_v2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_obfuscation_setup_v2.h) | 69 | 24 | 24 | 48 | 0 | 1 | 69 |
| 260 | 487 | [xla_npu/dependency/header/acl_nn_op/aclnn_quant_grouped_matmul_inplace_add.h](../xla_npu/dependency/header/acl_nn_op/aclnn_quant_grouped_matmul_inplace_add.h) | 67 | 24 | 24 | 48 | 0 | 1 | 67 |
| 261 | 475 | [xla_npu/dependency/header/acl_nn_op/aclnn_quant_matmul_v5.h](../xla_npu/dependency/header/acl_nn_op/aclnn_quant_matmul_v5.h) | 68 | 24 | 24 | 48 | 0 | 1 | 68 |
| 262 | 497 | [xla_npu/dependency/header/acl_nn_op/aclnn_quant_matmul_weight_nz.h](../xla_npu/dependency/header/acl_nn_op/aclnn_quant_matmul_weight_nz.h) | 66 | 24 | 24 | 48 | 0 | 1 | 66 |
| 263 | 398 | [xla_npu/dependency/header/acl_nn_op/aclnn_unique2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_unique2.h) | 82 | 24 | 24 | 48 | 0 | 1 | 82 |
| 264 | 446 | [xla_npu/dependency/header/acl_nn_op/aclnn_weight_quant_matmul_all_reduce.h](../xla_npu/dependency/header/acl_nn_op/aclnn_weight_quant_matmul_all_reduce.h) | 74 | 24 | 24 | 48 | 0 | 1 | 74 |
| 265 | 245 | [xla_npu/dependency/header/metadef/exe_graph/runtime/infer_shape_context.h](../xla_npu/dependency/header/metadef/exe_graph/runtime/infer_shape_context.h) | 127 | 24 | 24 | 48 | 0 | 1 | 127 |
| 266 | 38 | [xla_npu/pjrt/air/mair/translations/export_graphdef.cc](../xla_npu/pjrt/air/mair/translations/export_graphdef.cc) | 561 | 24 | 24 | 48 | 0 | 1 | 561 |
| 267 | 85 | [xla_npu/service/convert_to_afir/convert_to_afir_test.cc](../xla_npu/service/convert_to_afir/convert_to_afir_test.cc) | 292 | 48 | 0 | 48 | 48 | 1 | 340 |
| 268 | 494 | [xla_npu/dependency/header/acl_nn_op/aclnn_adaptive_avg_pool3d_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_adaptive_avg_pool3d_backward.h) | 68 | 23 | 23 | 46 | 0 | 1 | 68 |
| 269 | 463 | [xla_npu/dependency/header/acl_nn_op/aclnn_avgpool3d.h](../xla_npu/dependency/header/acl_nn_op/aclnn_avgpool3d.h) | 71 | 23 | 23 | 46 | 0 | 1 | 71 |
| 270 | 449 | [xla_npu/dependency/header/acl_nn_op/aclnn_dynamic_quant_v3.h](../xla_npu/dependency/header/acl_nn_op/aclnn_dynamic_quant_v3.h) | 74 | 23 | 23 | 46 | 0 | 1 | 74 |
| 271 | 405 | [xla_npu/dependency/header/acl_nn_op/aclnn_fake_quant_per_channel_affine_cachemask.h](../xla_npu/dependency/header/acl_nn_op/aclnn_fake_quant_per_channel_affine_cachemask.h) | 82 | 23 | 23 | 46 | 0 | 1 | 82 |
| 272 | 505 | [xla_npu/dependency/header/acl_nn_op/aclnn_flat_quant.h](../xla_npu/dependency/header/acl_nn_op/aclnn_flat_quant.h) | 66 | 23 | 23 | 46 | 0 | 1 | 66 |
| 273 | 472 | [xla_npu/dependency/header/acl_nn_op/aclnn_inplace_matmul_all_reduce_add_rms_norm.h](../xla_npu/dependency/header/acl_nn_op/aclnn_inplace_matmul_all_reduce_add_rms_norm.h) | 70 | 23 | 23 | 46 | 0 | 1 | 70 |
| 274 | 466 | [xla_npu/dependency/header/acl_nn_op/aclnn_isclose.h](../xla_npu/dependency/header/acl_nn_op/aclnn_isclose.h) | 71 | 23 | 23 | 46 | 0 | 1 | 71 |
| 275 | 511 | [xla_npu/dependency/header/acl_nn_op/aclnn_nll_loss.h](../xla_npu/dependency/header/acl_nn_op/aclnn_nll_loss.h) | 66 | 23 | 23 | 46 | 0 | 1 | 66 |
| 276 | 473 | [xla_npu/dependency/header/acl_nn_op/aclnn_nll_loss_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_nll_loss_backward.h) | 70 | 23 | 23 | 46 | 0 | 1 | 70 |
| 277 | 496 | [xla_npu/dependency/header/acl_nn_op/aclnn_obfuscation_setup.h](../xla_npu/dependency/header/acl_nn_op/aclnn_obfuscation_setup.h) | 68 | 23 | 23 | 46 | 0 | 1 | 68 |
| 278 | 453 | [xla_npu/dependency/header/acl_nn_op/aclnn_quant_matmul_all_reduce_v2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_quant_matmul_all_reduce_v2.h) | 73 | 23 | 23 | 46 | 0 | 1 | 73 |
| 279 | 526 | [xla_npu/dependency/header/acl_nn_op/aclnn_reflection_pad3d_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_reflection_pad3d_backward.h) | 65 | 23 | 23 | 46 | 0 | 1 | 65 |
| 280 | 489 | [xla_npu/dependency/header/acl_nn_op/aclnn_scatter_add.h](../xla_npu/dependency/header/acl_nn_op/aclnn_scatter_add.h) | 69 | 23 | 23 | 46 | 0 | 1 | 69 |
| 281 | 477 | [xla_npu/dependency/header/acl_nn_op/aclnn_soft_margin_loss_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_soft_margin_loss_backward.h) | 70 | 23 | 23 | 46 | 0 | 1 | 70 |
| 282 | 527 | [xla_npu/dependency/header/acl_nn_op/aclnn_triangular_solve.h](../xla_npu/dependency/header/acl_nn_op/aclnn_triangular_solve.h) | 65 | 23 | 23 | 46 | 0 | 1 | 65 |
| 283 | 1035 | [xla_npu/service/transform/npu_tril_rewriter.h](../xla_npu/service/transform/npu_tril_rewriter.h) | 0 | 43 | 2 | 45 | 41 | 2 | 41 |
| 284 | 138 | [xla_npu/dependency/external/afir/Ascend-MLIR/lib/Dialect/AFIR/Ops/Basic/cast.cpp](../xla_npu/dependency/external/afir/Ascend-MLIR/lib/Dialect/AFIR/Ops/Basic/cast.cpp) | 207 | 22 | 22 | 44 | 0 | 1 | 207 |
| 285 | 567 | [xla_npu/dependency/header/acl_nn_op/aclnn_ascend_quant_v3.h](../xla_npu/dependency/header/acl_nn_op/aclnn_ascend_quant_v3.h) | 62 | 22 | 22 | 44 | 0 | 1 | 62 |
| 286 | 482 | [xla_npu/dependency/header/acl_nn_op/aclnn_embedding_bag.h](../xla_npu/dependency/header/acl_nn_op/aclnn_embedding_bag.h) | 71 | 22 | 22 | 44 | 0 | 1 | 71 |
| 287 | 339 | [xla_npu/dependency/header/acl_nn_op/aclnn_embedding_dense_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_embedding_dense_backward.h) | 101 | 22 | 22 | 44 | 0 | 1 | 101 |
| 288 | 521 | [xla_npu/dependency/header/acl_nn_op/aclnn_gather.h](../xla_npu/dependency/header/acl_nn_op/aclnn_gather.h) | 67 | 22 | 22 | 44 | 0 | 1 | 67 |
| 289 | 547 | [xla_npu/dependency/header/acl_nn_op/aclnn_gelu_quant.h](../xla_npu/dependency/header/acl_nn_op/aclnn_gelu_quant.h) | 64 | 22 | 22 | 44 | 0 | 1 | 64 |
| 290 | 422 | [xla_npu/dependency/header/acl_nn_op/aclnn_glu.h](../xla_npu/dependency/header/acl_nn_op/aclnn_glu.h) | 82 | 22 | 22 | 44 | 0 | 1 | 82 |
| 291 | 430 | [xla_npu/dependency/header/acl_nn_op/aclnn_glu_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_glu_backward.h) | 81 | 22 | 22 | 44 | 0 | 1 | 81 |
| 292 | 458 | [xla_npu/dependency/header/acl_nn_op/aclnn_grouped_mat_mul_all_reduce.h](../xla_npu/dependency/header/acl_nn_op/aclnn_grouped_mat_mul_all_reduce.h) | 74 | 22 | 22 | 44 | 0 | 1 | 74 |
| 293 | 465 | [xla_npu/dependency/header/acl_nn_op/aclnn_index_select.h](../xla_npu/dependency/header/acl_nn_op/aclnn_index_select.h) | 73 | 22 | 22 | 44 | 0 | 1 | 73 |
| 294 | 536 | [xla_npu/dependency/header/acl_nn_op/aclnn_l1_loss_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_l1_loss_backward.h) | 66 | 22 | 22 | 44 | 0 | 1 | 66 |
| 295 | 414 | [xla_npu/dependency/header/acl_nn_op/aclnn_logical_xor.h](../xla_npu/dependency/header/acl_nn_op/aclnn_logical_xor.h) | 83 | 22 | 22 | 44 | 0 | 1 | 83 |
| 296 | 510 | [xla_npu/dependency/header/acl_nn_op/aclnn_moe_update_expert.h](../xla_npu/dependency/header/acl_nn_op/aclnn_moe_update_expert.h) | 68 | 22 | 22 | 44 | 0 | 1 | 68 |
| 297 | 524 | [xla_npu/dependency/header/acl_nn_op/aclnn_mse_loss_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_mse_loss_backward.h) | 67 | 22 | 22 | 44 | 0 | 1 | 67 |
| 298 | 474 | [xla_npu/dependency/header/acl_nn_op/aclnn_quant_grouped_matmul_dequant.h](../xla_npu/dependency/header/acl_nn_op/aclnn_quant_grouped_matmul_dequant.h) | 72 | 22 | 22 | 44 | 0 | 1 | 72 |
| 299 | 488 | [xla_npu/dependency/header/acl_nn_op/aclnn_quant_matmul_all_reduce.h](../xla_npu/dependency/header/acl_nn_op/aclnn_quant_matmul_all_reduce.h) | 71 | 22 | 22 | 44 | 0 | 1 | 71 |
| 300 | 476 | [xla_npu/dependency/header/acl_nn_op/aclnn_rsqrt.h](../xla_npu/dependency/header/acl_nn_op/aclnn_rsqrt.h) | 72 | 22 | 22 | 44 | 0 | 1 | 72 |
| 301 | 543 | [xla_npu/dependency/header/acl_nn_op/aclnn_scatter_update.h](../xla_npu/dependency/header/acl_nn_op/aclnn_scatter_update.h) | 65 | 22 | 22 | 44 | 0 | 1 | 65 |
| 302 | 450 | [xla_npu/dependency/header/acl_nn_op/aclnn_topk.h](../xla_npu/dependency/header/acl_nn_op/aclnn_topk.h) | 76 | 22 | 22 | 44 | 0 | 1 | 76 |
| 303 | 438 | [xla_npu/dependency/header/acl_nn_op/aclnn_unique_dim.h](../xla_npu/dependency/header/acl_nn_op/aclnn_unique_dim.h) | 80 | 22 | 22 | 44 | 0 | 1 | 80 |
| 304 | 546 | [xla_npu/dependency/header/acl_nn_op/aclnn_adaptive_avg_pool2d_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_adaptive_avg_pool2d_backward.h) | 66 | 21 | 21 | 42 | 0 | 1 | 66 |
| 305 | 401 | [xla_npu/dependency/header/acl_nn_op/aclnn_addmm.h](../xla_npu/dependency/header/acl_nn_op/aclnn_addmm.h) | 87 | 21 | 21 | 42 | 0 | 1 | 87 |
| 306 | 481 | [xla_npu/dependency/header/acl_nn_op/aclnn_addr.h](../xla_npu/dependency/header/acl_nn_op/aclnn_addr.h) | 73 | 21 | 21 | 42 | 0 | 1 | 73 |
| 307 | 410 | [xla_npu/dependency/header/acl_nn_op/aclnn_argmin.h](../xla_npu/dependency/header/acl_nn_op/aclnn_argmin.h) | 85 | 21 | 21 | 42 | 0 | 1 | 85 |
| 308 | 602 | [xla_npu/dependency/header/acl_nn_op/aclnn_ascend_quant.h](../xla_npu/dependency/header/acl_nn_op/aclnn_ascend_quant.h) | 60 | 21 | 21 | 42 | 0 | 1 | 60 |
| 309 | 411 | [xla_npu/dependency/header/acl_nn_op/aclnn_bitwise_not.h](../xla_npu/dependency/header/acl_nn_op/aclnn_bitwise_not.h) | 85 | 21 | 21 | 42 | 0 | 1 | 85 |
| 310 | 585 | [xla_npu/dependency/header/acl_nn_op/aclnn_circular_pad2d_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_circular_pad2d_backward.h) | 62 | 21 | 21 | 42 | 0 | 1 | 62 |
| 311 | 595 | [xla_npu/dependency/header/acl_nn_op/aclnn_circular_pad3d_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_circular_pad3d_backward.h) | 61 | 21 | 21 | 42 | 0 | 1 | 61 |
| 312 | 379 | [xla_npu/dependency/header/acl_nn_op/aclnn_dot.h](../xla_npu/dependency/header/acl_nn_op/aclnn_dot.h) | 93 | 21 | 21 | 42 | 0 | 1 | 93 |
| 313 | 457 | [xla_npu/dependency/header/acl_nn_op/aclnn_dropout_v3.h](../xla_npu/dependency/header/acl_nn_op/aclnn_dropout_v3.h) | 76 | 21 | 21 | 42 | 0 | 1 | 76 |
| 314 | 443 | [xla_npu/dependency/header/acl_nn_op/aclnn_flip.h](../xla_npu/dependency/header/acl_nn_op/aclnn_flip.h) | 80 | 21 | 21 | 42 | 0 | 1 | 80 |
| 315 | 470 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_addcdiv_scalar_v2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_addcdiv_scalar_v2.h) | 74 | 21 | 21 | 42 | 0 | 1 | 74 |
| 316 | 471 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_addcmul_scalar_v2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_addcmul_scalar_v2.h) | 74 | 21 | 21 | 42 | 0 | 1 | 74 |
| 317 | 483 | [xla_npu/dependency/header/acl_nn_op/aclnn_fused_cross_entropy_loss_with_max_sum.h](../xla_npu/dependency/header/acl_nn_op/aclnn_fused_cross_entropy_loss_with_max_sum.h) | 73 | 21 | 21 | 42 | 0 | 1 | 73 |
| 318 | 520 | [xla_npu/dependency/header/acl_nn_op/aclnn_fused_linear_cross_entropy_loss_grad.h](../xla_npu/dependency/header/acl_nn_op/aclnn_fused_linear_cross_entropy_loss_grad.h) | 69 | 21 | 21 | 42 | 0 | 1 | 69 |
| 319 | 506 | [xla_npu/dependency/header/acl_nn_op/aclnn_fused_linear_online_max_sum.h](../xla_npu/dependency/header/acl_nn_op/aclnn_fused_linear_online_max_sum.h) | 70 | 21 | 21 | 42 | 0 | 1 | 70 |
| 320 | 406 | [xla_npu/dependency/header/acl_nn_op/aclnn_gelu_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_gelu_backward.h) | 86 | 21 | 21 | 42 | 0 | 1 | 86 |
| 321 | 522 | [xla_npu/dependency/header/acl_nn_op/aclnn_grouped_matmul_v2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_grouped_matmul_v2.h) | 69 | 21 | 21 | 42 | 0 | 1 | 69 |
| 322 | 523 | [xla_npu/dependency/header/acl_nn_op/aclnn_grouped_matmul_v3.h](../xla_npu/dependency/header/acl_nn_op/aclnn_grouped_matmul_v3.h) | 69 | 21 | 21 | 42 | 0 | 1 | 69 |
| 323 | 507 | [xla_npu/dependency/header/acl_nn_op/aclnn_heaviside.h](../xla_npu/dependency/header/acl_nn_op/aclnn_heaviside.h) | 70 | 21 | 21 | 42 | 0 | 1 | 70 |
| 324 | 408 | [xla_npu/dependency/header/acl_nn_op/aclnn_inverse.h](../xla_npu/dependency/header/acl_nn_op/aclnn_inverse.h) | 86 | 21 | 21 | 42 | 0 | 1 | 86 |
| 325 | 535 | [xla_npu/dependency/header/acl_nn_op/aclnn_kthvalue.h](../xla_npu/dependency/header/acl_nn_op/aclnn_kthvalue.h) | 68 | 21 | 21 | 42 | 0 | 1 | 68 |
| 326 | 559 | [xla_npu/dependency/header/acl_nn_op/aclnn_l1_loss.h](../xla_npu/dependency/header/acl_nn_op/aclnn_l1_loss.h) | 65 | 21 | 21 | 42 | 0 | 1 | 65 |
| 327 | 537 | [xla_npu/dependency/header/acl_nn_op/aclnn_linalg_qr.h](../xla_npu/dependency/header/acl_nn_op/aclnn_linalg_qr.h) | 68 | 21 | 21 | 42 | 0 | 1 | 68 |
| 328 | 368 | [xla_npu/dependency/header/acl_nn_op/aclnn_masked_fill_scalar.h](../xla_npu/dependency/header/acl_nn_op/aclnn_masked_fill_scalar.h) | 98 | 21 | 21 | 42 | 0 | 1 | 98 |
| 329 | 364 | [xla_npu/dependency/header/acl_nn_op/aclnn_masked_fill_tensor.h](../xla_npu/dependency/header/acl_nn_op/aclnn_masked_fill_tensor.h) | 99 | 21 | 21 | 42 | 0 | 1 | 99 |
| 330 | 570 | [xla_npu/dependency/header/acl_nn_op/aclnn_masked_scale.h](../xla_npu/dependency/header/acl_nn_op/aclnn_masked_scale.h) | 64 | 21 | 21 | 42 | 0 | 1 | 64 |
| 331 | 385 | [xla_npu/dependency/header/acl_nn_op/aclnn_masked_scatter.h](../xla_npu/dependency/header/acl_nn_op/aclnn_masked_scatter.h) | 91 | 21 | 21 | 42 | 0 | 1 | 91 |
| 332 | 560 | [xla_npu/dependency/header/acl_nn_op/aclnn_matmul_all_reduce_v2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_matmul_all_reduce_v2.h) | 65 | 21 | 21 | 42 | 0 | 1 | 65 |
| 333 | 604 | [xla_npu/dependency/header/acl_nn_op/aclnn_moe_token_permute_with_routing_map.h](../xla_npu/dependency/header/acl_nn_op/aclnn_moe_token_permute_with_routing_map.h) | 60 | 21 | 21 | 42 | 0 | 1 | 60 |
| 334 | 589 | [xla_npu/dependency/header/acl_nn_op/aclnn_mv.h](../xla_npu/dependency/header/acl_nn_op/aclnn_mv.h) | 62 | 21 | 21 | 42 | 0 | 1 | 62 |
| 335 | 486 | [xla_npu/dependency/header/acl_nn_op/aclnn_npu_format_cast.h](../xla_npu/dependency/header/acl_nn_op/aclnn_npu_format_cast.h) | 73 | 21 | 21 | 42 | 0 | 1 | 73 |
| 336 | 550 | [xla_npu/dependency/header/acl_nn_op/aclnn_qr.h](../xla_npu/dependency/header/acl_nn_op/aclnn_qr.h) | 66 | 21 | 21 | 42 | 0 | 1 | 66 |
| 337 | 512 | [xla_npu/dependency/header/acl_nn_op/aclnn_quant_matmul_dequant.h](../xla_npu/dependency/header/acl_nn_op/aclnn_quant_matmul_dequant.h) | 70 | 21 | 21 | 42 | 0 | 1 | 70 |
| 338 | 582 | [xla_npu/dependency/header/acl_nn_op/aclnn_replication_pad3d_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_replication_pad3d_backward.h) | 63 | 21 | 21 | 42 | 0 | 1 | 63 |
| 339 | 607 | [xla_npu/dependency/header/acl_nn_op/aclnn_scale.h](../xla_npu/dependency/header/acl_nn_op/aclnn_scale.h) | 60 | 21 | 21 | 42 | 0 | 1 | 60 |
| 340 | 583 | [xla_npu/dependency/header/acl_nn_op/aclnn_smooth_l1_loss_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_smooth_l1_loss_backward.h) | 63 | 21 | 21 | 42 | 0 | 1 | 63 |
| 341 | 574 | [xla_npu/dependency/header/acl_nn_op/aclnn_sort.h](../xla_npu/dependency/header/acl_nn_op/aclnn_sort.h) | 64 | 21 | 21 | 42 | 0 | 1 | 64 |
| 342 | 352 | [xla_npu/dependency/header/acl_nn_op/aclnn_std.h](../xla_npu/dependency/header/acl_nn_op/aclnn_std.h) | 102 | 21 | 21 | 42 | 0 | 1 | 102 |
| 343 | 478 | [xla_npu/dependency/header/acl_nn_op/aclnn_threshold_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_threshold_backward.h) | 74 | 21 | 21 | 42 | 0 | 1 | 74 |
| 344 | 469 | [xla_npu/dependency/header/acl_nn_op/aclnn_unique.h](../xla_npu/dependency/header/acl_nn_op/aclnn_unique.h) | 75 | 21 | 21 | 42 | 0 | 1 | 75 |
| 345 | 552 | [xla_npu/dependency/header/acl_nn_op/aclnn_upsample_bicubic2d_aa.h](../xla_npu/dependency/header/acl_nn_op/aclnn_upsample_bicubic2d_aa.h) | 66 | 21 | 21 | 42 | 0 | 1 | 66 |
| 346 | 502 | [xla_npu/dependency/header/acl_nn_op/aclnn_upsample_bicubic2d_aa_grad.h](../xla_npu/dependency/header/acl_nn_op/aclnn_upsample_bicubic2d_aa_grad.h) | 71 | 21 | 21 | 42 | 0 | 1 | 71 |
| 347 | 513 | [xla_npu/dependency/header/acl_nn_op/aclnn_upsample_bilinear2d_aa_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_upsample_bilinear2d_aa_backward.h) | 70 | 21 | 21 | 42 | 0 | 1 | 70 |
| 348 | 514 | [xla_npu/dependency/header/acl_nn_op/aclnn_upsample_bilinear_2d_backward_v2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_upsample_bilinear_2d_backward_v2.h) | 70 | 21 | 21 | 42 | 0 | 1 | 70 |
| 349 | 528 | [xla_npu/dependency/header/acl_nn_op/aclnn_upsample_nearest_2d_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_upsample_nearest_2d_backward.h) | 69 | 21 | 21 | 42 | 0 | 1 | 69 |
| 350 | 529 | [xla_npu/dependency/header/acl_nn_op/aclnn_upsample_nearest_exact2d_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_upsample_nearest_exact2d_backward.h) | 69 | 21 | 21 | 42 | 0 | 1 | 69 |
| 351 | 1084 | [xla_npu/service/transform/npu_dot_fp32_left_order_rewriter.h](../xla_npu/service/transform/npu_dot_fp32_left_order_rewriter.h) | 0 | 42 | 0 | 42 | 42 | 1 | 42 |
| 352 | 1085 | [xla_npu/service/transform/npu_scalar_fma_compensation.h](../xla_npu/service/transform/npu_scalar_fma_compensation.h) | 0 | 42 | 0 | 42 | 42 | 1 | 42 |
| 353 | 580 | [xla_npu/dependency/header/acl_nn_op/aclnn_all_gather_matmul.h](../xla_npu/dependency/header/acl_nn_op/aclnn_all_gather_matmul.h) | 65 | 20 | 20 | 40 | 0 | 1 | 65 |
| 354 | 436 | [xla_npu/dependency/header/acl_nn_op/aclnn_argmax.h](../xla_npu/dependency/header/acl_nn_op/aclnn_argmax.h) | 84 | 20 | 20 | 40 | 0 | 1 | 84 |
| 355 | 627 | [xla_npu/dependency/header/acl_nn_op/aclnn_ascend_anti_quant.h](../xla_npu/dependency/header/acl_nn_op/aclnn_ascend_anti_quant.h) | 60 | 20 | 20 | 40 | 0 | 1 | 60 |
| 356 | 464 | [xla_npu/dependency/header/acl_nn_op/aclnn_batch_matmul.h](../xla_npu/dependency/header/acl_nn_op/aclnn_batch_matmul.h) | 77 | 20 | 20 | 40 | 0 | 1 | 77 |
| 357 | 611 | [xla_npu/dependency/header/acl_nn_op/aclnn_circular_pad3d.h](../xla_npu/dependency/header/acl_nn_op/aclnn_circular_pad3d.h) | 61 | 20 | 20 | 40 | 0 | 1 | 61 |
| 358 | 555 | [xla_npu/dependency/header/acl_nn_op/aclnn_cummax.h](../xla_npu/dependency/header/acl_nn_op/aclnn_cummax.h) | 67 | 20 | 20 | 40 | 0 | 1 | 67 |
| 359 | 556 | [xla_npu/dependency/header/acl_nn_op/aclnn_cummin.h](../xla_npu/dependency/header/acl_nn_op/aclnn_cummin.h) | 67 | 20 | 20 | 40 | 0 | 1 | 67 |
| 360 | 541 | [xla_npu/dependency/header/acl_nn_op/aclnn_fill_tensor.h](../xla_npu/dependency/header/acl_nn_op/aclnn_fill_tensor.h) | 69 | 20 | 20 | 40 | 0 | 1 | 69 |
| 361 | 518 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_add_list_v2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_add_list_v2.h) | 71 | 20 | 20 | 40 | 0 | 1 | 71 |
| 362 | 519 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_sub_list_v2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_sub_list_v2.h) | 71 | 20 | 20 | 40 | 0 | 1 | 71 |
| 363 | 548 | [xla_npu/dependency/header/acl_nn_op/aclnn_gelu_v2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_gelu_v2.h) | 68 | 20 | 20 | 40 | 0 | 1 | 68 |
| 364 | 633 | [xla_npu/dependency/header/acl_nn_op/aclnn_group_quant.h](../xla_npu/dependency/header/acl_nn_op/aclnn_group_quant.h) | 59 | 20 | 20 | 40 | 0 | 1 | 59 |
| 365 | 568 | [xla_npu/dependency/header/acl_nn_op/aclnn_grouped_dynamic_mx_quant.h](../xla_npu/dependency/header/acl_nn_op/aclnn_grouped_dynamic_mx_quant.h) | 66 | 20 | 20 | 40 | 0 | 1 | 66 |
| 366 | 558 | [xla_npu/dependency/header/acl_nn_op/aclnn_grouped_matmul.h](../xla_npu/dependency/header/acl_nn_op/aclnn_grouped_matmul.h) | 67 | 20 | 20 | 40 | 0 | 1 | 67 |
| 367 | 588 | [xla_npu/dependency/header/acl_nn_op/aclnn_grouped_matmul_add_v2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_grouped_matmul_add_v2.h) | 64 | 20 | 20 | 40 | 0 | 1 | 64 |
| 368 | 412 | [xla_npu/dependency/header/acl_nn_op/aclnn_hardshrink.h](../xla_npu/dependency/header/acl_nn_op/aclnn_hardshrink.h) | 87 | 20 | 20 | 40 | 0 | 1 | 87 |
| 369 | 538 | [xla_npu/dependency/header/acl_nn_op/aclnn_linspace.h](../xla_npu/dependency/header/acl_nn_op/aclnn_linspace.h) | 70 | 20 | 20 | 40 | 0 | 1 | 70 |
| 370 | 549 | [xla_npu/dependency/header/acl_nn_op/aclnn_moe_token_permute_with_routing_map_grad.h](../xla_npu/dependency/header/acl_nn_op/aclnn_moe_token_permute_with_routing_map_grad.h) | 68 | 20 | 20 | 40 | 0 | 1 | 68 |
| 371 | 360 | [xla_npu/dependency/header/acl_nn_op/aclnn_normal_out.h](../xla_npu/dependency/header/acl_nn_op/aclnn_normal_out.h) | 102 | 20 | 20 | 40 | 0 | 1 | 102 |
| 372 | 525 | [xla_npu/dependency/header/acl_nn_op/aclnn_pow_tensor_tensor.h](../xla_npu/dependency/header/acl_nn_op/aclnn_pow_tensor_tensor.h) | 71 | 20 | 20 | 40 | 0 | 1 | 71 |
| 373 | 630 | [xla_npu/dependency/header/acl_nn_op/aclnn_quant_matmul_v4.h](../xla_npu/dependency/header/acl_nn_op/aclnn_quant_matmul_v4.h) | 60 | 20 | 20 | 40 | 0 | 1 | 60 |
| 374 | 590 | [xla_npu/dependency/header/acl_nn_op/aclnn_reflection_pad1d_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_reflection_pad1d_backward.h) | 64 | 20 | 20 | 40 | 0 | 1 | 64 |
| 375 | 591 | [xla_npu/dependency/header/acl_nn_op/aclnn_replication_pad1d_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_replication_pad1d_backward.h) | 64 | 20 | 20 | 40 | 0 | 1 | 64 |
| 376 | 592 | [xla_npu/dependency/header/acl_nn_op/aclnn_roi_align.h](../xla_npu/dependency/header/acl_nn_op/aclnn_roi_align.h) | 64 | 20 | 20 | 40 | 0 | 1 | 64 |
| 377 | 551 | [xla_npu/dependency/header/acl_nn_op/aclnn_roi_align_v2_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_roi_align_v2_backward.h) | 68 | 20 | 20 | 40 | 0 | 1 | 68 |
| 378 | 498 | [xla_npu/dependency/header/acl_nn_op/aclnn_slogdet.h](../xla_npu/dependency/header/acl_nn_op/aclnn_slogdet.h) | 74 | 20 | 20 | 40 | 0 | 1 | 74 |
| 379 | 415 | [xla_npu/dependency/header/acl_nn_op/aclnn_softshrink.h](../xla_npu/dependency/header/acl_nn_op/aclnn_softshrink.h) | 87 | 20 | 20 | 40 | 0 | 1 | 87 |
| 380 | 416 | [xla_npu/dependency/header/acl_nn_op/aclnn_upsample_nearest_2d.h](../xla_npu/dependency/header/acl_nn_op/aclnn_upsample_nearest_2d.h) | 87 | 20 | 20 | 40 | 0 | 1 | 87 |
| 381 | 563 | [xla_npu/dependency/header/acl_nn_op/aclnn_upsample_nearest_2d_v2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_upsample_nearest_2d_v2.h) | 67 | 20 | 20 | 40 | 0 | 1 | 67 |
| 382 | 553 | [xla_npu/dependency/header/acl_nn_op/aclnn_upsample_nearest_exact1d_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_upsample_nearest_exact1d_backward.h) | 68 | 20 | 20 | 40 | 0 | 1 | 68 |
| 383 | 515 | [xla_npu/dependency/header/acl_nn_op/aclnn_upsample_nearest_exact2d.h](../xla_npu/dependency/header/acl_nn_op/aclnn_upsample_nearest_exact2d.h) | 72 | 20 | 20 | 40 | 0 | 1 | 72 |
| 384 | 623 | [xla_npu/dependency/header/acl_nn_op/aclnn_weight_quant_batch_matmul_v2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_weight_quant_batch_matmul_v2.h) | 61 | 20 | 20 | 40 | 0 | 1 | 61 |
| 385 | 461 | [xla_npu/dependency/header/metadef/exe_graph/lowering/lowering_opt.h](../xla_npu/dependency/header/metadef/exe_graph/lowering/lowering_opt.h) | 78 | 20 | 20 | 40 | 0 | 1 | 78 |
| 386 | 300 | [xla_npu/dependency/header/metadef/exe_graph/runtime/range.h](../xla_npu/dependency/header/metadef/exe_graph/runtime/range.h) | 119 | 20 | 20 | 40 | 0 | 1 | 119 |
| 387 | 23 | [xla_npu/service/npu_executable.cc](../xla_npu/service/npu_executable.cc) | 916 | 27 | 12 | 39 | 15 | 3 | 931 |
| 388 | 540 | [xla_npu/dependency/header/acl_nn_op/aclnn_abs.h](../xla_npu/dependency/header/acl_nn_op/aclnn_abs.h) | 71 | 19 | 19 | 38 | 0 | 1 | 71 |
| 389 | 636 | [xla_npu/dependency/header/acl_nn_op/aclnn_circular_pad2d.h](../xla_npu/dependency/header/acl_nn_op/aclnn_circular_pad2d.h) | 60 | 19 | 19 | 38 | 0 | 1 | 60 |
| 390 | 495 | [xla_npu/dependency/header/acl_nn_op/aclnn_equal.h](../xla_npu/dependency/header/acl_nn_op/aclnn_equal.h) | 76 | 19 | 19 | 38 | 0 | 1 | 76 |
| 391 | 542 | [xla_npu/dependency/header/acl_nn_op/aclnn_flatten.h](../xla_npu/dependency/header/acl_nn_op/aclnn_flatten.h) | 71 | 19 | 19 | 38 | 0 | 1 | 71 |
| 392 | 380 | [xla_npu/dependency/header/acl_nn_op/aclnn_floor_divide.h](../xla_npu/dependency/header/acl_nn_op/aclnn_floor_divide.h) | 97 | 19 | 19 | 38 | 0 | 1 | 97 |
| 393 | 557 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_add_scalar_v2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_add_scalar_v2.h) | 69 | 19 | 19 | 38 | 0 | 1 | 69 |
| 394 | 612 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_div_scalar_v2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_div_scalar_v2.h) | 63 | 19 | 19 | 38 | 0 | 1 | 63 |
| 395 | 613 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_maximum_scalar_v2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_maximum_scalar_v2.h) | 63 | 19 | 19 | 38 | 0 | 1 | 63 |
| 396 | 614 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_minimum_scalar_v2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_minimum_scalar_v2.h) | 63 | 19 | 19 | 38 | 0 | 1 | 63 |
| 397 | 615 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_mul_scalar_v2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_mul_scalar_v2.h) | 63 | 19 | 19 | 38 | 0 | 1 | 63 |
| 398 | 616 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_pow_scalar_v2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_pow_scalar_v2.h) | 63 | 19 | 19 | 38 | 0 | 1 | 63 |
| 399 | 629 | [xla_npu/dependency/header/acl_nn_op/aclnn_grouped_matmul_swiglu_quant.h](../xla_npu/dependency/header/acl_nn_op/aclnn_grouped_matmul_swiglu_quant.h) | 62 | 19 | 19 | 38 | 0 | 1 | 62 |
| 400 | 617 | [xla_npu/dependency/header/acl_nn_op/aclnn_grouped_matmul_swiglu_quant_weight_nz.h](../xla_npu/dependency/header/acl_nn_op/aclnn_grouped_matmul_swiglu_quant_weight_nz.h) | 63 | 19 | 19 | 38 | 0 | 1 | 63 |
| 401 | 484 | [xla_npu/dependency/header/acl_nn_op/aclnn_isneginf.h](../xla_npu/dependency/header/acl_nn_op/aclnn_isneginf.h) | 77 | 19 | 19 | 38 | 0 | 1 | 77 |
| 402 | 485 | [xla_npu/dependency/header/acl_nn_op/aclnn_isposinf.h](../xla_npu/dependency/header/acl_nn_op/aclnn_isposinf.h) | 77 | 19 | 19 | 38 | 0 | 1 | 77 |
| 403 | 603 | [xla_npu/dependency/header/acl_nn_op/aclnn_linalg_cross.h](../xla_npu/dependency/header/acl_nn_op/aclnn_linalg_cross.h) | 64 | 19 | 19 | 38 | 0 | 1 | 64 |
| 404 | 571 | [xla_npu/dependency/header/acl_nn_op/aclnn_matmul_all_reduce.h](../xla_npu/dependency/header/acl_nn_op/aclnn_matmul_all_reduce.h) | 68 | 19 | 19 | 38 | 0 | 1 | 68 |
| 405 | 572 | [xla_npu/dependency/header/acl_nn_op/aclnn_matmul_reduce_scatter.h](../xla_npu/dependency/header/acl_nn_op/aclnn_matmul_reduce_scatter.h) | 68 | 19 | 19 | 38 | 0 | 1 | 68 |
| 406 | 459 | [xla_npu/dependency/header/acl_nn_op/aclnn_max_dim.h](../xla_npu/dependency/header/acl_nn_op/aclnn_max_dim.h) | 80 | 19 | 19 | 38 | 0 | 1 | 80 |
| 407 | 597 | [xla_npu/dependency/header/acl_nn_op/aclnn_maximum.h](../xla_npu/dependency/header/acl_nn_op/aclnn_maximum.h) | 65 | 19 | 19 | 38 | 0 | 1 | 65 |
| 408 | 460 | [xla_npu/dependency/header/acl_nn_op/aclnn_min_dim.h](../xla_npu/dependency/header/acl_nn_op/aclnn_min_dim.h) | 80 | 19 | 19 | 38 | 0 | 1 | 80 |
| 409 | 598 | [xla_npu/dependency/header/acl_nn_op/aclnn_minimum.h](../xla_npu/dependency/header/acl_nn_op/aclnn_minimum.h) | 65 | 19 | 19 | 38 | 0 | 1 | 65 |
| 410 | 509 | [xla_npu/dependency/header/acl_nn_op/aclnn_moe_token_unpermute_with_routing_map.h](../xla_npu/dependency/header/acl_nn_op/aclnn_moe_token_unpermute_with_routing_map.h) | 74 | 19 | 19 | 38 | 0 | 1 | 74 |
| 411 | 647 | [xla_npu/dependency/header/acl_nn_op/aclnn_mse_loss.h](../xla_npu/dependency/header/acl_nn_op/aclnn_mse_loss.h) | 59 | 19 | 19 | 38 | 0 | 1 | 59 |
| 412 | 638 | [xla_npu/dependency/header/acl_nn_op/aclnn_mse_loss_out.h](../xla_npu/dependency/header/acl_nn_op/aclnn_mse_loss_out.h) | 60 | 19 | 19 | 38 | 0 | 1 | 60 |
| 413 | 573 | [xla_npu/dependency/header/acl_nn_op/aclnn_multi_scale_deformable_attention_grad.h](../xla_npu/dependency/header/acl_nn_op/aclnn_multi_scale_deformable_attention_grad.h) | 68 | 19 | 19 | 38 | 0 | 1 | 68 |
| 414 | 648 | [xla_npu/dependency/header/acl_nn_op/aclnn_nll_loss2d.h](../xla_npu/dependency/header/acl_nn_op/aclnn_nll_loss2d.h) | 59 | 19 | 19 | 38 | 0 | 1 | 59 |
| 415 | 605 | [xla_npu/dependency/header/acl_nn_op/aclnn_obfuscation_calculate_v2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_obfuscation_calculate_v2.h) | 64 | 19 | 19 | 38 | 0 | 1 | 64 |
| 416 | 561 | [xla_npu/dependency/header/acl_nn_op/aclnn_put.h](../xla_npu/dependency/header/acl_nn_op/aclnn_put.h) | 69 | 19 | 19 | 38 | 0 | 1 | 69 |
| 417 | 650 | [xla_npu/dependency/header/acl_nn_op/aclnn_quant_matmul_v3.h](../xla_npu/dependency/header/acl_nn_op/aclnn_quant_matmul_v3.h) | 59 | 19 | 19 | 38 | 0 | 1 | 59 |
| 418 | 640 | [xla_npu/dependency/header/acl_nn_op/aclnn_reflection_pad2d.h](../xla_npu/dependency/header/acl_nn_op/aclnn_reflection_pad2d.h) | 60 | 19 | 19 | 38 | 0 | 1 | 60 |
| 419 | 631 | [xla_npu/dependency/header/acl_nn_op/aclnn_reflection_pad3d.h](../xla_npu/dependency/header/acl_nn_op/aclnn_reflection_pad3d.h) | 62 | 19 | 19 | 38 | 0 | 1 | 62 |
| 420 | 620 | [xla_npu/dependency/header/acl_nn_op/aclnn_roi_align_v2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_roi_align_v2.h) | 63 | 19 | 19 | 38 | 0 | 1 | 63 |
| 421 | 634 | [xla_npu/dependency/header/acl_nn_op/aclnn_soft_margin_loss.h](../xla_npu/dependency/header/acl_nn_op/aclnn_soft_margin_loss.h) | 61 | 19 | 19 | 38 | 0 | 1 | 61 |
| 422 | 657 | [xla_npu/dependency/header/acl_nn_op/aclnn_swish_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_swish_backward.h) | 58 | 19 | 19 | 38 | 0 | 1 | 58 |
| 423 | 562 | [xla_npu/dependency/header/acl_nn_op/aclnn_tanh_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_tanh_backward.h) | 69 | 19 | 19 | 38 | 0 | 1 | 69 |
| 424 | 609 | [xla_npu/dependency/header/acl_nn_op/aclnn_upsample_nearest_1d_v2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_upsample_nearest_1d_v2.h) | 64 | 19 | 19 | 38 | 0 | 1 | 64 |
| 425 | 575 | [xla_npu/dependency/header/acl_nn_op/aclnn_upsample_nearest_exact1d.h](../xla_npu/dependency/header/acl_nn_op/aclnn_upsample_nearest_exact1d.h) | 68 | 19 | 19 | 38 | 0 | 1 | 68 |
| 426 | 399 | [xla_npu/dependency/header/metadef/exe_graph/runtime/storage_shape.h](../xla_npu/dependency/header/metadef/exe_graph/runtime/storage_shape.h) | 92 | 19 | 19 | 38 | 0 | 1 | 92 |
| 427 | 610 | [xla_npu/dependency/header/acl_nn_op/aclnn_affine_grid.h](../xla_npu/dependency/header/acl_nn_op/aclnn_affine_grid.h) | 65 | 18 | 18 | 36 | 0 | 1 | 65 |
| 428 | 667 | [xla_npu/dependency/header/acl_nn_op/aclnn_complex.h](../xla_npu/dependency/header/acl_nn_op/aclnn_complex.h) | 58 | 18 | 18 | 36 | 0 | 1 | 58 |
| 429 | 452 | [xla_npu/dependency/header/acl_nn_op/aclnn_embedding.h](../xla_npu/dependency/header/acl_nn_op/aclnn_embedding.h) | 83 | 18 | 18 | 36 | 0 | 1 | 83 |
| 430 | 501 | [xla_npu/dependency/header/acl_nn_op/aclnn_expand.h](../xla_npu/dependency/header/acl_nn_op/aclnn_expand.h) | 77 | 18 | 18 | 36 | 0 | 1 | 77 |
| 431 | 586 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_round_off_number_v2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_round_off_number_v2.h) | 68 | 18 | 18 | 36 | 0 | 1 | 68 |
| 432 | 587 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_sub_scalar_v2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_sub_scalar_v2.h) | 68 | 18 | 18 | 36 | 0 | 1 | 68 |
| 433 | 628 | [xla_npu/dependency/header/acl_nn_op/aclnn_gcd.h](../xla_npu/dependency/header/acl_nn_op/aclnn_gcd.h) | 64 | 18 | 18 | 36 | 0 | 1 | 64 |
| 434 | 596 | [xla_npu/dependency/header/acl_nn_op/aclnn_gelu.h](../xla_npu/dependency/header/acl_nn_op/aclnn_gelu.h) | 67 | 18 | 18 | 36 | 0 | 1 | 67 |
| 435 | 645 | [xla_npu/dependency/header/acl_nn_op/aclnn_grouped_matmul_add.h](../xla_npu/dependency/header/acl_nn_op/aclnn_grouped_matmul_add.h) | 61 | 18 | 18 | 36 | 0 | 1 | 61 |
| 436 | 646 | [xla_npu/dependency/header/acl_nn_op/aclnn_isin.h](../xla_npu/dependency/header/acl_nn_op/aclnn_isin.h) | 61 | 18 | 18 | 36 | 0 | 1 | 61 |
| 437 | 637 | [xla_npu/dependency/header/acl_nn_op/aclnn_leaky_relu_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_leaky_relu_backward.h) | 62 | 18 | 18 | 36 | 0 | 1 | 62 |
| 438 | 618 | [xla_npu/dependency/header/acl_nn_op/aclnn_logaddexp.h](../xla_npu/dependency/header/acl_nn_op/aclnn_logaddexp.h) | 65 | 18 | 18 | 36 | 0 | 1 | 65 |
| 439 | 619 | [xla_npu/dependency/header/acl_nn_op/aclnn_logaddexp2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_logaddexp2.h) | 65 | 18 | 18 | 36 | 0 | 1 | 65 |
| 440 | 569 | [xla_npu/dependency/header/acl_nn_op/aclnn_logdet.h](../xla_npu/dependency/header/acl_nn_op/aclnn_logdet.h) | 70 | 18 | 18 | 36 | 0 | 1 | 70 |
| 441 | 508 | [xla_npu/dependency/header/acl_nn_op/aclnn_logsigmoid.h](../xla_npu/dependency/header/acl_nn_op/aclnn_logsigmoid.h) | 76 | 18 | 18 | 36 | 0 | 1 | 76 |
| 442 | 639 | [xla_npu/dependency/header/acl_nn_op/aclnn_obfuscation_calculate.h](../xla_npu/dependency/header/acl_nn_op/aclnn_obfuscation_calculate.h) | 62 | 18 | 18 | 36 | 0 | 1 | 62 |
| 443 | 649 | [xla_npu/dependency/header/acl_nn_op/aclnn_precision_compare.h](../xla_npu/dependency/header/acl_nn_op/aclnn_precision_compare.h) | 61 | 18 | 18 | 36 | 0 | 1 | 61 |
| 444 | 695 | [xla_npu/dependency/header/acl_nn_op/aclnn_reduce_log_sum.h](../xla_npu/dependency/header/acl_nn_op/aclnn_reduce_log_sum.h) | 55 | 18 | 18 | 36 | 0 | 1 | 55 |
| 445 | 664 | [xla_npu/dependency/header/acl_nn_op/aclnn_reflection_pad1d.h](../xla_npu/dependency/header/acl_nn_op/aclnn_reflection_pad1d.h) | 59 | 18 | 18 | 36 | 0 | 1 | 59 |
| 446 | 665 | [xla_npu/dependency/header/acl_nn_op/aclnn_replication_pad3d.h](../xla_npu/dependency/header/acl_nn_op/aclnn_replication_pad3d.h) | 59 | 18 | 18 | 36 | 0 | 1 | 59 |
| 447 | 680 | [xla_npu/dependency/header/acl_nn_op/aclnn_roll.h](../xla_npu/dependency/header/acl_nn_op/aclnn_roll.h) | 57 | 18 | 18 | 36 | 0 | 1 | 57 |
| 448 | 651 | [xla_npu/dependency/header/acl_nn_op/aclnn_rotary_position_embedding.h](../xla_npu/dependency/header/acl_nn_op/aclnn_rotary_position_embedding.h) | 61 | 18 | 18 | 36 | 0 | 1 | 61 |
| 449 | 681 | [xla_npu/dependency/header/acl_nn_op/aclnn_selu_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_selu_backward.h) | 57 | 18 | 18 | 36 | 0 | 1 | 57 |
| 450 | 608 | [xla_npu/dependency/header/acl_nn_op/aclnn_shrink.h](../xla_npu/dependency/header/acl_nn_op/aclnn_shrink.h) | 66 | 18 | 18 | 36 | 0 | 1 | 66 |
| 451 | 670 | [xla_npu/dependency/header/acl_nn_op/aclnn_sigmoid_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_sigmoid_backward.h) | 58 | 18 | 18 | 36 | 0 | 1 | 58 |
| 452 | 544 | [xla_npu/dependency/header/acl_nn_op/aclnn_softplus.h](../xla_npu/dependency/header/acl_nn_op/aclnn_softplus.h) | 73 | 18 | 18 | 36 | 0 | 1 | 73 |
| 453 | 622 | [xla_npu/dependency/header/acl_nn_op/aclnn_take.h](../xla_npu/dependency/header/acl_nn_op/aclnn_take.h) | 65 | 18 | 18 | 36 | 0 | 1 | 65 |
| 454 | 503 | [xla_npu/dependency/header/acl_nn_op/aclnn_zero.h](../xla_npu/dependency/header/acl_nn_op/aclnn_zero.h) | 77 | 18 | 18 | 36 | 0 | 1 | 77 |
| 455 | 334 | [xla_npu/dependency/header/metadef/exe_graph/runtime/infer_datatype_context.h](../xla_npu/dependency/header/metadef/exe_graph/runtime/infer_datatype_context.h) | 112 | 18 | 18 | 36 | 0 | 1 | 112 |
| 456 | 132 | [xla_npu/service/npu_alias_info.cc](../xla_npu/service/npu_alias_info.cc) | 222 | 26 | 10 | 36 | 16 | 2 | 238 |
| 457 | 16 | [xla_npu/dependency/external/afir/Ascend-MLIR/lib/Conversion/AFIRToASCIRText/AFIRToASCIRText.cpp](../xla_npu/dependency/external/afir/Ascend-MLIR/lib/Conversion/AFIRToASCIRText/AFIRToASCIRText.cpp) | 1251 | 30 | 4 | 34 | 26 | 4 | 1277 |
| 458 | 644 | [xla_npu/dependency/header/acl_nn_op/aclnn_amax.h](../xla_npu/dependency/header/acl_nn_op/aclnn_amax.h) | 63 | 17 | 17 | 34 | 0 | 1 | 63 |
| 459 | 659 | [xla_npu/dependency/header/acl_nn_op/aclnn_amin.h](../xla_npu/dependency/header/acl_nn_op/aclnn_amin.h) | 61 | 17 | 17 | 34 | 0 | 1 | 61 |
| 460 | 677 | [xla_npu/dependency/header/acl_nn_op/aclnn_aminmax.h](../xla_npu/dependency/header/acl_nn_op/aclnn_aminmax.h) | 59 | 17 | 17 | 34 | 0 | 1 | 59 |
| 461 | 678 | [xla_npu/dependency/header/acl_nn_op/aclnn_aminmax_dim.h](../xla_npu/dependency/header/acl_nn_op/aclnn_aminmax_dim.h) | 59 | 17 | 17 | 34 | 0 | 1 | 59 |
| 462 | 654 | [xla_npu/dependency/header/acl_nn_op/aclnn_cast.h](../xla_npu/dependency/header/acl_nn_op/aclnn_cast.h) | 62 | 17 | 17 | 34 | 0 | 1 | 62 |
| 463 | 655 | [xla_npu/dependency/header/acl_nn_op/aclnn_distribute_barrier_v2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_distribute_barrier_v2.h) | 62 | 17 | 17 | 34 | 0 | 1 | 62 |
| 464 | 661 | [xla_npu/dependency/header/acl_nn_op/aclnn_fill_diagonal.h](../xla_npu/dependency/header/acl_nn_op/aclnn_fill_diagonal.h) | 61 | 17 | 17 | 34 | 0 | 1 | 61 |
| 465 | 656 | [xla_npu/dependency/header/acl_nn_op/aclnn_fill_scalar.h](../xla_npu/dependency/header/acl_nn_op/aclnn_fill_scalar.h) | 62 | 17 | 17 | 34 | 0 | 1 | 62 |
| 466 | 682 | [xla_npu/dependency/header/acl_nn_op/aclnn_hardshrink_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_hardshrink_backward.h) | 58 | 17 | 17 | 34 | 0 | 1 | 58 |
| 467 | 581 | [xla_npu/dependency/header/acl_nn_op/aclnn_isfinite.h](../xla_npu/dependency/header/acl_nn_op/aclnn_isfinite.h) | 71 | 17 | 17 | 34 | 0 | 1 | 71 |
| 468 | 693 | [xla_npu/dependency/header/acl_nn_op/aclnn_linalg_vector_norm.h](../xla_npu/dependency/header/acl_nn_op/aclnn_linalg_vector_norm.h) | 57 | 17 | 17 | 34 | 0 | 1 | 57 |
| 469 | 683 | [xla_npu/dependency/header/acl_nn_op/aclnn_masked_select.h](../xla_npu/dependency/header/acl_nn_op/aclnn_masked_select.h) | 58 | 17 | 17 | 34 | 0 | 1 | 58 |
| 470 | 679 | [xla_npu/dependency/header/acl_nn_op/aclnn_max_v2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_max_v2.h) | 59 | 17 | 17 | 34 | 0 | 1 | 59 |
| 471 | 669 | [xla_npu/dependency/header/acl_nn_op/aclnn_quantize.h](../xla_npu/dependency/header/acl_nn_op/aclnn_quantize.h) | 60 | 17 | 17 | 34 | 0 | 1 | 60 |
| 472 | 606 | [xla_npu/dependency/header/acl_nn_op/aclnn_real.h](../xla_npu/dependency/header/acl_nn_op/aclnn_real.h) | 68 | 17 | 17 | 34 | 0 | 1 | 68 |
| 473 | 684 | [xla_npu/dependency/header/acl_nn_op/aclnn_replication_pad1d.h](../xla_npu/dependency/header/acl_nn_op/aclnn_replication_pad1d.h) | 58 | 17 | 17 | 34 | 0 | 1 | 58 |
| 474 | 685 | [xla_npu/dependency/header/acl_nn_op/aclnn_replication_pad2d.h](../xla_npu/dependency/header/acl_nn_op/aclnn_replication_pad2d.h) | 58 | 17 | 17 | 34 | 0 | 1 | 58 |
| 475 | 686 | [xla_npu/dependency/header/acl_nn_op/aclnn_scatter_nd_update.h](../xla_npu/dependency/header/acl_nn_op/aclnn_scatter_nd_update.h) | 58 | 17 | 17 | 34 | 0 | 1 | 58 |
| 476 | 706 | [xla_npu/dependency/header/acl_nn_op/aclnn_softshrink_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_softshrink_backward.h) | 56 | 17 | 17 | 34 | 0 | 1 | 56 |
| 477 | 707 | [xla_npu/dependency/header/acl_nn_op/aclnn_split_tensor.h](../xla_npu/dependency/header/acl_nn_op/aclnn_split_tensor.h) | 56 | 17 | 17 | 34 | 0 | 1 | 56 |
| 478 | 687 | [xla_npu/dependency/header/acl_nn_op/aclnn_split_with_size.h](../xla_npu/dependency/header/acl_nn_op/aclnn_split_with_size.h) | 58 | 17 | 17 | 34 | 0 | 1 | 58 |
| 479 | 621 | [xla_npu/dependency/header/acl_nn_op/aclnn_square.h](../xla_npu/dependency/header/acl_nn_op/aclnn_square.h) | 67 | 17 | 17 | 34 | 0 | 1 | 67 |
| 480 | 671 | [xla_npu/dependency/header/acl_nn_op/aclnn_tf_scatter_add.h](../xla_npu/dependency/header/acl_nn_op/aclnn_tf_scatter_add.h) | 60 | 17 | 17 | 34 | 0 | 1 | 60 |
| 481 | 584 | [xla_npu/dependency/header/acl_nn_op/aclnn_all.h](../xla_npu/dependency/header/acl_nn_op/aclnn_all.h) | 72 | 16 | 16 | 32 | 0 | 1 | 72 |
| 482 | 594 | [xla_npu/dependency/header/acl_nn_op/aclnn_attention_update.h](../xla_npu/dependency/header/acl_nn_op/aclnn_attention_update.h) | 71 | 16 | 16 | 32 | 0 | 1 | 71 |
| 483 | 701 | [xla_npu/dependency/header/acl_nn_op/aclnn_channel_shuffle.h](../xla_npu/dependency/header/acl_nn_op/aclnn_channel_shuffle.h) | 58 | 16 | 16 | 32 | 0 | 1 | 58 |
| 484 | 710 | [xla_npu/dependency/header/acl_nn_op/aclnn_constant_pad_nd.h](../xla_npu/dependency/header/acl_nn_op/aclnn_constant_pad_nd.h) | 57 | 16 | 16 | 32 | 0 | 1 | 57 |
| 485 | 692 | [xla_npu/dependency/header/acl_nn_op/aclnn_copy.h](../xla_npu/dependency/header/acl_nn_op/aclnn_copy.h) | 59 | 16 | 16 | 32 | 0 | 1 | 59 |
| 486 | 660 | [xla_npu/dependency/header/acl_nn_op/aclnn_digamma.h](../xla_npu/dependency/header/acl_nn_op/aclnn_digamma.h) | 63 | 16 | 16 | 32 | 0 | 1 | 63 |
| 487 | 702 | [xla_npu/dependency/header/acl_nn_op/aclnn_fast_gelu.h](../xla_npu/dependency/header/acl_nn_op/aclnn_fast_gelu.h) | 58 | 16 | 16 | 32 | 0 | 1 | 58 |
| 488 | 134 | [xla_npu/dependency/header/acl_nn_op/aclnn_flash_attention_score.h](../xla_npu/dependency/header/acl_nn_op/aclnn_flash_attention_score.h) | 223 | 16 | 16 | 32 | 0 | 1 | 223 |
| 489 | 744 | [xla_npu/dependency/header/acl_nn_op/aclnn_gelu_backward_v2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_gelu_backward_v2.h) | 53 | 16 | 16 | 32 | 0 | 1 | 53 |
| 490 | 662 | [xla_npu/dependency/header/acl_nn_op/aclnn_global_average_pool.h](../xla_npu/dependency/header/acl_nn_op/aclnn_global_average_pool.h) | 63 | 16 | 16 | 32 | 0 | 1 | 63 |
| 491 | 704 | [xla_npu/dependency/header/acl_nn_op/aclnn_hardsigmoid_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_hardsigmoid_backward.h) | 58 | 16 | 16 | 32 | 0 | 1 | 58 |
| 492 | 737 | [xla_npu/dependency/header/acl_nn_op/aclnn_index_put_impl.h](../xla_npu/dependency/header/acl_nn_op/aclnn_index_put_impl.h) | 54 | 16 | 16 | 32 | 0 | 1 | 54 |
| 493 | 663 | [xla_npu/dependency/header/acl_nn_op/aclnn_lgamma.h](../xla_npu/dependency/header/acl_nn_op/aclnn_lgamma.h) | 63 | 16 | 16 | 32 | 0 | 1 | 63 |
| 494 | 711 | [xla_npu/dependency/header/acl_nn_op/aclnn_logsumexp.h](../xla_npu/dependency/header/acl_nn_op/aclnn_logsumexp.h) | 57 | 16 | 16 | 32 | 0 | 1 | 57 |
| 495 | 668 | [xla_npu/dependency/header/acl_nn_op/aclnn_multi_scale_deformable_attn_function.h](../xla_npu/dependency/header/acl_nn_op/aclnn_multi_scale_deformable_attn_function.h) | 62 | 16 | 16 | 32 | 0 | 1 | 62 |
| 496 | 721 | [xla_npu/dependency/header/acl_nn_op/aclnn_prelu_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_prelu_backward.h) | 56 | 16 | 16 | 32 | 0 | 1 | 56 |
| 497 | 734 | [xla_npu/dependency/header/acl_nn_op/aclnn_swish.h](../xla_npu/dependency/header/acl_nn_op/aclnn_swish.h) | 55 | 16 | 16 | 32 | 0 | 1 | 55 |
| 498 | 708 | [xla_npu/dependency/header/acl_nn_op/aclnn_transpose_batch_mat_mul.h](../xla_npu/dependency/header/acl_nn_op/aclnn_transpose_batch_mat_mul.h) | 58 | 16 | 16 | 32 | 0 | 1 | 58 |
| 499 | 336 | [xla_npu/dependency/header/metadef/register/op_binary_resource_manager.h](../xla_npu/dependency/header/metadef/register/op_binary_resource_manager.h) | 115 | 16 | 16 | 32 | 0 | 1 | 115 |
| 500 | 1165 | [xla_npu/service/transform/npu_variadic_reduce_expander.h](../xla_npu/service/transform/npu_variadic_reduce_expander.h) | 0 | 31 | 0 | 31 | 31 | 1 | 31 |
| 501 | 700 | [xla_npu/dependency/header/acl_nn_op/aclnn_aminmax_all.h](../xla_npu/dependency/header/acl_nn_op/aclnn_aminmax_all.h) | 60 | 15 | 15 | 30 | 0 | 1 | 60 |
| 502 | 717 | [xla_npu/dependency/header/acl_nn_op/aclnn_distribute_barrier.h](../xla_npu/dependency/header/acl_nn_op/aclnn_distribute_barrier.h) | 58 | 15 | 15 | 30 | 0 | 1 | 58 |
| 503 | 736 | [xla_npu/dependency/header/acl_nn_op/aclnn_elastic_receivable_test.h](../xla_npu/dependency/header/acl_nn_op/aclnn_elastic_receivable_test.h) | 56 | 15 | 15 | 30 | 0 | 1 | 56 |
| 504 | 718 | [xla_npu/dependency/header/acl_nn_op/aclnn_eye.h](../xla_npu/dependency/header/acl_nn_op/aclnn_eye.h) | 58 | 15 | 15 | 30 | 0 | 1 | 58 |
| 505 | 751 | [xla_npu/dependency/header/acl_nn_op/aclnn_fast_gelu_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_fast_gelu_backward.h) | 54 | 15 | 15 | 30 | 0 | 1 | 54 |
| 506 | 703 | [xla_npu/dependency/header/acl_nn_op/aclnn_global_max_pool.h](../xla_npu/dependency/header/acl_nn_op/aclnn_global_max_pool.h) | 60 | 15 | 15 | 30 | 0 | 1 | 60 |
| 507 | 719 | [xla_npu/dependency/header/acl_nn_op/aclnn_histc.h](../xla_npu/dependency/header/acl_nn_op/aclnn_histc.h) | 58 | 15 | 15 | 30 | 0 | 1 | 58 |
| 508 | 738 | [xla_npu/dependency/header/acl_nn_op/aclnn_logsoftmax.h](../xla_npu/dependency/header/acl_nn_op/aclnn_logsoftmax.h) | 56 | 15 | 15 | 30 | 0 | 1 | 56 |
| 509 | 729 | [xla_npu/dependency/header/acl_nn_op/aclnn_maxn.h](../xla_npu/dependency/header/acl_nn_op/aclnn_maxn.h) | 57 | 15 | 15 | 30 | 0 | 1 | 57 |
| 510 | 730 | [xla_npu/dependency/header/acl_nn_op/aclnn_minn.h](../xla_npu/dependency/header/acl_nn_op/aclnn_minn.h) | 57 | 15 | 15 | 30 | 0 | 1 | 57 |
| 511 | 705 | [xla_npu/dependency/header/acl_nn_op/aclnn_moe_distribute_buffer_reset.h](../xla_npu/dependency/header/acl_nn_op/aclnn_moe_distribute_buffer_reset.h) | 60 | 15 | 15 | 30 | 0 | 1 | 60 |
| 512 | 731 | [xla_npu/dependency/header/acl_nn_op/aclnn_nonzero.h](../xla_npu/dependency/header/acl_nn_op/aclnn_nonzero.h) | 57 | 15 | 15 | 30 | 0 | 1 | 57 |
| 513 | 712 | [xla_npu/dependency/header/acl_nn_op/aclnn_pdist.h](../xla_npu/dependency/header/acl_nn_op/aclnn_pdist.h) | 59 | 15 | 15 | 30 | 0 | 1 | 59 |
| 514 | 720 | [xla_npu/dependency/header/acl_nn_op/aclnn_polar.h](../xla_npu/dependency/header/acl_nn_op/aclnn_polar.h) | 58 | 15 | 15 | 30 | 0 | 1 | 58 |
| 515 | 732 | [xla_npu/dependency/header/acl_nn_op/aclnn_resize.h](../xla_npu/dependency/header/acl_nn_op/aclnn_resize.h) | 57 | 15 | 15 | 30 | 0 | 1 | 57 |
| 516 | 739 | [xla_npu/dependency/header/acl_nn_op/aclnn_sign_bits_pack.h](../xla_npu/dependency/header/acl_nn_op/aclnn_sign_bits_pack.h) | 56 | 15 | 15 | 30 | 0 | 1 | 56 |
| 517 | 722 | [xla_npu/dependency/header/acl_nn_op/aclnn_sign_bits_unpack.h](../xla_npu/dependency/header/acl_nn_op/aclnn_sign_bits_unpack.h) | 58 | 15 | 15 | 30 | 0 | 1 | 58 |
| 518 | 713 | [xla_npu/dependency/header/acl_nn_op/aclnn_sinkhorn.h](../xla_npu/dependency/header/acl_nn_op/aclnn_sinkhorn.h) | 59 | 15 | 15 | 30 | 0 | 1 | 59 |
| 519 | 723 | [xla_npu/dependency/header/acl_nn_op/aclnn_sum.h](../xla_npu/dependency/header/acl_nn_op/aclnn_sum.h) | 58 | 15 | 15 | 30 | 0 | 1 | 58 |
| 520 | 1168 | [xla_npu/dependency/interface/acl_custom_op_interface.cc](../xla_npu/dependency/interface/acl_custom_op_interface.cc) | 0 | 30 | 0 | 30 | 30 | 2 | 30 |
| 521 | 419 | [xla_npu/client/npu_module.cc](../xla_npu/client/npu_module.cc) | 98 | 14 | 14 | 28 | 0 | 1 | 98 |
| 522 | 743 | [xla_npu/dependency/header/acl_nn_op/aclnn_apply_top_k_top_p.h](../xla_npu/dependency/header/acl_nn_op/aclnn_apply_top_k_top_p.h) | 57 | 14 | 14 | 28 | 0 | 1 | 57 |
| 523 | 754 | [xla_npu/dependency/header/acl_nn_op/aclnn_diag_flat.h](../xla_npu/dependency/header/acl_nn_op/aclnn_diag_flat.h) | 55 | 14 | 14 | 28 | 0 | 1 | 55 |
| 524 | 728 | [xla_npu/dependency/header/acl_nn_op/aclnn_einsum.h](../xla_npu/dependency/header/acl_nn_op/aclnn_einsum.h) | 59 | 14 | 14 | 28 | 0 | 1 | 59 |
| 525 | 755 | [xla_npu/dependency/header/acl_nn_op/aclnn_elastic_receivable_info_collect.h](../xla_npu/dependency/header/acl_nn_op/aclnn_elastic_receivable_info_collect.h) | 55 | 14 | 14 | 28 | 0 | 1 | 55 |
| 526 | 123 | [xla_npu/dependency/header/acl_nn_op/aclnn_flash_attention_score_grad.h](../xla_npu/dependency/header/acl_nn_op/aclnn_flash_attention_score_grad.h) | 238 | 14 | 14 | 28 | 0 | 1 | 238 |
| 527 | 759 | [xla_npu/dependency/header/acl_nn_op/aclnn_norm.h](../xla_npu/dependency/header/acl_nn_op/aclnn_norm.h) | 54 | 14 | 14 | 28 | 0 | 1 | 54 |
| 528 | 733 | [xla_npu/dependency/header/acl_nn_op/aclnn_softmax_cross_entropy_with_logits.h](../xla_npu/dependency/header/acl_nn_op/aclnn_softmax_cross_entropy_with_logits.h) | 59 | 14 | 14 | 28 | 0 | 1 | 59 |
| 529 | 1190 | [xla_npu/service/transform/npu_fma_addcmul_rewriter.h](../xla_npu/service/transform/npu_fma_addcmul_rewriter.h) | 0 | 28 | 0 | 28 | 28 | 1 | 28 |
| 530 | 492 | [xla_npu/service/transform/npu_gemm_fusion_swap_operands.h](../xla_npu/service/transform/npu_gemm_fusion_swap_operands.h) | 87 | 14 | 14 | 28 | 0 | 1 | 87 |
| 531 | 727 | [xla_npu/dependency/header/acl_nn_op/aclnn_diag.h](../xla_npu/dependency/header/acl_nn_op/aclnn_diag.h) | 61 | 13 | 13 | 26 | 0 | 1 | 61 |
| 532 | 763 | [xla_npu/dependency/header/acl_nn_op/aclnn_isin_tensor_scalar.h](../xla_npu/dependency/header/acl_nn_op/aclnn_isin_tensor_scalar.h) | 55 | 13 | 13 | 26 | 0 | 1 | 55 |
| 533 | 772 | [xla_npu/dependency/header/acl_nn_op/aclnn_trans_quant_param_v3.h](../xla_npu/dependency/header/acl_nn_op/aclnn_trans_quant_param_v3.h) | 52 | 13 | 13 | 26 | 0 | 1 | 52 |
| 534 | 740 | [xla_npu/dependency/header/acl_nn_op/aclnn_upsample_bilinear2d_aa.h](../xla_npu/dependency/header/acl_nn_op/aclnn_upsample_bilinear2d_aa.h) | 60 | 13 | 13 | 26 | 0 | 1 | 60 |
| 535 | 211 | [xla_npu/dependency/tool/function_loader.h](../xla_npu/dependency/tool/function_loader.h) | 164 | 13 | 13 | 26 | 0 | 1 | 164 |
| 536 | 344 | [xla_npu/dependency/tool/log.cc](../xla_npu/dependency/tool/log.cc) | 119 | 13 | 13 | 26 | 0 | 1 | 119 |
| 537 | 790 | [xla_npu/dependency/header/acl_nn_op/aclnn_max.h](../xla_npu/dependency/header/acl_nn_op/aclnn_max.h) | 50 | 12 | 12 | 24 | 0 | 1 | 50 |
| 538 | 791 | [xla_npu/dependency/header/acl_nn_op/aclnn_min.h](../xla_npu/dependency/header/acl_nn_op/aclnn_min.h) | 50 | 12 | 12 | 24 | 0 | 1 | 50 |
| 539 | 779 | [xla_npu/dependency/header/acl_nn_op/aclnn_signbit.h](../xla_npu/dependency/header/acl_nn_op/aclnn_signbit.h) | 52 | 12 | 12 | 24 | 0 | 1 | 52 |
| 540 | 784 | [xla_npu/dependency/header/acl_nn_op/aclnn_trans_quant_param_v2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_trans_quant_param_v2.h) | 51 | 12 | 12 | 24 | 0 | 1 | 51 |
| 541 | 564 | [xla_npu/service/transform/npu_fusion_block_level_rewriter.h](../xla_npu/service/transform/npu_fusion_block_level_rewriter.h) | 84 | 16 | 7 | 23 | 9 | 3 | 93 |
| 542 | 1203 | [xla_npu/service/transform/npu_scatter_window_expander.h](../xla_npu/service/transform/npu_scatter_window_expander.h) | 0 | 23 | 0 | 23 | 23 | 1 | 23 |
| 543 | 44 | [xla_npu/dependency/header/ge/ge_api.h](../xla_npu/dependency/header/ge/ge_api.h) | 508 | 11 | 11 | 22 | 0 | 1 | 508 |
| 544 | 776 | [xla_npu/common/npu_config.h](../xla_npu/common/npu_config.h) | 55 | 11 | 10 | 21 | 1 | 4 | 56 |
| 545 | 313 | [xla_npu/common/npu_tool.cc](../xla_npu/common/npu_tool.cc) | 132 | 10 | 10 | 20 | 0 | 1 | 132 |
| 546 | 137 | [xla_npu/dependency/header/metadef/register/tuning_bank_key_registry.h](../xla_npu/dependency/header/metadef/register/tuning_bank_key_registry.h) | 232 | 10 | 10 | 20 | 0 | 1 | 232 |
| 547 | 92 | [xla_npu/pjrt/air/mair/GATHER_ANALYSIS.md](../xla_npu/pjrt/air/mair/GATHER_ANALYSIS.md) | 305 | 10 | 10 | 20 | 0 | 1 | 305 |
| 548 | 675 | [xla_npu/service/transform/npu_fusion_dispatch_pipeline.h](../xla_npu/service/transform/npu_fusion_dispatch_pipeline.h) | 74 | 10 | 10 | 20 | 0 | 1 | 74 |
| 549 | 1210 | [xla_npu/service/transform/npu_right_triangular_solve_expander.h](../xla_npu/service/transform/npu_right_triangular_solve_expander.h) | 0 | 20 | 0 | 20 | 20 | 1 | 20 |
| 550 | 104 | [xla_npu/stream_executor/npu_executor.cc](../xla_npu/stream_executor/npu_executor.cc) | 275 | 10 | 10 | 20 | 0 | 2 | 275 |
| 551 | 834 | [xla_npu/dependency/header/acl_nn_op/aclnn_fused_matmul.h](../xla_npu/dependency/header/acl_nn_op/aclnn_fused_matmul.h) | 47 | 9 | 9 | 18 | 0 | 1 | 47 |
| 552 | 207 | [xla_npu/stream_executor/npu_auto_fuse_kernel.cc](../xla_npu/stream_executor/npu_auto_fuse_kernel.cc) | 174 | 9 | 9 | 18 | 0 | 1 | 174 |
| 553 | 624 | [xla_npu/backend/runtime/npu_aclnn_thunk.h](../xla_npu/backend/runtime/npu_aclnn_thunk.h) | 83 | 17 | 0 | 17 | 17 | 1 | 100 |
| 554 | 354 | [xla_npu/dependency/interface/acl_interface.h](../xla_npu/dependency/interface/acl_interface.h) | 127 | 17 | 0 | 17 | 17 | 2 | 144 |
| 555 | 765 | [xla_npu/dependency/interface/acl_nn_interface.h](../xla_npu/dependency/interface/acl_nn_interface.h) | 64 | 17 | 0 | 17 | 17 | 2 | 81 |
| 556 | 130 | [xla_npu/dependency/external/afir/Ascend-MLIR/include/Dialect/AFIR/Base.td](../xla_npu/dependency/external/afir/Ascend-MLIR/include/Dialect/AFIR/Base.td) | 242 | 8 | 8 | 16 | 0 | 1 | 242 |
| 557 | 531 | [xla_npu/service/kernel/npu_custom_kernel.h](../xla_npu/service/kernel/npu_custom_kernel.h) | 95 | 8 | 8 | 16 | 0 | 1 | 95 |
| 558 | 916 | [xla_npu/service/transform/npu_hlo_printer_pass.cc](../xla_npu/service/transform/npu_hlo_printer_pass.cc) | 35 | 14 | 2 | 16 | 12 | 3 | 47 |
| 559 | 760 | [xla_npu/dependency/header/metadef/exe_graph/runtime/tiling_parse_context.h](../xla_npu/dependency/header/metadef/exe_graph/runtime/tiling_parse_context.h) | 68 | 7 | 7 | 14 | 0 | 1 | 68 |
| 560 | 577 | [xla_npu/service/kernel/npu_custom_kernel.cc](../xla_npu/service/kernel/npu_custom_kernel.cc) | 92 | 7 | 7 | 14 | 0 | 1 | 92 |
| 561 | 848 | [xla_npu/service/npu_alias_info.h](../xla_npu/service/npu_alias_info.h) | 48 | 14 | 0 | 14 | 14 | 1 | 62 |
| 562 | 403 | [xla_npu/service/npu_ir_emitter_unnested.h](../xla_npu/service/npu_ir_emitter_unnested.h) | 116 | 12 | 1 | 13 | 11 | 6 | 127 |
| 563 | 643 | [xla_npu/client/npu_stream_client.cc](../xla_npu/client/npu_stream_client.cc) | 85 | 6 | 6 | 12 | 0 | 1 | 85 |
| 564 | 210 | [xla_npu/dependency/header/acl_nn/opdev/aicpu/aicpu_args_handler.h](../xla_npu/dependency/header/acl_nn/opdev/aicpu/aicpu_args_handler.h) | 178 | 6 | 6 | 12 | 0 | 1 | 178 |
| 565 | 964 | [xla_npu/dependency/interface/BUILD](../xla_npu/dependency/interface/BUILD) | 36 | 11 | 1 | 12 | 10 | 1 | 46 |
| 566 | 635 | [xla_npu/pjrt/air/mair/CMakeLists.txt](../xla_npu/pjrt/air/mair/CMakeLists.txt) | 87 | 6 | 6 | 12 | 0 | 1 | 87 |
| 567 | 433 | [xla_npu/service/transform/npu_fusion_pipeline.cc](../xla_npu/service/transform/npu_fusion_pipeline.cc) | 113 | 9 | 3 | 12 | 6 | 2 | 119 |
| 568 | 874 | [xla_npu/common/npu_config.cc](../xla_npu/common/npu_config.cc) | 45 | 7 | 4 | 11 | 3 | 3 | 48 |
| 569 | 698 | [xla_npu/service/transform/npu_priority_fusion.h](../xla_npu/service/transform/npu_priority_fusion.h) | 80 | 11 | 0 | 11 | 11 | 1 | 91 |
| 570 | 532 | [xla_npu/backend/runtime/npu_custom_kernel_thunk.h](../xla_npu/backend/runtime/npu_custom_kernel_thunk.h) | 100 | 5 | 5 | 10 | 0 | 1 | 100 |
| 571 | 694 | [xla_npu/dependency/header/acl_nn_op/aclnn_nsa_compress_attention_infer.h](../xla_npu/dependency/header/acl_nn_op/aclnn_nsa_compress_attention_infer.h) | 81 | 5 | 5 | 10 | 0 | 1 | 81 |
| 572 | 599 | [xla_npu/dependency/header/metadef/graph/utils/args_format_desc_utils.h](../xla_npu/dependency/header/metadef/graph/utils/args_format_desc_utils.h) | 93 | 5 | 5 | 10 | 0 | 1 | 93 |
| 573 | 169 | [xla_npu/pjrt/air/mair/GATHERV2_IMPLEMENTATION.md](../xla_npu/pjrt/air/mair/GATHERV2_IMPLEMENTATION.md) | 204 | 5 | 5 | 10 | 0 | 1 | 204 |
| 574 | 192 | [xla_npu/stream_executor/npu_stream.cc](../xla_npu/stream_executor/npu_stream.cc) | 191 | 8 | 2 | 10 | 6 | 3 | 197 |
| 575 | 177 | [xla_npu/backend/runtime/npu_custom_kernel_thunk.cc](../xla_npu/backend/runtime/npu_custom_kernel_thunk.cc) | 202 | 4 | 4 | 8 | 0 | 1 | 202 |
| 576 | 321 | [xla_npu/backend/runtime/npu_sequential_thunk.cc](../xla_npu/backend/runtime/npu_sequential_thunk.cc) | 143 | 7 | 1 | 8 | 6 | 1 | 149 |
| 577 | 769 | [xla_npu/dependency/header/acl_nn_op/aclnn_incre_flash_attention_v2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_incre_flash_attention_v2.h) | 71 | 4 | 4 | 8 | 0 | 1 | 71 |
| 578 | 799 | [xla_npu/dependency/header/acl_nn_op/aclnn_index_fill.h](../xla_npu/dependency/header/acl_nn_op/aclnn_index_fill.h) | 63 | 4 | 4 | 8 | 0 | 1 | 63 |
| 579 | 835 | [xla_npu/dependency/header/acl_nn_op/aclnn_index_fill_tensor.h](../xla_npu/dependency/header/acl_nn_op/aclnn_index_fill_tensor.h) | 57 | 4 | 4 | 8 | 0 | 1 | 57 |
| 580 | 810 | [xla_npu/dependency/header/acl_nn_op/aclnn_layer_norm.h](../xla_npu/dependency/header/acl_nn_op/aclnn_layer_norm.h) | 60 | 4 | 4 | 8 | 0 | 1 | 60 |
| 581 | 836 | [xla_npu/dependency/header/acl_nn_op/aclnn_mean.h](../xla_npu/dependency/header/acl_nn_op/aclnn_mean.h) | 57 | 4 | 4 | 8 | 0 | 1 | 57 |
| 582 | 823 | [xla_npu/dependency/header/acl_nn_op/aclnn_reciprocal.h](../xla_npu/dependency/header/acl_nn_op/aclnn_reciprocal.h) | 58 | 4 | 4 | 8 | 0 | 1 | 58 |
| 583 | 824 | [xla_npu/dependency/header/acl_nn_op/aclnn_rrelu_with_noise.h](../xla_npu/dependency/header/acl_nn_op/aclnn_rrelu_with_noise.h) | 58 | 4 | 4 | 8 | 0 | 1 | 58 |
| 584 | 839 | [xla_npu/dependency/header/acl_nn_op/aclnn_var.h](../xla_npu/dependency/header/acl_nn_op/aclnn_var.h) | 56 | 4 | 4 | 8 | 0 | 1 | 56 |
| 585 | 840 | [xla_npu/dependency/header/acl_nn_op/aclnn_x_log_y_scalar_other.h](../xla_npu/dependency/header/acl_nn_op/aclnn_x_log_y_scalar_other.h) | 56 | 4 | 4 | 8 | 0 | 1 | 56 |
| 586 | 841 | [xla_npu/dependency/header/acl_nn_op/aclnn_x_log_y_tensor.h](../xla_npu/dependency/header/acl_nn_op/aclnn_x_log_y_tensor.h) | 56 | 4 | 4 | 8 | 0 | 1 | 56 |
| 587 | 756 | [xla_npu/dependency/header/metadef/exe_graph/runtime/context_extend.h](../xla_npu/dependency/header/metadef/exe_graph/runtime/context_extend.h) | 75 | 4 | 4 | 8 | 0 | 1 | 75 |
| 588 | 24 | [xla_npu/dependency/header/metadef/graph/operator.h](../xla_npu/dependency/header/metadef/graph/operator.h) | 932 | 4 | 4 | 8 | 0 | 1 | 932 |
| 589 | 794 | [xla_npu/dependency/tool/log.h](../xla_npu/dependency/tool/log.h) | 66 | 4 | 4 | 8 | 0 | 1 | 66 |
| 590 | 748 | [xla_npu/stream_executor/npu_auto_fuse_kernel.h](../xla_npu/stream_executor/npu_auto_fuse_kernel.h) | 77 | 4 | 4 | 8 | 0 | 1 | 77 |
| 591 | 376 | [xla_npu/backend/codegen/BUILD](../xla_npu/backend/codegen/BUILD) | 130 | 4 | 3 | 7 | 1 | 2 | 131 |
| 592 | 195 | [xla_npu/dependency/external/afir/Ascend-MLIR/include/Dialect/AFIR/Core/Attrs.td](../xla_npu/dependency/external/afir/Ascend-MLIR/include/Dialect/AFIR/Core/Attrs.td) | 193 | 3 | 3 | 6 | 0 | 1 | 193 |
| 593 | 68 | [xla_npu/dependency/header/acl_nn/opdev/common_types.h](../xla_npu/dependency/header/acl_nn/opdev/common_types.h) | 408 | 3 | 3 | 6 | 0 | 1 | 408 |
| 594 | 42 | [xla_npu/dependency/header/acl_nn/opdev/small_vector.h](../xla_npu/dependency/header/acl_nn/opdev/small_vector.h) | 548 | 3 | 3 | 6 | 0 | 1 | 548 |
| 595 | 1041 | [xla_npu/dependency/header/acl_nn_op/aclnn_arange.h](../xla_npu/dependency/header/acl_nn_op/aclnn_arange.h) | 38 | 3 | 3 | 6 | 0 | 1 | 38 |
| 596 | 953 | [xla_npu/dependency/header/acl_nn_op/aclnn_binary_cross_entropy_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_binary_cross_entropy_backward.h) | 42 | 3 | 3 | 6 | 0 | 1 | 42 |
| 597 | 954 | [xla_npu/dependency/header/acl_nn_op/aclnn_binary_cross_entropy_with_logits_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_binary_cross_entropy_with_logits_backward.h) | 42 | 3 | 3 | 6 | 0 | 1 | 42 |
| 598 | 884 | [xla_npu/dependency/header/acl_nn_op/aclnn_binary_cross_entropy_with_logits_target_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_binary_cross_entropy_with_logits_target_backward.h) | 49 | 3 | 3 | 6 | 0 | 1 | 49 |
| 599 | 802 | [xla_npu/dependency/header/acl_nn_op/aclnn_incre_flash_attention.h](../xla_npu/dependency/header/acl_nn_op/aclnn_incre_flash_attention.h) | 64 | 3 | 3 | 6 | 0 | 1 | 64 |
| 600 | 745 | [xla_npu/dependency/header/acl_nn_op/aclnn_incre_flash_attention_v3.h](../xla_npu/dependency/header/acl_nn_op/aclnn_incre_flash_attention_v3.h) | 79 | 3 | 3 | 6 | 0 | 1 | 79 |
| 601 | 746 | [xla_npu/dependency/header/acl_nn_op/aclnn_incre_flash_attention_v4.h](../xla_npu/dependency/header/acl_nn_op/aclnn_incre_flash_attention_v4.h) | 79 | 3 | 3 | 6 | 0 | 1 | 79 |
| 602 | 998 | [xla_npu/dependency/header/acl_nn_op/aclnn_silu_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_silu_backward.h) | 40 | 3 | 3 | 6 | 0 | 1 | 40 |
| 603 | 252 | [xla_npu/dependency/header/metadef/register/op_impl_kernel_registry.h](../xla_npu/dependency/header/metadef/register/op_impl_kernel_registry.h) | 167 | 3 | 3 | 6 | 0 | 1 | 167 |
| 604 | 366 | [xla_npu/pjrt/air/mair/stablehlo2ge.cpp](../xla_npu/pjrt/air/mair/stablehlo2ge.cpp) | 135 | 3 | 3 | 6 | 0 | 1 | 135 |
| 605 | 1072 | [xla_npu/service/kernel/npu_compile_result.h](../xla_npu/service/kernel/npu_compile_result.h) | 37 | 3 | 3 | 6 | 0 | 1 | 37 |
| 606 | 881 | [xla_npu/dependency/interface/acl_interface.cc](../xla_npu/dependency/interface/acl_interface.cc) | 51 | 5 | 0 | 5 | 5 | 2 | 56 |
| 607 | 32 | [xla_npu/service/BUILD](../xla_npu/service/BUILD) | 709 | 4 | 1 | 5 | 3 | 4 | 712 |
| 608 | 845 | [xla_npu/common/npu_constant.h](../xla_npu/common/npu_constant.h) | 58 | 2 | 2 | 4 | 0 | 1 | 58 |
| 609 | 1204 | [xla_npu/dependency/header/acl/BUILD](../xla_npu/dependency/header/acl/BUILD) | 18 | 2 | 2 | 4 | 0 | 1 | 18 |
| 610 | 1198 | [xla_npu/dependency/header/acl_nn/BUILD](../xla_npu/dependency/header/acl_nn/BUILD) | 21 | 2 | 2 | 4 | 0 | 1 | 21 |
| 611 | 73 | [xla_npu/dependency/header/acl_nn/opdev/aicpu/aicpu_task.h](../xla_npu/dependency/header/acl_nn/opdev/aicpu/aicpu_task.h) | 387 | 2 | 2 | 4 | 0 | 1 | 387 |
| 612 | 1199 | [xla_npu/dependency/header/acl_nn_op/BUILD](../xla_npu/dependency/header/acl_nn_op/BUILD) | 21 | 2 | 2 | 4 | 0 | 1 | 21 |
| 613 | 1037 | [xla_npu/dependency/header/acl_nn_op/aclnn_adaptive_avg_pool2d.h](../xla_npu/dependency/header/acl_nn_op/aclnn_adaptive_avg_pool2d.h) | 40 | 2 | 2 | 4 | 0 | 1 | 40 |
| 614 | 1006 | [xla_npu/dependency/header/acl_nn_op/aclnn_adaptive_avg_pool3d.h](../xla_npu/dependency/header/acl_nn_op/aclnn_adaptive_avg_pool3d.h) | 41 | 2 | 2 | 4 | 0 | 1 | 41 |
| 615 | 1038 | [xla_npu/dependency/header/acl_nn_op/aclnn_adaptive_max_pool2d.h](../xla_npu/dependency/header/acl_nn_op/aclnn_adaptive_max_pool2d.h) | 40 | 2 | 2 | 4 | 0 | 1 | 40 |
| 616 | 1039 | [xla_npu/dependency/header/acl_nn_op/aclnn_adaptive_max_pool3d.h](../xla_npu/dependency/header/acl_nn_op/aclnn_adaptive_max_pool3d.h) | 40 | 2 | 2 | 4 | 0 | 1 | 40 |
| 617 | 1065 | [xla_npu/dependency/header/acl_nn_op/aclnn_addmv.h](../xla_npu/dependency/header/acl_nn_op/aclnn_addmv.h) | 39 | 2 | 2 | 4 | 0 | 1 | 39 |
| 618 | 988 | [xla_npu/dependency/header/acl_nn_op/aclnn_any.h](../xla_npu/dependency/header/acl_nn_op/aclnn_any.h) | 42 | 2 | 2 | 4 | 0 | 1 | 42 |
| 619 | 1040 | [xla_npu/dependency/header/acl_nn_op/aclnn_apply_adam_w_v2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_apply_adam_w_v2.h) | 40 | 2 | 2 | 4 | 0 | 1 | 40 |
| 620 | 1080 | [xla_npu/dependency/header/acl_nn_op/aclnn_apply_rotary_pos_emb.h](../xla_npu/dependency/header/acl_nn_op/aclnn_apply_rotary_pos_emb.h) | 38 | 2 | 2 | 4 | 0 | 1 | 38 |
| 621 | 1081 | [xla_npu/dependency/header/acl_nn_op/aclnn_apply_rotary_pos_emb_v2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_apply_rotary_pos_emb_v2.h) | 38 | 2 | 2 | 4 | 0 | 1 | 38 |
| 622 | 989 | [xla_npu/dependency/header/acl_nn_op/aclnn_argsort.h](../xla_npu/dependency/header/acl_nn_op/aclnn_argsort.h) | 42 | 2 | 2 | 4 | 0 | 1 | 42 |
| 623 | 972 | [xla_npu/dependency/header/acl_nn_op/aclnn_attention_worker_scheduler.h](../xla_npu/dependency/header/acl_nn_op/aclnn_attention_worker_scheduler.h) | 43 | 2 | 2 | 4 | 0 | 1 | 43 |
| 624 | 973 | [xla_npu/dependency/header/acl_nn_op/aclnn_avgpool2d.h](../xla_npu/dependency/header/acl_nn_op/aclnn_avgpool2d.h) | 43 | 2 | 2 | 4 | 0 | 1 | 43 |
| 625 | 974 | [xla_npu/dependency/header/acl_nn_op/aclnn_avgpool2d_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_avgpool2d_backward.h) | 43 | 2 | 2 | 4 | 0 | 1 | 43 |
| 626 | 1042 | [xla_npu/dependency/header/acl_nn_op/aclnn_batch_norm.h](../xla_npu/dependency/header/acl_nn_op/aclnn_batch_norm.h) | 40 | 2 | 2 | 4 | 0 | 1 | 40 |
| 627 | 931 | [xla_npu/dependency/header/acl_nn_op/aclnn_batch_norm_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_batch_norm_backward.h) | 45 | 2 | 2 | 4 | 0 | 1 | 45 |
| 628 | 952 | [xla_npu/dependency/header/acl_nn_op/aclnn_batch_norm_backward_reduce.h](../xla_npu/dependency/header/acl_nn_op/aclnn_batch_norm_backward_reduce.h) | 44 | 2 | 2 | 4 | 0 | 1 | 44 |
| 629 | 1043 | [xla_npu/dependency/header/acl_nn_op/aclnn_batch_norm_elemt.h](../xla_npu/dependency/header/acl_nn_op/aclnn_batch_norm_elemt.h) | 40 | 2 | 2 | 4 | 0 | 1 | 40 |
| 630 | 990 | [xla_npu/dependency/header/acl_nn_op/aclnn_batch_norm_elemt_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_batch_norm_elemt_backward.h) | 42 | 2 | 2 | 4 | 0 | 1 | 42 |
| 631 | 1007 | [xla_npu/dependency/header/acl_nn_op/aclnn_batch_norm_gather_stats_with_counts.h](../xla_npu/dependency/header/acl_nn_op/aclnn_batch_norm_gather_stats_with_counts.h) | 41 | 2 | 2 | 4 | 0 | 1 | 41 |
| 632 | 1008 | [xla_npu/dependency/header/acl_nn_op/aclnn_batch_norm_stats.h](../xla_npu/dependency/header/acl_nn_op/aclnn_batch_norm_stats.h) | 41 | 2 | 2 | 4 | 0 | 1 | 41 |
| 633 | 908 | [xla_npu/dependency/header/acl_nn_op/aclnn_batchmatmul_quant.h](../xla_npu/dependency/header/acl_nn_op/aclnn_batchmatmul_quant.h) | 47 | 2 | 2 | 4 | 0 | 1 | 47 |
| 634 | 975 | [xla_npu/dependency/header/acl_nn_op/aclnn_binary_cross_entropy.h](../xla_npu/dependency/header/acl_nn_op/aclnn_binary_cross_entropy.h) | 43 | 2 | 2 | 4 | 0 | 1 | 43 |
| 635 | 1009 | [xla_npu/dependency/header/acl_nn_op/aclnn_binary_cross_entropy_with_logits.h](../xla_npu/dependency/header/acl_nn_op/aclnn_binary_cross_entropy_with_logits.h) | 41 | 2 | 2 | 4 | 0 | 1 | 41 |
| 636 | 991 | [xla_npu/dependency/header/acl_nn_op/aclnn_cat.h](../xla_npu/dependency/header/acl_nn_op/aclnn_cat.h) | 42 | 2 | 2 | 4 | 0 | 1 | 42 |
| 637 | 1010 | [xla_npu/dependency/header/acl_nn_op/aclnn_chamfer_distance_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_chamfer_distance_backward.h) | 41 | 2 | 2 | 4 | 0 | 1 | 41 |
| 638 | 1044 | [xla_npu/dependency/header/acl_nn_op/aclnn_convert_weight_to_int4_pack.h](../xla_npu/dependency/header/acl_nn_op/aclnn_convert_weight_to_int4_pack.h) | 40 | 2 | 2 | 4 | 0 | 1 | 40 |
| 639 | 992 | [xla_npu/dependency/header/acl_nn_op/aclnn_dynamic_quant.h](../xla_npu/dependency/header/acl_nn_op/aclnn_dynamic_quant.h) | 42 | 2 | 2 | 4 | 0 | 1 | 42 |
| 640 | 1013 | [xla_npu/dependency/header/acl_nn_op/aclnn_dynamic_quant_v2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_dynamic_quant_v2.h) | 41 | 2 | 2 | 4 | 0 | 1 | 41 |
| 641 | 920 | [xla_npu/dependency/header/acl_nn_op/aclnn_ffn_worker_scheduler.h](../xla_npu/dependency/header/acl_nn_op/aclnn_ffn_worker_scheduler.h) | 46 | 2 | 2 | 4 | 0 | 1 | 46 |
| 642 | 851 | [xla_npu/dependency/header/acl_nn_op/aclnn_flash_attention_score_grad_vx.h](../xla_npu/dependency/header/acl_nn_op/aclnn_flash_attention_score_grad_vx.h) | 57 | 2 | 2 | 4 | 0 | 1 | 57 |
| 643 | 870 | [xla_npu/dependency/header/acl_nn_op/aclnn_flash_attention_score_vx.h](../xla_npu/dependency/header/acl_nn_op/aclnn_flash_attention_score_vx.h) | 54 | 2 | 2 | 4 | 0 | 1 | 54 |
| 644 | 1066 | [xla_npu/dependency/header/acl_nn_op/aclnn_gemm.h](../xla_npu/dependency/header/acl_nn_op/aclnn_gemm.h) | 39 | 2 | 2 | 4 | 0 | 1 | 39 |
| 645 | 1014 | [xla_npu/dependency/header/acl_nn_op/aclnn_ger.h](../xla_npu/dependency/header/acl_nn_op/aclnn_ger.h) | 41 | 2 | 2 | 4 | 0 | 1 | 41 |
| 646 | 932 | [xla_npu/dependency/header/acl_nn_op/aclnn_grid_sampler3d.h](../xla_npu/dependency/header/acl_nn_op/aclnn_grid_sampler3d.h) | 45 | 2 | 2 | 4 | 0 | 1 | 45 |
| 647 | 976 | [xla_npu/dependency/header/acl_nn_op/aclnn_grid_sampler3d_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_grid_sampler3d_backward.h) | 43 | 2 | 2 | 4 | 0 | 1 | 43 |
| 648 | 993 | [xla_npu/dependency/header/acl_nn_op/aclnn_group_norm.h](../xla_npu/dependency/header/acl_nn_op/aclnn_group_norm.h) | 42 | 2 | 2 | 4 | 0 | 1 | 42 |
| 649 | 955 | [xla_npu/dependency/header/acl_nn_op/aclnn_group_norm_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_group_norm_backward.h) | 44 | 2 | 2 | 4 | 0 | 1 | 44 |
| 650 | 1015 | [xla_npu/dependency/header/acl_nn_op/aclnn_im2col.h](../xla_npu/dependency/header/acl_nn_op/aclnn_im2col.h) | 41 | 2 | 2 | 4 | 0 | 1 | 41 |
| 651 | 994 | [xla_npu/dependency/header/acl_nn_op/aclnn_im2col_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_im2col_backward.h) | 42 | 2 | 2 | 4 | 0 | 1 | 42 |
| 652 | 1046 | [xla_npu/dependency/header/acl_nn_op/aclnn_index.h](../xla_npu/dependency/header/acl_nn_op/aclnn_index.h) | 40 | 2 | 2 | 4 | 0 | 1 | 40 |
| 653 | 1047 | [xla_npu/dependency/header/acl_nn_op/aclnn_index_add.h](../xla_npu/dependency/header/acl_nn_op/aclnn_index_add.h) | 40 | 2 | 2 | 4 | 0 | 1 | 40 |
| 654 | 995 | [xla_npu/dependency/header/acl_nn_op/aclnn_kl_div.h](../xla_npu/dependency/header/acl_nn_op/aclnn_kl_div.h) | 42 | 2 | 2 | 4 | 0 | 1 | 42 |
| 655 | 1016 | [xla_npu/dependency/header/acl_nn_op/aclnn_kl_div_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_kl_div_backward.h) | 41 | 2 | 2 | 4 | 0 | 1 | 41 |
| 656 | 933 | [xla_npu/dependency/header/acl_nn_op/aclnn_layer_norm_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_layer_norm_backward.h) | 45 | 2 | 2 | 4 | 0 | 1 | 45 |
| 657 | 887 | [xla_npu/dependency/header/acl_nn_op/aclnn_lightning_indexer_grad.h](../xla_npu/dependency/header/acl_nn_op/aclnn_lightning_indexer_grad.h) | 50 | 2 | 2 | 4 | 0 | 1 | 50 |
| 658 | 1017 | [xla_npu/dependency/header/acl_nn_op/aclnn_logsigmoid_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_logsigmoid_backward.h) | 41 | 2 | 2 | 4 | 0 | 1 | 41 |
| 659 | 1018 | [xla_npu/dependency/header/acl_nn_op/aclnn_logsoftmax_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_logsoftmax_backward.h) | 41 | 2 | 2 | 4 | 0 | 1 | 41 |
| 660 | 909 | [xla_npu/dependency/header/acl_nn_op/aclnn_matmul_compress.h](../xla_npu/dependency/header/acl_nn_op/aclnn_matmul_compress.h) | 47 | 2 | 2 | 4 | 0 | 1 | 47 |
| 661 | 977 | [xla_npu/dependency/header/acl_nn_op/aclnn_matmul_compress_dequant.h](../xla_npu/dependency/header/acl_nn_op/aclnn_matmul_compress_dequant.h) | 43 | 2 | 2 | 4 | 0 | 1 | 43 |
| 662 | 996 | [xla_npu/dependency/header/acl_nn_op/aclnn_max_pool.h](../xla_npu/dependency/header/acl_nn_op/aclnn_max_pool.h) | 42 | 2 | 2 | 4 | 0 | 1 | 42 |
| 663 | 956 | [xla_npu/dependency/header/acl_nn_op/aclnn_max_unpool2d.h](../xla_npu/dependency/header/acl_nn_op/aclnn_max_unpool2d.h) | 44 | 2 | 2 | 4 | 0 | 1 | 44 |
| 664 | 1019 | [xla_npu/dependency/header/acl_nn_op/aclnn_max_unpool2d_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_max_unpool2d_backward.h) | 41 | 2 | 2 | 4 | 0 | 1 | 41 |
| 665 | 934 | [xla_npu/dependency/header/acl_nn_op/aclnn_max_unpool3d.h](../xla_npu/dependency/header/acl_nn_op/aclnn_max_unpool3d.h) | 45 | 2 | 2 | 4 | 0 | 1 | 45 |
| 666 | 997 | [xla_npu/dependency/header/acl_nn_op/aclnn_max_unpool3d_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_max_unpool3d_backward.h) | 42 | 2 | 2 | 4 | 0 | 1 | 42 |
| 667 | 1067 | [xla_npu/dependency/header/acl_nn_op/aclnn_mish_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_mish_backward.h) | 39 | 2 | 2 | 4 | 0 | 1 | 39 |
| 668 | 1020 | [xla_npu/dependency/header/acl_nn_op/aclnn_moe_finalize_routing_v2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_moe_finalize_routing_v2.h) | 41 | 2 | 2 | 4 | 0 | 1 | 41 |
| 669 | 957 | [xla_npu/dependency/header/acl_nn_op/aclnn_moe_finalize_routing_v2_grad.h](../xla_npu/dependency/header/acl_nn_op/aclnn_moe_finalize_routing_v2_grad.h) | 44 | 2 | 2 | 4 | 0 | 1 | 44 |
| 670 | 1068 | [xla_npu/dependency/header/acl_nn_op/aclnn_moe_init_routing_v2_grad.h](../xla_npu/dependency/header/acl_nn_op/aclnn_moe_init_routing_v2_grad.h) | 39 | 2 | 2 | 4 | 0 | 1 | 39 |
| 671 | 910 | [xla_npu/dependency/header/acl_nn_op/aclnn_moe_init_routing_v3.h](../xla_npu/dependency/header/acl_nn_op/aclnn_moe_init_routing_v3.h) | 47 | 2 | 2 | 4 | 0 | 1 | 47 |
| 672 | 1069 | [xla_npu/dependency/header/acl_nn_op/aclnn_moe_token_permute.h](../xla_npu/dependency/header/acl_nn_op/aclnn_moe_token_permute.h) | 39 | 2 | 2 | 4 | 0 | 1 | 39 |
| 673 | 1082 | [xla_npu/dependency/header/acl_nn_op/aclnn_moe_token_permute_grad.h](../xla_npu/dependency/header/acl_nn_op/aclnn_moe_token_permute_grad.h) | 38 | 2 | 2 | 4 | 0 | 1 | 38 |
| 674 | 958 | [xla_npu/dependency/header/acl_nn_op/aclnn_moe_token_unpermute.h](../xla_npu/dependency/header/acl_nn_op/aclnn_moe_token_unpermute.h) | 44 | 2 | 2 | 4 | 0 | 1 | 44 |
| 675 | 935 | [xla_npu/dependency/header/acl_nn_op/aclnn_moe_token_unpermute_grad.h](../xla_npu/dependency/header/acl_nn_op/aclnn_moe_token_unpermute_grad.h) | 45 | 2 | 2 | 4 | 0 | 1 | 45 |
| 676 | 1021 | [xla_npu/dependency/header/acl_nn_op/aclnn_multilabel_margin_loss.h](../xla_npu/dependency/header/acl_nn_op/aclnn_multilabel_margin_loss.h) | 41 | 2 | 2 | 4 | 0 | 1 | 41 |
| 677 | 1048 | [xla_npu/dependency/header/acl_nn_op/aclnn_nonzero_v2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_nonzero_v2.h) | 40 | 2 | 2 | 4 | 0 | 1 | 40 |
| 678 | 816 | [xla_npu/dependency/header/acl_nn_op/aclnn_norm_rope_concat_grad.h](../xla_npu/dependency/header/acl_nn_op/aclnn_norm_rope_concat_grad.h) | 63 | 2 | 2 | 4 | 0 | 1 | 63 |
| 679 | 902 | [xla_npu/dependency/header/acl_nn_op/aclnn_nsa_compress_attention.h](../xla_npu/dependency/header/acl_nn_op/aclnn_nsa_compress_attention.h) | 48 | 2 | 2 | 4 | 0 | 1 | 48 |
| 680 | 921 | [xla_npu/dependency/header/acl_nn_op/aclnn_nsa_selected_attention.h](../xla_npu/dependency/header/acl_nn_op/aclnn_nsa_selected_attention.h) | 46 | 2 | 2 | 4 | 0 | 1 | 46 |
| 681 | 873 | [xla_npu/dependency/header/acl_nn_op/aclnn_nsa_selected_attention_grad.h](../xla_npu/dependency/header/acl_nn_op/aclnn_nsa_selected_attention_grad.h) | 53 | 2 | 2 | 4 | 0 | 1 | 53 |
| 682 | 879 | [xla_npu/dependency/header/acl_nn_op/aclnn_nsa_selected_attention_infer.h](../xla_npu/dependency/header/acl_nn_op/aclnn_nsa_selected_attention_infer.h) | 52 | 2 | 2 | 4 | 0 | 1 | 52 |
| 683 | 1049 | [xla_npu/dependency/header/acl_nn_op/aclnn_one_hot.h](../xla_npu/dependency/header/acl_nn_op/aclnn_one_hot.h) | 40 | 2 | 2 | 4 | 0 | 1 | 40 |
| 684 | 1050 | [xla_npu/dependency/header/acl_nn_op/aclnn_ones.h](../xla_npu/dependency/header/acl_nn_op/aclnn_ones.h) | 40 | 2 | 2 | 4 | 0 | 1 | 40 |
| 685 | 1022 | [xla_npu/dependency/header/acl_nn_op/aclnn_pdist_forward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_pdist_forward.h) | 41 | 2 | 2 | 4 | 0 | 1 | 41 |
| 686 | 978 | [xla_npu/dependency/header/acl_nn_op/aclnn_permute.h](../xla_npu/dependency/header/acl_nn_op/aclnn_permute.h) | 43 | 2 | 2 | 4 | 0 | 1 | 43 |
| 687 | 1051 | [xla_npu/dependency/header/acl_nn_op/aclnn_prelu.h](../xla_npu/dependency/header/acl_nn_op/aclnn_prelu.h) | 40 | 2 | 2 | 4 | 0 | 1 | 40 |
| 688 | 922 | [xla_npu/dependency/header/acl_nn_op/aclnn_quantized_batch_norm.h](../xla_npu/dependency/header/acl_nn_op/aclnn_quantized_batch_norm.h) | 46 | 2 | 2 | 4 | 0 | 1 | 46 |
| 689 | 913 | [xla_npu/dependency/header/acl_nn_op/aclnn_randperm.h](../xla_npu/dependency/header/acl_nn_op/aclnn_randperm.h) | 47 | 2 | 2 | 4 | 0 | 1 | 47 |
| 690 | 1053 | [xla_npu/dependency/header/acl_nn_op/aclnn_reduce_nansum.h](../xla_npu/dependency/header/acl_nn_op/aclnn_reduce_nansum.h) | 40 | 2 | 2 | 4 | 0 | 1 | 40 |
| 691 | 1070 | [xla_npu/dependency/header/acl_nn_op/aclnn_reduce_sum.h](../xla_npu/dependency/header/acl_nn_op/aclnn_reduce_sum.h) | 39 | 2 | 2 | 4 | 0 | 1 | 39 |
| 692 | 1023 | [xla_npu/dependency/header/acl_nn_op/aclnn_repeat.h](../xla_npu/dependency/header/acl_nn_op/aclnn_repeat.h) | 41 | 2 | 2 | 4 | 0 | 1 | 41 |
| 693 | 1054 | [xla_npu/dependency/header/acl_nn_op/aclnn_s_where.h](../xla_npu/dependency/header/acl_nn_op/aclnn_s_where.h) | 40 | 2 | 2 | 4 | 0 | 1 | 40 |
| 694 | 1055 | [xla_npu/dependency/header/acl_nn_op/aclnn_scatter_nd.h](../xla_npu/dependency/header/acl_nn_op/aclnn_scatter_nd.h) | 40 | 2 | 2 | 4 | 0 | 1 | 40 |
| 695 | 1024 | [xla_npu/dependency/header/acl_nn_op/aclnn_sign.h](../xla_npu/dependency/header/acl_nn_op/aclnn_sign.h) | 41 | 2 | 2 | 4 | 0 | 1 | 41 |
| 696 | 1056 | [xla_npu/dependency/header/acl_nn_op/aclnn_silu.h](../xla_npu/dependency/header/acl_nn_op/aclnn_silu.h) | 40 | 2 | 2 | 4 | 0 | 1 | 40 |
| 697 | 999 | [xla_npu/dependency/header/acl_nn_op/aclnn_slice.h](../xla_npu/dependency/header/acl_nn_op/aclnn_slice.h) | 42 | 2 | 2 | 4 | 0 | 1 | 42 |
| 698 | 1057 | [xla_npu/dependency/header/acl_nn_op/aclnn_slice_v2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_slice_v2.h) | 40 | 2 | 2 | 4 | 0 | 1 | 40 |
| 699 | 1025 | [xla_npu/dependency/header/acl_nn_op/aclnn_smooth_l1_loss.h](../xla_npu/dependency/header/acl_nn_op/aclnn_smooth_l1_loss.h) | 41 | 2 | 2 | 4 | 0 | 1 | 41 |
| 700 | 1090 | [xla_npu/dependency/header/acl_nn_op/aclnn_softmax.h](../xla_npu/dependency/header/acl_nn_op/aclnn_softmax.h) | 37 | 2 | 2 | 4 | 0 | 1 | 37 |
| 701 | 1026 | [xla_npu/dependency/header/acl_nn_op/aclnn_softmax_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_softmax_backward.h) | 41 | 2 | 2 | 4 | 0 | 1 | 41 |
| 702 | 1027 | [xla_npu/dependency/header/acl_nn_op/aclnn_softplus_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_softplus_backward.h) | 41 | 2 | 2 | 4 | 0 | 1 | 41 |
| 703 | 1028 | [xla_npu/dependency/header/acl_nn_op/aclnn_stack.h](../xla_npu/dependency/header/acl_nn_op/aclnn_stack.h) | 41 | 2 | 2 | 4 | 0 | 1 | 41 |
| 704 | 1000 | [xla_npu/dependency/header/acl_nn_op/aclnn_std_mean_correction.h](../xla_npu/dependency/header/acl_nn_op/aclnn_std_mean_correction.h) | 42 | 2 | 2 | 4 | 0 | 1 | 42 |
| 705 | 903 | [xla_npu/dependency/header/acl_nn_op/aclnn_sync_batch_norm_gather_stats.h](../xla_npu/dependency/header/acl_nn_op/aclnn_sync_batch_norm_gather_stats.h) | 48 | 2 | 2 | 4 | 0 | 1 | 48 |
| 706 | 936 | [xla_npu/dependency/header/acl_nn_op/aclnn_trace.h](../xla_npu/dependency/header/acl_nn_op/aclnn_trace.h) | 45 | 2 | 2 | 4 | 0 | 1 | 45 |
| 707 | 1001 | [xla_npu/dependency/header/acl_nn_op/aclnn_upsample_bicubic_2d.h](../xla_npu/dependency/header/acl_nn_op/aclnn_upsample_bicubic_2d.h) | 42 | 2 | 2 | 4 | 0 | 1 | 42 |
| 708 | 1002 | [xla_npu/dependency/header/acl_nn_op/aclnn_upsample_bicubic_2d_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_upsample_bicubic_2d_backward.h) | 42 | 2 | 2 | 4 | 0 | 1 | 42 |
| 709 | 959 | [xla_npu/dependency/header/acl_nn_op/aclnn_upsample_bilinear_2d.h](../xla_npu/dependency/header/acl_nn_op/aclnn_upsample_bilinear_2d.h) | 44 | 2 | 2 | 4 | 0 | 1 | 44 |
| 710 | 960 | [xla_npu/dependency/header/acl_nn_op/aclnn_upsample_bilinear_2d_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_upsample_bilinear_2d_backward.h) | 44 | 2 | 2 | 4 | 0 | 1 | 44 |
| 711 | 979 | [xla_npu/dependency/header/acl_nn_op/aclnn_upsample_linear_1d.h](../xla_npu/dependency/header/acl_nn_op/aclnn_upsample_linear_1d.h) | 43 | 2 | 2 | 4 | 0 | 1 | 43 |
| 712 | 961 | [xla_npu/dependency/header/acl_nn_op/aclnn_upsample_linear_1d_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_upsample_linear_1d_backward.h) | 44 | 2 | 2 | 4 | 0 | 1 | 44 |
| 713 | 1058 | [xla_npu/dependency/header/acl_nn_op/aclnn_upsample_nearest_1d.h](../xla_npu/dependency/header/acl_nn_op/aclnn_upsample_nearest_1d.h) | 40 | 2 | 2 | 4 | 0 | 1 | 40 |
| 714 | 962 | [xla_npu/dependency/header/acl_nn_op/aclnn_upsample_nearest_1d_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_upsample_nearest_1d_backward.h) | 44 | 2 | 2 | 4 | 0 | 1 | 44 |
| 715 | 937 | [xla_npu/dependency/header/acl_nn_op/aclnn_upsample_nearest_3d.h](../xla_npu/dependency/header/acl_nn_op/aclnn_upsample_nearest_3d.h) | 45 | 2 | 2 | 4 | 0 | 1 | 45 |
| 716 | 923 | [xla_npu/dependency/header/acl_nn_op/aclnn_upsample_nearest_3d_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_upsample_nearest_3d_backward.h) | 46 | 2 | 2 | 4 | 0 | 1 | 46 |
| 717 | 938 | [xla_npu/dependency/header/acl_nn_op/aclnn_upsample_nearest_exact3d.h](../xla_npu/dependency/header/acl_nn_op/aclnn_upsample_nearest_exact3d.h) | 45 | 2 | 2 | 4 | 0 | 1 | 45 |
| 718 | 939 | [xla_npu/dependency/header/acl_nn_op/aclnn_upsample_nearest_exact3d_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_upsample_nearest_exact3d_backward.h) | 45 | 2 | 2 | 4 | 0 | 1 | 45 |
| 719 | 940 | [xla_npu/dependency/header/acl_nn_op/aclnn_upsample_trilinear_3d.h](../xla_npu/dependency/header/acl_nn_op/aclnn_upsample_trilinear_3d.h) | 45 | 2 | 2 | 4 | 0 | 1 | 45 |
| 720 | 980 | [xla_npu/dependency/header/acl_nn_op/aclnn_upsample_trilinear_3d_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_upsample_trilinear_3d_backward.h) | 43 | 2 | 2 | 4 | 0 | 1 | 43 |
| 721 | 1059 | [xla_npu/dependency/header/acl_nn_op/aclnn_var_mean.h](../xla_npu/dependency/header/acl_nn_op/aclnn_var_mean.h) | 40 | 2 | 2 | 4 | 0 | 1 | 40 |
| 722 | 888 | [xla_npu/dependency/header/acl_nn_op/aclnn_weight_quant_batch_matmul.h](../xla_npu/dependency/header/acl_nn_op/aclnn_weight_quant_batch_matmul.h) | 50 | 2 | 2 | 4 | 0 | 1 | 50 |
| 723 | 981 | [xla_npu/dependency/header/acl_nn_op/aclnn_weight_quant_batch_matmul_v3.h](../xla_npu/dependency/header/acl_nn_op/aclnn_weight_quant_batch_matmul_v3.h) | 43 | 2 | 2 | 4 | 0 | 1 | 43 |
| 724 | 1060 | [xla_npu/dependency/header/acl_nn_op/aclnn_x_log_y_scalar_self.h](../xla_npu/dependency/header/acl_nn_op/aclnn_x_log_y_scalar_self.h) | 40 | 2 | 2 | 4 | 0 | 1 | 40 |
| 725 | 817 | [xla_npu/dependency/header/metadef/exe_graph/runtime/kernel_run_context.h](../xla_npu/dependency/header/metadef/exe_graph/runtime/kernel_run_context.h) | 63 | 2 | 2 | 4 | 0 | 1 | 63 |
| 726 | 1112 | [xla_npu/dependency/interface/acl_nn_interface.cc](../xla_npu/dependency/interface/acl_nn_interface.cc) | 34 | 4 | 0 | 4 | 4 | 2 | 38 |
| 727 | 40 | [xla_npu/dependency/stub/ge_api_stub.cc](../xla_npu/dependency/stub/ge_api_stub.cc) | 595 | 2 | 2 | 4 | 0 | 1 | 595 |
| 728 | 206 | [xla_npu/pjrt/air/mair/BATCHMATMUL_OPTIMIZATION.md](../xla_npu/pjrt/air/mair/BATCHMATMUL_OPTIMIZATION.md) | 188 | 2 | 2 | 4 | 0 | 1 | 188 |
| 729 | 842 | [xla_npu/pjrt/air/mair/BUILD](../xla_npu/pjrt/air/mair/BUILD) | 60 | 2 | 2 | 4 | 0 | 1 | 60 |
| 730 | 632 | [xla_npu/pjrt/air/mair/translations/export_graphdef.h](../xla_npu/pjrt/air/mair/translations/export_graphdef.h) | 96 | 2 | 2 | 4 | 0 | 1 | 96 |
| 731 | 27 | [xla_npu/pjrt/air/npu/npu_client.cc](../xla_npu/pjrt/air/npu/npu_client.cc) | 811 | 2 | 2 | 4 | 0 | 1 | 811 |
| 732 | 672 | [xla_npu/pjrt/c/BUILD](../xla_npu/pjrt/c/BUILD) | 90 | 2 | 2 | 4 | 0 | 1 | 90 |
| 733 | 441 | [xla_npu/service/convert_to_afir/BUILD](../xla_npu/service/convert_to_afir/BUILD) | 119 | 3 | 1 | 4 | 2 | 2 | 121 |
| 734 | 447 | [xla_npu/service/npu_buffer_allocations.cc](../xla_npu/service/npu_buffer_allocations.cc) | 118 | 1 | 3 | 4 | -2 | 1 | 116 |
| 735 | 189 | [xla_npu/service/npu_compiler.h](../xla_npu/service/npu_compiler.h) | 198 | 2 | 2 | 4 | 0 | 1 | 198 |
| 736 | 247 | [xla_npu/stream_executor/BUILD](../xla_npu/stream_executor/BUILD) | 171 | 3 | 1 | 4 | 2 | 2 | 173 |
| 737 | 61 | [xla_npu/dependency/external/afir/Ascend-MLIR/lib/Dialect/AFIR/Ops/Basic/math.cpp](../xla_npu/dependency/external/afir/Ascend-MLIR/lib/Dialect/AFIR/Ops/Basic/math.cpp) | 437 | 1 | 2 | 3 | -1 | 1 | 436 |
| 738 | 742 | [xla_npu/common/BUILD](../xla_npu/common/BUILD) | 83 | 1 | 1 | 2 | 0 | 1 | 83 |
| 739 | 1086 | [xla_npu/common/npu_operation.h](../xla_npu/common/npu_operation.h) | 39 | 1 | 1 | 2 | 0 | 1 | 39 |
| 740 | 855 | [xla_npu/dependency/external/afir/Ascend-MLIR/include/Dialect/AFIR/CMakeLists.txt](../xla_npu/dependency/external/afir/Ascend-MLIR/include/Dialect/AFIR/CMakeLists.txt) | 58 | 1 | 1 | 2 | 0 | 1 | 58 |
| 741 | 105 | [xla_npu/dependency/external/npuir/BUILD](../xla_npu/dependency/external/npuir/BUILD) | 289 | 1 | 1 | 2 | 0 | 1 | 289 |
| 742 | 37 | [xla_npu/dependency/header/acl_nn/opdev/op_arg_def.h](../xla_npu/dependency/header/acl_nn/opdev/op_arg_def.h) | 651 | 1 | 1 | 2 | 0 | 1 | 651 |
| 743 | 64 | [xla_npu/dependency/header/acl_nn/opdev/op_dfx.h](../xla_npu/dependency/header/acl_nn/opdev/op_dfx.h) | 421 | 1 | 1 | 2 | 0 | 1 | 421 |
| 744 | 900 | [xla_npu/dependency/header/acl_nn_op/aclnn_deformable_conv2d.h](../xla_npu/dependency/header/acl_nn_op/aclnn_deformable_conv2d.h) | 50 | 1 | 1 | 2 | 0 | 1 | 50 |
| 745 | 1045 | [xla_npu/dependency/header/acl_nn_op/aclnn_dropout.h](../xla_npu/dependency/header/acl_nn_op/aclnn_dropout.h) | 42 | 1 | 1 | 2 | 0 | 1 | 42 |
| 746 | 1011 | [xla_npu/dependency/header/acl_nn_op/aclnn_dropout_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_dropout_backward.h) | 43 | 1 | 1 | 2 | 0 | 1 | 43 |
| 747 | 1012 | [xla_npu/dependency/header/acl_nn_op/aclnn_dropout_do_mask.h](../xla_npu/dependency/header/acl_nn_op/aclnn_dropout_do_mask.h) | 43 | 1 | 1 | 2 | 0 | 1 | 43 |
| 748 | 1164 | [xla_npu/pjrt/air/mair/ir/CMakeLists.txt](../xla_npu/pjrt/air/mair/ir/CMakeLists.txt) | 29 | 1 | 1 | 2 | 0 | 1 | 29 |
| 749 | 33 | [xla_npu/pjrt/air/mair/ir/mair_ops.td](../xla_npu/pjrt/air/mair/ir/mair_ops.td) | 710 | 1 | 1 | 2 | 0 | 1 | 710 |
| 750 | 1097 | [xla_npu/pjrt/air/mair/stablehlo2mair_converter.cpp](../xla_npu/pjrt/air/mair/stablehlo2mair_converter.cpp) | 38 | 1 | 1 | 2 | 0 | 1 | 38 |
| 751 | 1157 | [xla_npu/pjrt/air/mair/stablehlo2mair_converter.h](../xla_npu/pjrt/air/mair/stablehlo2mair_converter.h) | 30 | 1 | 1 | 2 | 0 | 1 | 30 |
| 752 | 1205 | [xla_npu/pjrt/air/mair/transforms/CMakeLists.txt](../xla_npu/pjrt/air/mair/transforms/CMakeLists.txt) | 20 | 1 | 1 | 2 | 0 | 1 | 20 |
| 753 | 176 | [xla_npu/pjrt/air/npu/BUILD](../xla_npu/pjrt/air/npu/BUILD) | 209 | 1 | 1 | 2 | 0 | 1 | 209 |
| 754 | 229 | [xla_npu/pjrt/air/npu/service/npu_executable.cc](../xla_npu/pjrt/air/npu/service/npu_executable.cc) | 179 | 1 | 1 | 2 | 0 | 1 | 179 |
| 755 | 1206 | [xla_npu/pjrt/npu/BUILD](../xla_npu/pjrt/npu/BUILD) | 20 | 1 | 1 | 2 | 0 | 1 | 20 |
| 756 | 892 | [xla_npu/pjrt/npu/npu_pjrt_stream_executor_client.h](../xla_npu/pjrt/npu/npu_pjrt_stream_executor_client.h) | 52 | 1 | 1 | 2 | 0 | 1 | 52 |
| 757 | 1073 | [xla_npu/service/kernel/npu_kernel_config.h](../xla_npu/service/kernel/npu_kernel_config.h) | 41 | 1 | 1 | 2 | 0 | 1 | 41 |
| 758 | 145 | [xla_npu/service/npu_compile_module_to_llvm_ir.cc](../xla_npu/service/npu_compile_module_to_llvm_ir.cc) | 238 | 2 | 0 | 2 | 2 | 1 | 240 |
| 759 | 231 | [xla_npu/dependency/external/afir/BUILD](../xla_npu/dependency/external/afir/BUILD) | 177 | 1 | 0 | 1 | 1 | 1 | 178 |

### 8.2 后续无改动的文件

| 后续名次 | 全史名次 | 文件 | 首次 | 后续新增 | 后续删除 | 后续累计 | 净增 | 次数 | HEAD 行数 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 760 | 1106 | [xla_npu/backend/codegen/npu_custom.h](../xla_npu/backend/codegen/npu_custom.h) | 38 | 0 | 0 | 0 | 0 | 0 | 38 |
| 761 | 1211 | [xla_npu/backend/codegen/npu_fusion_emitter.cc](../xla_npu/backend/codegen/npu_fusion_emitter.cc) | 18 | 0 | 0 | 0 | 0 | 0 | 18 |
| 762 | 926 | [xla_npu/backend/codegen/npu_fusion_emitter.h](../xla_npu/backend/codegen/npu_fusion_emitter.h) | 49 | 0 | 0 | 0 | 0 | 0 | 49 |
| 763 | 578 | [xla_npu/backend/runtime/npu_call_thunk.cc](../xla_npu/backend/runtime/npu_call_thunk.cc) | 105 | 0 | 0 | 0 | 0 | 0 | 105 |
| 764 | 830 | [xla_npu/backend/runtime/npu_call_thunk.h](../xla_npu/backend/runtime/npu_call_thunk.h) | 65 | 0 | 0 | 0 | 0 | 0 | 65 |
| 765 | 831 | [xla_npu/backend/runtime/npu_sequential_thunk.h](../xla_npu/backend/runtime/npu_sequential_thunk.h) | 65 | 0 | 0 | 0 | 0 | 0 | 65 |
| 766 | 1091 | [xla_npu/backend/runtime/npu_thunk_id.h](../xla_npu/backend/runtime/npu_thunk_id.h) | 40 | 0 | 0 | 0 | 0 | 0 | 40 |
| 767 | 984 | [xla_npu/backend/runtime/npu_thunk_pass_pipeline.cc](../xla_npu/backend/runtime/npu_thunk_pass_pipeline.cc) | 46 | 0 | 0 | 0 | 0 | 0 | 46 |
| 768 | 750 | [xla_npu/backend/runtime/npu_thunk_pass_pipeline.h](../xla_npu/backend/runtime/npu_thunk_pass_pipeline.h) | 84 | 0 | 0 | 0 | 0 | 0 | 84 |
| 769 | 945 | [xla_npu/client/BUILD](../xla_npu/client/BUILD) | 48 | 0 | 0 | 0 | 0 | 0 | 48 |
| 770 | 1197 | [xla_npu/client/npu_module.h](../xla_npu/client/npu_module.h) | 25 | 0 | 0 | 0 | 0 | 0 | 25 |
| 771 | 1169 | [xla_npu/client/npu_stream_client.h](../xla_npu/client/npu_stream_client.h) | 29 | 0 | 0 | 0 | 0 | 0 | 29 |
| 772 | 1202 | [xla_npu/common/npu_constant.cc](../xla_npu/common/npu_constant.cc) | 23 | 0 | 0 | 0 | 0 | 0 | 23 |
| 773 | 504 | [xla_npu/common/npu_operation.cc](../xla_npu/common/npu_operation.cc) | 112 | 0 | 0 | 0 | 0 | 0 | 112 |
| 774 | 1226 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/CMakeLists.txt](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/CMakeLists.txt) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 775 | 1216 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/API/CMakeLists.txt](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/API/CMakeLists.txt) | 16 | 0 | 0 | 0 | 0 | 0 | 16 |
| 776 | 72 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/API/Types.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/API/Types.td) | 393 | 0 | 0 | 0 | 0 | 0 | 393 |
| 777 | 1223 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/CMakeLists.txt](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/CMakeLists.txt) | 10 | 0 | 0 | 0 | 0 | 0 | 10 |
| 778 | 1224 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/CMakeLists.txt](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/CMakeLists.txt) | 10 | 0 | 0 | 0 | 0 | 0 | 10 |
| 779 | 917 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Adv/Activation.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Adv/Activation.td) | 50 | 0 | 0 | 0 | 0 | 0 | 50 |
| 780 | 905 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Adv/Kfc.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Adv/Kfc.td) | 51 | 0 | 0 | 0 | 0 | 0 | 51 |
| 781 | 676 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Adv/Math.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Adv/Math.td) | 93 | 0 | 0 | 0 | 0 | 0 | 93 |
| 782 | 89 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Adv/Matmul.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Adv/Matmul.td) | 329 | 0 | 0 | 0 | 0 | 0 | 329 |
| 783 | 1139 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Adv/Normalization.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Adv/Normalization.td) | 34 | 0 | 0 | 0 | 0 | 0 | 34 |
| 784 | 1115 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Adv/Quantization.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Adv/Quantization.td) | 37 | 0 | 0 | 0 | 0 | 0 | 37 |
| 785 | 865 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Adv/Sort.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Adv/Sort.td) | 58 | 0 | 0 | 0 | 0 | 0 | 58 |
| 786 | 787 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Asc.h](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Asc.h) | 74 | 0 | 0 | 0 | 0 | 0 | 74 |
| 787 | 872 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/AscFuncTrait.h](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/AscFuncTrait.h) | 57 | 0 | 0 | 0 | 0 | 0 | 57 |
| 788 | 82 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Base.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Base.td) | 349 | 0 | 0 | 0 | 0 | 0 | 349 |
| 789 | 767 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/Common.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/Common.td) | 80 | 0 | 0 | 0 | 0 | 0 | 80 |
| 790 | 1036 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpAipp.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpAipp.td) | 44 | 0 | 0 | 0 | 0 | 0 | 44 |
| 791 | 875 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpAtomic.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpAtomic.td) | 56 | 0 | 0 | 0 | 0 | 0 | 56 |
| 792 | 846 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpBlockSync.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpBlockSync.td) | 62 | 0 | 0 | 0 | 0 | 0 | 62 |
| 793 | 1173 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpCache.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpCache.td) | 28 | 0 | 0 | 0 | 0 | 0 | 28 |
| 794 | 1174 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpCommon.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpCommon.td) | 28 | 0 | 0 | 0 | 0 | 0 | 28 |
| 795 | 1175 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpConv2d.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpConv2d.td) | 28 | 0 | 0 | 0 | 0 | 0 | 28 |
| 796 | 853 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpCopy.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpCopy.td) | 60 | 0 | 0 | 0 | 0 | 0 | 60 |
| 797 | 832 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpDataCache.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpDataCache.td) | 65 | 0 | 0 | 0 | 0 | 0 | 65 |
| 798 | 1140 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpDataConversion.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpDataConversion.td) | 34 | 0 | 0 | 0 | 0 | 0 | 34 |
| 799 | 306 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpDataCopy.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpDataCopy.td) | 155 | 0 | 0 | 0 | 0 | 0 | 155 |
| 800 | 1176 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpDetermineComputeSync.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpDetermineComputeSync.td) | 28 | 0 | 0 | 0 | 0 | 0 | 28 |
| 801 | 857 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpDumpTensor.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpDumpTensor.td) | 59 | 0 | 0 | 0 | 0 | 0 | 59 |
| 802 | 1143 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpFixpipe.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpFixpipe.td) | 33 | 0 | 0 | 0 | 0 | 0 | 33 |
| 803 | 1177 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpGemm.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpGemm.td) | 28 | 0 | 0 | 0 | 0 | 0 | 28 |
| 804 | 1087 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpHcclContext.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpHcclContext.td) | 41 | 0 | 0 | 0 | 0 | 0 | 41 |
| 805 | 1178 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpLimits.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpLimits.td) | 28 | 0 | 0 | 0 | 0 | 0 | 28 |
| 806 | 1179 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpListTensor.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpListTensor.td) | 28 | 0 | 0 | 0 | 0 | 0 | 28 |
| 807 | 626 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpMm.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpMm.td) | 100 | 0 | 0 | 0 | 0 | 0 | 100 |
| 808 | 946 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpProposal.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpProposal.td) | 48 | 0 | 0 | 0 | 0 | 0 | 48 |
| 809 | 918 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpScalar.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpScalar.td) | 50 | 0 | 0 | 0 | 0 | 0 | 50 |
| 810 | 1180 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpSetAtomic.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpSetAtomic.td) | 28 | 0 | 0 | 0 | 0 | 0 | 28 |
| 811 | 1107 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpSwapMem.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpSwapMem.td) | 38 | 0 | 0 | 0 | 0 | 0 | 38 |
| 812 | 782 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpSysVar.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpSysVar.td) | 75 | 0 | 0 | 0 | 0 | 0 | 75 |
| 813 | 1004 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpVecBilinearInterpolation.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpVecBilinearInterpolation.td) | 45 | 0 | 0 | 0 | 0 | 0 | 45 |
| 814 | 985 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpVecBinary.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpVecBinary.td) | 46 | 0 | 0 | 0 | 0 | 0 | 46 |
| 815 | 1128 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpVecBinaryScalar.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpVecBinaryScalar.td) | 36 | 0 | 0 | 0 | 0 | 0 | 36 |
| 816 | 1100 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpVecBrcb.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpVecBrcb.td) | 39 | 0 | 0 | 0 | 0 | 0 | 39 |
| 817 | 395 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpVecCmpsel.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpVecCmpsel.td) | 130 | 0 | 0 | 0 | 0 | 0 | 130 |
| 818 | 1181 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpVecCreatevecindex.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpVecCreatevecindex.td) | 28 | 0 | 0 | 0 | 0 | 0 | 28 |
| 819 | 967 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpVecDuplicate.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpVecDuplicate.td) | 47 | 0 | 0 | 0 | 0 | 0 | 47 |
| 820 | 849 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpVecGather.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpVecGather.td) | 61 | 0 | 0 | 0 | 0 | 0 | 61 |
| 821 | 898 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpVecGatherMask.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpVecGatherMask.td) | 52 | 0 | 0 | 0 | 0 | 0 | 52 |
| 822 | 1166 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpVecMulCast.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpVecMulCast.td) | 30 | 0 | 0 | 0 | 0 | 0 | 30 |
| 823 | 75 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpVecReduce.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpVecReduce.td) | 374 | 0 | 0 | 0 | 0 | 0 | 374 |
| 824 | 809 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpVecScatter.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpVecScatter.td) | 68 | 0 | 0 | 0 | 0 | 0 | 68 |
| 825 | 1182 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpVecTensor.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpVecTensor.td) | 28 | 0 | 0 | 0 | 0 | 0 | 28 |
| 826 | 1167 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpVecTernaryScalar.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpVecTernaryScalar.td) | 30 | 0 | 0 | 0 | 0 | 0 | 30 |
| 827 | 1183 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpVecTranspose.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpVecTranspose.td) | 28 | 0 | 0 | 0 | 0 | 0 | 28 |
| 828 | 1108 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpVecUnary.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpVecUnary.td) | 38 | 0 | 0 | 0 | 0 | 0 | 38 |
| 829 | 752 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpVecVconv.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpVecVconv.td) | 83 | 0 | 0 | 0 | 0 | 0 | 83 |
| 830 | 1184 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpVecVpadding.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpVecVpadding.td) | 28 | 0 | 0 | 0 | 0 | 0 | 28 |
| 831 | 1185 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/ProfTrace.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/ProfTrace.td) | 28 | 0 | 0 | 0 | 0 | 0 | 28 |
| 832 | 947 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/CMakeLists.txt](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/CMakeLists.txt) | 48 | 0 | 0 | 0 | 0 | 0 | 48 |
| 833 | 49 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Core/Attributes.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Core/Attributes.td) | 507 | 0 | 0 | 0 | 0 | 0 | 507 |
| 834 | 893 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Core/Base.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Core/Base.td) | 53 | 0 | 0 | 0 | 0 | 0 | 53 |
| 835 | 1088 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Core/Interfaces.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Core/Interfaces.td) | 41 | 0 | 0 | 0 | 0 | 0 | 41 |
| 836 | 96 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Core/Tensor.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Core/Tensor.td) | 313 | 0 | 0 | 0 | 0 | 0 | 313 |
| 837 | 198 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Core/Types.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Core/Types.td) | 194 | 0 | 0 | 0 | 0 | 0 | 194 |
| 838 | 1075 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Dialect.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Dialect.td) | 42 | 0 | 0 | 0 | 0 | 0 | 42 |
| 839 | 876 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Fwk/TBuf.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Fwk/TBuf.td) | 56 | 0 | 0 | 0 | 0 | 0 | 56 |
| 840 | 788 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Fwk/TBufPool.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Fwk/TBufPool.td) | 74 | 0 | 0 | 0 | 0 | 0 | 74 |
| 841 | 480 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Fwk/TPipe.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Fwk/TPipe.td) | 115 | 0 | 0 | 0 | 0 | 0 | 115 |
| 842 | 188 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Fwk/TQue.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Fwk/TQue.td) | 202 | 0 | 0 | 0 | 0 | 0 | 202 |
| 843 | 219 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Interfaces.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Interfaces.td) | 186 | 0 | 0 | 0 | 0 | 0 | 186 |
| 844 | 330 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Ops.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/IR/Ops.td) | 148 | 0 | 0 | 0 | 0 | 0 | 148 |
| 845 | 1217 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/Transforms/CMakeLists.txt](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/Transforms/CMakeLists.txt) | 13 | 0 | 0 | 0 | 0 | 0 | 13 |
| 846 | 906 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/Transforms/Passes.h](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/Transforms/Passes.h) | 51 | 0 | 0 | 0 | 0 | 0 | 51 |
| 847 | 533 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/Transforms/Passes.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/Transforms/Passes.td) | 110 | 0 | 0 | 0 | 0 | 0 | 110 |
| 848 | 1116 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/Utils/Attributes.h](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/Utils/Attributes.h) | 37 | 0 | 0 | 0 | 0 | 0 | 37 |
| 849 | 858 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/Utils/Utils.h](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Asc/Utils/Utils.h) | 59 | 0 | 0 | 0 | 0 | 0 | 59 |
| 850 | 1225 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/CMakeLists.txt](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/CMakeLists.txt) | 10 | 0 | 0 | 0 | 0 | 0 | 10 |
| 851 | 1227 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/EmitAsc/CMakeLists.txt](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/EmitAsc/CMakeLists.txt) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 852 | 948 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/EmitAsc/IR/Attributes.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/EmitAsc/IR/Attributes.td) | 48 | 0 | 0 | 0 | 0 | 0 | 48 |
| 853 | 1170 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/EmitAsc/IR/CMakeLists.txt](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/EmitAsc/IR/CMakeLists.txt) | 29 | 0 | 0 | 0 | 0 | 0 | 29 |
| 854 | 1117 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/EmitAsc/IR/Dialect.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/EmitAsc/IR/Dialect.td) | 37 | 0 | 0 | 0 | 0 | 0 | 37 |
| 855 | 968 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/EmitAsc/IR/EmitAsc.h](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/EmitAsc/IR/EmitAsc.h) | 47 | 0 | 0 | 0 | 0 | 0 | 47 |
| 856 | 236 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/EmitAsc/IR/Ops.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/EmitAsc/IR/Ops.td) | 177 | 0 | 0 | 0 | 0 | 0 | 177 |
| 857 | 1154 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/EmitAsc/IR/Types.td](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/EmitAsc/IR/Types.td) | 32 | 0 | 0 | 0 | 0 | 0 | 32 |
| 858 | 1162 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/EmitAsc/Utils/Attributes.h](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/EmitAsc/Utils/Attributes.h) | 31 | 0 | 0 | 0 | 0 | 0 | 31 |
| 859 | 777 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Utils/ConstantOpBuilder.h](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Utils/ConstantOpBuilder.h) | 76 | 0 | 0 | 0 | 0 | 0 | 76 |
| 860 | 866 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Utils/Inlining.h](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Utils/Inlining.h) | 58 | 0 | 0 | 0 | 0 | 0 | 58 |
| 861 | 927 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Utils/Registration.h](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Utils/Registration.h) | 49 | 0 | 0 | 0 | 0 | 0 | 49 |
| 862 | 1076 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Utils/Utils.h](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Dialect/Utils/Utils.h) | 42 | 0 | 0 | 0 | 0 | 0 | 42 |
| 863 | 1163 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Adv/Activation.h](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Adv/Activation.h) | 31 | 0 | 0 | 0 | 0 | 0 | 31 |
| 864 | 1101 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Adv/Kfc.h](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Adv/Kfc.h) | 39 | 0 | 0 | 0 | 0 | 0 | 39 |
| 865 | 500 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Adv/Math.h](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Adv/Math.h) | 113 | 0 | 0 | 0 | 0 | 0 | 113 |
| 866 | 1129 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Adv/Matmul.h](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Adv/Matmul.h) | 36 | 0 | 0 | 0 | 0 | 0 | 36 |
| 867 | 1171 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Adv/Normalization.h](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Adv/Normalization.h) | 29 | 0 | 0 | 0 | 0 | 0 | 29 |
| 868 | 1172 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Adv/Quantization.h](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Adv/Quantization.h) | 29 | 0 | 0 | 0 | 0 | 0 | 29 |
| 869 | 1144 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Basic/Aipp.h](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Basic/Aipp.h) | 33 | 0 | 0 | 0 | 0 | 0 | 33 |
| 870 | 1136 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Basic/BlockSync.h](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Basic/BlockSync.h) | 35 | 0 | 0 | 0 | 0 | 0 | 35 |
| 871 | 1092 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Basic/DataConversion.h](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Basic/DataConversion.h) | 40 | 0 | 0 | 0 | 0 | 0 | 40 |
| 872 | 1118 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Basic/DataCopy.h](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Basic/DataCopy.h) | 37 | 0 | 0 | 0 | 0 | 0 | 37 |
| 873 | 1145 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Basic/DumpTensor.h](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Basic/DumpTensor.h) | 33 | 0 | 0 | 0 | 0 | 0 | 33 |
| 874 | 850 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Basic/OtherOps.h](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Basic/OtherOps.h) | 61 | 0 | 0 | 0 | 0 | 0 | 61 |
| 875 | 1146 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Basic/Scalar.h](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Basic/Scalar.h) | 33 | 0 | 0 | 0 | 0 | 0 | 33 |
| 876 | 1119 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Basic/SwapMem.h](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Basic/SwapMem.h) | 37 | 0 | 0 | 0 | 0 | 0 | 37 |
| 877 | 1137 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Basic/SysVar.h](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Basic/SysVar.h) | 35 | 0 | 0 | 0 | 0 | 0 | 35 |
| 878 | 1147 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Basic/VecBilinearInterpolation.h](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Basic/VecBilinearInterpolation.h) | 33 | 0 | 0 | 0 | 0 | 0 | 33 |
| 879 | 280 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Basic/VecBinary.h](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Basic/VecBinary.h) | 165 | 0 | 0 | 0 | 0 | 0 | 165 |
| 880 | 768 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Basic/VecBinaryScalar.h](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Basic/VecBinaryScalar.h) | 80 | 0 | 0 | 0 | 0 | 0 | 80 |
| 881 | 789 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Basic/VecCmpsel.h](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Basic/VecCmpsel.h) | 74 | 0 | 0 | 0 | 0 | 0 | 74 |
| 882 | 1120 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Basic/VecDuplicate.h](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Basic/VecDuplicate.h) | 37 | 0 | 0 | 0 | 0 | 0 | 37 |
| 883 | 1063 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Basic/VecGather.h](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Basic/VecGather.h) | 43 | 0 | 0 | 0 | 0 | 0 | 43 |
| 884 | 1148 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Basic/VecGatherMask.h](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Basic/VecGatherMask.h) | 33 | 0 | 0 | 0 | 0 | 0 | 33 |
| 885 | 771 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Basic/VecReduce.h](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Basic/VecReduce.h) | 78 | 0 | 0 | 0 | 0 | 0 | 78 |
| 886 | 1149 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Basic/VecScatter.h](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Basic/VecScatter.h) | 33 | 0 | 0 | 0 | 0 | 0 | 33 |
| 887 | 725 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Basic/VecTernaryScalar.h](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Basic/VecTernaryScalar.h) | 87 | 0 | 0 | 0 | 0 | 0 | 87 |
| 888 | 666 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Basic/VecUnary.h](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Basic/VecUnary.h) | 94 | 0 | 0 | 0 | 0 | 0 | 94 |
| 889 | 778 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Basic/VecVconv.h](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Basic/VecVconv.h) | 76 | 0 | 0 | 0 | 0 | 0 | 76 |
| 890 | 149 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/CodeEmitter.h](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/CodeEmitter.h) | 232 | 0 | 0 | 0 | 0 | 0 | 232 |
| 891 | 331 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Common.h](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Common.h) | 148 | 0 | 0 | 0 | 0 | 0 | 148 |
| 892 | 1121 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Core/GlobalTensor.h](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Core/GlobalTensor.h) | 37 | 0 | 0 | 0 | 0 | 0 | 37 |
| 893 | 1077 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Core/LocalTensor.h](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Core/LocalTensor.h) | 42 | 0 | 0 | 0 | 0 | 0 | 42 |
| 894 | 1122 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Core/ShapeInfo.h](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Core/ShapeInfo.h) | 37 | 0 | 0 | 0 | 0 | 0 | 37 |
| 895 | 867 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/EmitAsc.h](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/EmitAsc.h) | 58 | 0 | 0 | 0 | 0 | 0 | 58 |
| 896 | 1064 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/EmitNameStack.h](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/EmitNameStack.h) | 43 | 0 | 0 | 0 | 0 | 0 | 43 |
| 897 | 442 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/External/Arith.h](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/External/Arith.h) | 122 | 0 | 0 | 0 | 0 | 0 | 122 |
| 898 | 1109 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/External/Emitc.h](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/External/Emitc.h) | 38 | 0 | 0 | 0 | 0 | 0 | 38 |
| 899 | 1141 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/External/Func.h](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/External/Func.h) | 34 | 0 | 0 | 0 | 0 | 0 | 34 |
| 900 | 593 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/External/Math.h](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/External/Math.h) | 103 | 0 | 0 | 0 | 0 | 0 | 103 |
| 901 | 1150 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/External/MemRef.h](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/External/MemRef.h) | 33 | 0 | 0 | 0 | 0 | 0 | 33 |
| 902 | 1093 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/External/Scf.h](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/External/Scf.h) | 40 | 0 | 0 | 0 | 0 | 0 | 40 |
| 903 | 1130 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Fwk/TBuf.h](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Fwk/TBuf.h) | 36 | 0 | 0 | 0 | 0 | 0 | 36 |
| 904 | 907 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Fwk/TQue.h](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Fwk/TQue.h) | 51 | 0 | 0 | 0 | 0 | 0 | 51 |
| 905 | 1186 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Translation.h](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Translation.h) | 28 | 0 | 0 | 0 | 0 | 0 | 28 |
| 906 | 690 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/UniversalEmitter.h](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/UniversalEmitter.h) | 91 | 0 | 0 | 0 | 0 | 0 | 91 |
| 907 | 854 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Utils.h](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/include/ascir/Target/Asc/Utils.h) | 60 | 0 | 0 | 0 | 0 | 0 | 60 |
| 908 | 434 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/lib/Dialect/Asc/IR/Attributes.cpp](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/lib/Dialect/Asc/IR/Attributes.cpp) | 124 | 0 | 0 | 0 | 0 | 0 | 124 |
| 909 | 969 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/lib/Dialect/Asc/IR/Dialect.cpp](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/lib/Dialect/Asc/IR/Dialect.cpp) | 47 | 0 | 0 | 0 | 0 | 0 | 47 |
| 910 | 1212 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/lib/Dialect/Asc/IR/OpInterfaces.cpp](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/lib/Dialect/Asc/IR/OpInterfaces.cpp) | 18 | 0 | 0 | 0 | 0 | 0 | 18 |
| 911 | 579 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/lib/Dialect/Asc/IR/Ops.cpp](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/lib/Dialect/Asc/IR/Ops.cpp) | 105 | 0 | 0 | 0 | 0 | 0 | 105 |
| 912 | 1213 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/lib/Dialect/Asc/IR/TypeInterfaces.cpp](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/lib/Dialect/Asc/IR/TypeInterfaces.cpp) | 18 | 0 | 0 | 0 | 0 | 0 | 18 |
| 913 | 140 | [xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/lib/Dialect/Asc/IR/Types.cpp](../xla_npu/dependency/external/afir/Ascend-MLIR/externals/pyasc/lib/Dialect/Asc/IR/Types.cpp) | 249 | 0 | 0 | 0 | 0 | 0 | 249 |
| 914 | 600 | [xla_npu/dependency/external/afir/Ascend-MLIR/include/CAPI/Dialects.h](../xla_npu/dependency/external/afir/Ascend-MLIR/include/CAPI/Dialects.h) | 102 | 0 | 0 | 0 | 0 | 0 | 102 |
| 915 | 1151 | [xla_npu/dependency/external/afir/Ascend-MLIR/include/CAPI/Passes.h](../xla_npu/dependency/external/afir/Ascend-MLIR/include/CAPI/Passes.h) | 33 | 0 | 0 | 0 | 0 | 0 | 33 |
| 916 | 1131 | [xla_npu/dependency/external/afir/Ascend-MLIR/include/Conversion/AFIRToASCIR/AFIRToASCIR.h](../xla_npu/dependency/external/afir/Ascend-MLIR/include/Conversion/AFIRToASCIR/AFIRToASCIR.h) | 36 | 0 | 0 | 0 | 0 | 0 | 36 |
| 917 | 1191 | [xla_npu/dependency/external/afir/Ascend-MLIR/include/Conversion/AFIRToASCIR/AFIRToASCIRText.h](../xla_npu/dependency/external/afir/Ascend-MLIR/include/Conversion/AFIRToASCIR/AFIRToASCIRText.h) | 27 | 0 | 0 | 0 | 0 | 0 | 27 |
| 918 | 493 | [xla_npu/dependency/external/afir/Ascend-MLIR/include/Conversion/AFIRToASCIR/DialectBuilder.h](../xla_npu/dependency/external/afir/Ascend-MLIR/include/Conversion/AFIRToASCIR/DialectBuilder.h) | 114 | 0 | 0 | 0 | 0 | 0 | 114 |
| 919 | 1005 | [xla_npu/dependency/external/afir/Ascend-MLIR/include/Conversion/AFIRToASCIR/Math/Elementwise.h](../xla_npu/dependency/external/afir/Ascend-MLIR/include/Conversion/AFIRToASCIR/Math/Elementwise.h) | 45 | 0 | 0 | 0 | 0 | 0 | 45 |
| 920 | 1123 | [xla_npu/dependency/external/afir/Ascend-MLIR/include/Conversion/AFIRToASCIRText/AFIRToASCIRText.h](../xla_npu/dependency/external/afir/Ascend-MLIR/include/Conversion/AFIRToASCIRText/AFIRToASCIRText.h) | 37 | 0 | 0 | 0 | 0 | 0 | 37 |
| 921 | 1220 | [xla_npu/dependency/external/afir/Ascend-MLIR/include/Conversion/CMakeLists.txt](../xla_npu/dependency/external/afir/Ascend-MLIR/include/Conversion/CMakeLists.txt) | 12 | 0 | 0 | 0 | 0 | 0 | 12 |
| 922 | 1078 | [xla_npu/dependency/external/afir/Ascend-MLIR/include/Conversion/Passes.h](../xla_npu/dependency/external/afir/Ascend-MLIR/include/Conversion/Passes.h) | 42 | 0 | 0 | 0 | 0 | 0 | 42 |
| 923 | 753 | [xla_npu/dependency/external/afir/Ascend-MLIR/include/Conversion/Passes.td](../xla_npu/dependency/external/afir/Ascend-MLIR/include/Conversion/Passes.td) | 83 | 0 | 0 | 0 | 0 | 0 | 83 |
| 924 | 919 | [xla_npu/dependency/external/afir/Ascend-MLIR/include/Dialect/AFIR/AFIR.h](../xla_npu/dependency/external/afir/Ascend-MLIR/include/Dialect/AFIR/AFIR.h) | 50 | 0 | 0 | 0 | 0 | 0 | 50 |
| 925 | 843 | [xla_npu/dependency/external/afir/Ascend-MLIR/include/Dialect/AFIR/Core/Base.td](../xla_npu/dependency/external/afir/Ascend-MLIR/include/Dialect/AFIR/Core/Base.td) | 63 | 0 | 0 | 0 | 0 | 0 | 63 |
| 926 | 381 | [xla_npu/dependency/external/afir/Ascend-MLIR/include/Dialect/AFIR/Core/Enums.td](../xla_npu/dependency/external/afir/Ascend-MLIR/include/Dialect/AFIR/Core/Enums.td) | 134 | 0 | 0 | 0 | 0 | 0 | 134 |
| 927 | 804 | [xla_npu/dependency/external/afir/Ascend-MLIR/include/Dialect/AFIR/Dialect.td](../xla_npu/dependency/external/afir/Ascend-MLIR/include/Dialect/AFIR/Dialect.td) | 69 | 0 | 0 | 0 | 0 | 0 | 69 |
| 928 | 345 | [xla_npu/dependency/external/afir/Ascend-MLIR/include/Dialect/AFIR/DialectBuilder.h](../xla_npu/dependency/external/afir/Ascend-MLIR/include/Dialect/AFIR/DialectBuilder.h) | 144 | 0 | 0 | 0 | 0 | 0 | 144 |
| 929 | 949 | [xla_npu/dependency/external/afir/Ascend-MLIR/include/Dialect/AFIR/Ops.td](../xla_npu/dependency/external/afir/Ascend-MLIR/include/Dialect/AFIR/Ops.td) | 48 | 0 | 0 | 0 | 0 | 0 | 48 |
| 930 | 868 | [xla_npu/dependency/external/afir/Ascend-MLIR/include/Dialect/AFIR/Ops/Adv/concat.td](../xla_npu/dependency/external/afir/Ascend-MLIR/include/Dialect/AFIR/Ops/Adv/concat.td) | 58 | 0 | 0 | 0 | 0 | 0 | 58 |
| 931 | 819 | [xla_npu/dependency/external/afir/Ascend-MLIR/include/Dialect/AFIR/Ops/Adv/reduce.td](../xla_npu/dependency/external/afir/Ascend-MLIR/include/Dialect/AFIR/Ops/Adv/reduce.td) | 66 | 0 | 0 | 0 | 0 | 0 | 66 |
| 932 | 1132 | [xla_npu/dependency/external/afir/Ascend-MLIR/include/Dialect/AFIR/Ops/Basic/broadcast.td](../xla_npu/dependency/external/afir/Ascend-MLIR/include/Dialect/AFIR/Ops/Basic/broadcast.td) | 36 | 0 | 0 | 0 | 0 | 0 | 36 |
| 933 | 1079 | [xla_npu/dependency/external/afir/Ascend-MLIR/include/Dialect/AFIR/Ops/Basic/cast.td](../xla_npu/dependency/external/afir/Ascend-MLIR/include/Dialect/AFIR/Ops/Basic/cast.td) | 42 | 0 | 0 | 0 | 0 | 0 | 42 |
| 934 | 1094 | [xla_npu/dependency/external/afir/Ascend-MLIR/include/Dialect/AFIR/Ops/Basic/condition.td](../xla_npu/dependency/external/afir/Ascend-MLIR/include/Dialect/AFIR/Ops/Basic/condition.td) | 40 | 0 | 0 | 0 | 0 | 0 | 40 |
| 935 | 928 | [xla_npu/dependency/external/afir/Ascend-MLIR/include/Dialect/AFIR/Ops/Basic/data.td](../xla_npu/dependency/external/afir/Ascend-MLIR/include/Dialect/AFIR/Ops/Basic/data.td) | 49 | 0 | 0 | 0 | 0 | 0 | 49 |
| 936 | 322 | [xla_npu/dependency/external/afir/Ascend-MLIR/include/Dialect/AFIR/Ops/Basic/math.td](../xla_npu/dependency/external/afir/Ascend-MLIR/include/Dialect/AFIR/Ops/Basic/math.td) | 151 | 0 | 0 | 0 | 0 | 0 | 151 |
| 937 | 1089 | [xla_npu/dependency/external/afir/Ascend-MLIR/include/Dialect/AFIR/TransformOps/AFIRTransformOps.h](../xla_npu/dependency/external/afir/Ascend-MLIR/include/Dialect/AFIR/TransformOps/AFIRTransformOps.h) | 41 | 0 | 0 | 0 | 0 | 0 | 41 |
| 938 | 820 | [xla_npu/dependency/external/afir/Ascend-MLIR/include/Dialect/AFIR/TransformOps/AFIRTransformOps.td](../xla_npu/dependency/external/afir/Ascend-MLIR/include/Dialect/AFIR/TransformOps/AFIRTransformOps.td) | 66 | 0 | 0 | 0 | 0 | 0 | 66 |
| 939 | 1221 | [xla_npu/dependency/external/afir/Ascend-MLIR/include/Dialect/AFIR/TransformOps/CMakeLists.txt](../xla_npu/dependency/external/afir/Ascend-MLIR/include/Dialect/AFIR/TransformOps/CMakeLists.txt) | 12 | 0 | 0 | 0 | 0 | 0 | 12 |
| 940 | 1218 | [xla_npu/dependency/external/afir/Ascend-MLIR/include/Dialect/AFIR/Transforms/CMakeLists.txt](../xla_npu/dependency/external/afir/Ascend-MLIR/include/Dialect/AFIR/Transforms/CMakeLists.txt) | 13 | 0 | 0 | 0 | 0 | 0 | 13 |
| 941 | 986 | [xla_npu/dependency/external/afir/Ascend-MLIR/include/Dialect/AFIR/Transforms/Passes.h](../xla_npu/dependency/external/afir/Ascend-MLIR/include/Dialect/AFIR/Transforms/Passes.h) | 46 | 0 | 0 | 0 | 0 | 0 | 46 |
| 942 | 877 | [xla_npu/dependency/external/afir/Ascend-MLIR/include/Dialect/AFIR/Transforms/Passes.td](../xla_npu/dependency/external/afir/Ascend-MLIR/include/Dialect/AFIR/Transforms/Passes.td) | 56 | 0 | 0 | 0 | 0 | 0 | 56 |
| 943 | 208 | [xla_npu/dependency/external/afir/Ascend-MLIR/include/Utils/AFIRShapeInferUtils.h](../xla_npu/dependency/external/afir/Ascend-MLIR/include/Utils/AFIRShapeInferUtils.h) | 191 | 0 | 0 | 0 | 0 | 0 | 191 |
| 944 | 899 | [xla_npu/dependency/external/afir/Ascend-MLIR/include/Utils/AFIRUtils.h](../xla_npu/dependency/external/afir/Ascend-MLIR/include/Utils/AFIRUtils.h) | 52 | 0 | 0 | 0 | 0 | 0 | 52 |
| 945 | 601 | [xla_npu/dependency/external/afir/Ascend-MLIR/lib/Dialect/AFIR/Dialect.cpp](../xla_npu/dependency/external/afir/Ascend-MLIR/lib/Dialect/AFIR/Dialect.cpp) | 102 | 0 | 0 | 0 | 0 | 0 | 102 |
| 946 | 302 | [xla_npu/dependency/external/afir/Ascend-MLIR/lib/Dialect/AFIR/Ops/Adv/concat.cpp](../xla_npu/dependency/external/afir/Ascend-MLIR/lib/Dialect/AFIR/Ops/Adv/concat.cpp) | 158 | 0 | 0 | 0 | 0 | 0 | 158 |
| 947 | 534 | [xla_npu/dependency/external/afir/Ascend-MLIR/lib/Dialect/AFIR/Ops/Adv/reduce.cpp](../xla_npu/dependency/external/afir/Ascend-MLIR/lib/Dialect/AFIR/Ops/Adv/reduce.cpp) | 110 | 0 | 0 | 0 | 0 | 0 | 110 |
| 948 | 566 | [xla_npu/dependency/external/afir/Ascend-MLIR/lib/Dialect/AFIR/Ops/Adv/transpose.cpp](../xla_npu/dependency/external/afir/Ascend-MLIR/lib/Dialect/AFIR/Ops/Adv/transpose.cpp) | 106 | 0 | 0 | 0 | 0 | 0 | 106 |
| 949 | 314 | [xla_npu/dependency/external/afir/Ascend-MLIR/lib/Dialect/AFIR/Ops/Basic/broadcast.cpp](../xla_npu/dependency/external/afir/Ascend-MLIR/lib/Dialect/AFIR/Ops/Basic/broadcast.cpp) | 152 | 0 | 0 | 0 | 0 | 0 | 152 |
| 950 | 517 | [xla_npu/dependency/external/afir/Ascend-MLIR/lib/Dialect/AFIR/Ops/Basic/condition.cpp](../xla_npu/dependency/external/afir/Ascend-MLIR/lib/Dialect/AFIR/Ops/Basic/condition.cpp) | 111 | 0 | 0 | 0 | 0 | 0 | 111 |
| 951 | 653 | [xla_npu/dependency/external/afir/Ascend-MLIR/lib/Dialect/AFIR/Ops/Basic/data.cpp](../xla_npu/dependency/external/afir/Ascend-MLIR/lib/Dialect/AFIR/Ops/Basic/data.cpp) | 96 | 0 | 0 | 0 | 0 | 0 | 96 |
| 952 | 162 | [xla_npu/dependency/external/afir/Ascend-MLIR/lib/Dialect/AFIR/Transforms/AFIRAddAxis.cpp](../xla_npu/dependency/external/afir/Ascend-MLIR/lib/Dialect/AFIR/Transforms/AFIRAddAxis.cpp) | 218 | 0 | 0 | 0 | 0 | 0 | 218 |
| 953 | 133 | [xla_npu/dependency/external/afir/Ascend-MLIR/lib/Dialect/AFIR/Transforms/AFIRInferShapes.cpp](../xla_npu/dependency/external/afir/Ascend-MLIR/lib/Dialect/AFIR/Transforms/AFIRInferShapes.cpp) | 257 | 0 | 0 | 0 | 0 | 0 | 257 |
| 954 | 833 | [xla_npu/dependency/external/afir/Ascend-MLIR/lib/Dialect/AFIR/Transforms/Canonicalize.cpp](../xla_npu/dependency/external/afir/Ascend-MLIR/lib/Dialect/AFIR/Transforms/Canonicalize.cpp) | 65 | 0 | 0 | 0 | 0 | 0 | 65 |
| 955 | 844 | [xla_npu/dependency/external/afir/Ascend-MLIR/lib/Utils/AFIRUtils.cpp](../xla_npu/dependency/external/afir/Ascend-MLIR/lib/Utils/AFIRUtils.cpp) | 63 | 0 | 0 | 0 | 0 | 0 | 63 |
| 956 | 128 | [xla_npu/dependency/external/afir/generated/externals/pyasc/include/ascir/API/Types.h.inc](../xla_npu/dependency/external/afir/generated/externals/pyasc/include/ascir/API/Types.h.inc) | 262 | 0 | 0 | 0 | 0 | 0 | 262 |
| 957 | 200 | [xla_npu/dependency/external/afir/generated/externals/pyasc/include/ascir/API/Types.td.inc](../xla_npu/dependency/external/afir/generated/externals/pyasc/include/ascir/API/Types.td.inc) | 192 | 0 | 0 | 0 | 0 | 0 | 192 |
| 958 | 12 | [xla_npu/dependency/external/afir/generated/externals/pyasc/include/ascir/Dialect/Asc/IR/AscendCAttributes.cpp.inc](../xla_npu/dependency/external/afir/generated/externals/pyasc/include/ascir/Dialect/Asc/IR/AscendCAttributes.cpp.inc) | 1761 | 0 | 0 | 0 | 0 | 0 | 1761 |
| 959 | 311 | [xla_npu/dependency/external/afir/generated/externals/pyasc/include/ascir/Dialect/Asc/IR/AscendCAttributes.h.inc](../xla_npu/dependency/external/afir/generated/externals/pyasc/include/ascir/Dialect/Asc/IR/AscendCAttributes.h.inc) | 153 | 0 | 0 | 0 | 0 | 0 | 153 |
| 960 | 1110 | [xla_npu/dependency/external/afir/generated/externals/pyasc/include/ascir/Dialect/Asc/IR/AscendCDialect.cpp.inc](../xla_npu/dependency/external/afir/generated/externals/pyasc/include/ascir/Dialect/Asc/IR/AscendCDialect.cpp.inc) | 38 | 0 | 0 | 0 | 0 | 0 | 38 |
| 961 | 856 | [xla_npu/dependency/external/afir/generated/externals/pyasc/include/ascir/Dialect/Asc/IR/AscendCDialect.h.inc](../xla_npu/dependency/external/afir/generated/externals/pyasc/include/ascir/Dialect/Asc/IR/AscendCDialect.h.inc) | 60 | 0 | 0 | 0 | 0 | 0 | 60 |
| 962 | 13 | [xla_npu/dependency/external/afir/generated/externals/pyasc/include/ascir/Dialect/Asc/IR/AscendCEnums.cpp.inc](../xla_npu/dependency/external/afir/generated/externals/pyasc/include/ascir/Dialect/Asc/IR/AscendCEnums.cpp.inc) | 1729 | 0 | 0 | 0 | 0 | 0 | 1729 |
| 963 | 7 | [xla_npu/dependency/external/afir/generated/externals/pyasc/include/ascir/Dialect/Asc/IR/AscendCEnums.h.inc](../xla_npu/dependency/external/afir/generated/externals/pyasc/include/ascir/Dialect/Asc/IR/AscendCEnums.h.inc) | 3387 | 0 | 0 | 0 | 0 | 0 | 3387 |
| 964 | 70 | [xla_npu/dependency/external/afir/generated/externals/pyasc/include/ascir/Dialect/Asc/IR/AscendCOpInterfaces.cpp.inc](../xla_npu/dependency/external/afir/generated/externals/pyasc/include/ascir/Dialect/Asc/IR/AscendCOpInterfaces.cpp.inc) | 403 | 0 | 0 | 0 | 0 | 0 | 403 |
| 965 | 10 | [xla_npu/dependency/external/afir/generated/externals/pyasc/include/ascir/Dialect/Asc/IR/AscendCOpInterfaces.h.inc](../xla_npu/dependency/external/afir/generated/externals/pyasc/include/ascir/Dialect/Asc/IR/AscendCOpInterfaces.h.inc) | 2101 | 0 | 0 | 0 | 0 | 0 | 2101 |
| 966 | 1 | [xla_npu/dependency/external/afir/generated/externals/pyasc/include/ascir/Dialect/Asc/IR/AscendCOps.cpp.inc](../xla_npu/dependency/external/afir/generated/externals/pyasc/include/ascir/Dialect/Asc/IR/AscendCOps.cpp.inc) | 119282 | 0 | 0 | 0 | 0 | 0 | 119282 |
| 967 | 2 | [xla_npu/dependency/external/afir/generated/externals/pyasc/include/ascir/Dialect/Asc/IR/AscendCOps.h.inc](../xla_npu/dependency/external/afir/generated/externals/pyasc/include/ascir/Dialect/Asc/IR/AscendCOps.h.inc) | 94334 | 0 | 0 | 0 | 0 | 0 | 94334 |
| 968 | 929 | [xla_npu/dependency/external/afir/generated/externals/pyasc/include/ascir/Dialect/Asc/IR/AscendCTypeInterfaces.cpp.inc](../xla_npu/dependency/external/afir/generated/externals/pyasc/include/ascir/Dialect/Asc/IR/AscendCTypeInterfaces.cpp.inc) | 49 | 0 | 0 | 0 | 0 | 0 | 49 |
| 969 | 86 | [xla_npu/dependency/external/afir/generated/externals/pyasc/include/ascir/Dialect/Asc/IR/AscendCTypeInterfaces.h.inc](../xla_npu/dependency/external/afir/generated/externals/pyasc/include/ascir/Dialect/Asc/IR/AscendCTypeInterfaces.h.inc) | 334 | 0 | 0 | 0 | 0 | 0 | 334 |
| 970 | 9 | [xla_npu/dependency/external/afir/generated/externals/pyasc/include/ascir/Dialect/Asc/IR/AscendCTypes.cpp.inc](../xla_npu/dependency/external/afir/generated/externals/pyasc/include/ascir/Dialect/Asc/IR/AscendCTypes.cpp.inc) | 2305 | 0 | 0 | 0 | 0 | 0 | 2305 |
| 971 | 18 | [xla_npu/dependency/external/afir/generated/externals/pyasc/include/ascir/Dialect/Asc/IR/AscendCTypes.h.inc](../xla_npu/dependency/external/afir/generated/externals/pyasc/include/ascir/Dialect/Asc/IR/AscendCTypes.h.inc) | 1158 | 0 | 0 | 0 | 0 | 0 | 1158 |
| 972 | 157 | [xla_npu/dependency/external/afir/generated/include/Conversion/Passes.h.inc](../xla_npu/dependency/external/afir/generated/include/Conversion/Passes.h.inc) | 224 | 0 | 0 | 0 | 0 | 0 | 224 |
| 973 | 19 | [xla_npu/dependency/external/afir/generated/include/Dialect/AFIR/Attrs.cpp.inc](../xla_npu/dependency/external/afir/generated/include/Dialect/AFIR/Attrs.cpp.inc) | 1150 | 0 | 0 | 0 | 0 | 0 | 1150 |
| 974 | 201 | [xla_npu/dependency/external/afir/generated/include/Dialect/AFIR/Attrs.h.inc](../xla_npu/dependency/external/afir/generated/include/Dialect/AFIR/Attrs.h.inc) | 192 | 0 | 0 | 0 | 0 | 0 | 192 |
| 975 | 1111 | [xla_npu/dependency/external/afir/generated/include/Dialect/AFIR/Dialect.cpp.inc](../xla_npu/dependency/external/afir/generated/include/Dialect/AFIR/Dialect.cpp.inc) | 38 | 0 | 0 | 0 | 0 | 0 | 38 |
| 976 | 869 | [xla_npu/dependency/external/afir/generated/include/Dialect/AFIR/Dialect.h.inc](../xla_npu/dependency/external/afir/generated/include/Dialect/AFIR/Dialect.h.inc) | 58 | 0 | 0 | 0 | 0 | 0 | 58 |
| 977 | 101 | [xla_npu/dependency/external/afir/generated/include/Dialect/AFIR/Enums.cpp.inc](../xla_npu/dependency/external/afir/generated/include/Dialect/AFIR/Enums.cpp.inc) | 298 | 0 | 0 | 0 | 0 | 0 | 298 |
| 978 | 41 | [xla_npu/dependency/external/afir/generated/include/Dialect/AFIR/Enums.h.inc](../xla_npu/dependency/external/afir/generated/include/Dialect/AFIR/Enums.h.inc) | 587 | 0 | 0 | 0 | 0 | 0 | 587 |
| 979 | 3 | [xla_npu/dependency/external/afir/generated/include/Dialect/AFIR/Ops.cpp.inc](../xla_npu/dependency/external/afir/generated/include/Dialect/AFIR/Ops.cpp.inc) | 32558 | 0 | 0 | 0 | 0 | 0 | 32558 |
| 980 | 4 | [xla_npu/dependency/external/afir/generated/include/Dialect/AFIR/Ops.h.inc](../xla_npu/dependency/external/afir/generated/include/Dialect/AFIR/Ops.h.inc) | 16688 | 0 | 0 | 0 | 0 | 0 | 16688 |
| 981 | 60 | [xla_npu/dependency/external/afir/generated/include/Dialect/AFIR/TransformOps/AFIRTransformOps.cpp.inc](../xla_npu/dependency/external/afir/generated/include/Dialect/AFIR/TransformOps/AFIRTransformOps.cpp.inc) | 446 | 0 | 0 | 0 | 0 | 0 | 446 |
| 982 | 112 | [xla_npu/dependency/external/afir/generated/include/Dialect/AFIR/TransformOps/AFIRTransformOps.h.inc](../xla_npu/dependency/external/afir/generated/include/Dialect/AFIR/TransformOps/AFIRTransformOps.h.inc) | 283 | 0 | 0 | 0 | 0 | 0 | 283 |
| 983 | 103 | [xla_npu/dependency/external/afir/generated/include/Dialect/AFIR/Transforms/Passes.h.inc](../xla_npu/dependency/external/afir/generated/include/Dialect/AFIR/Transforms/Passes.h.inc) | 296 | 0 | 0 | 0 | 0 | 0 | 296 |
| 984 | 435 | [xla_npu/dependency/header/acl/acl.h](../xla_npu/dependency/header/acl/acl.h) | 124 | 0 | 0 | 0 | 0 | 0 | 124 |
| 985 | 29 | [xla_npu/dependency/header/acl/acl_base.h](../xla_npu/dependency/header/acl/acl_base.h) | 776 | 0 | 0 | 0 | 0 | 0 | 776 |
| 986 | 14 | [xla_npu/dependency/header/acl/acl_mdl.h](../xla_npu/dependency/header/acl/acl_mdl.h) | 1601 | 0 | 0 | 0 | 0 | 0 | 1601 |
| 987 | 39 | [xla_npu/dependency/header/acl/acl_op.h](../xla_npu/dependency/header/acl/acl_op.h) | 602 | 0 | 0 | 0 | 0 | 0 | 602 |
| 988 | 6 | [xla_npu/dependency/header/acl/acl_rt.h](../xla_npu/dependency/header/acl/acl_rt.h) | 3781 | 0 | 0 | 0 | 0 | 0 | 3781 |
| 989 | 699 | [xla_npu/dependency/header/acl/error_codes/ge_error_codes.h](../xla_npu/dependency/header/acl/error_codes/ge_error_codes.h) | 90 | 0 | 0 | 0 | 0 | 0 | 90 |
| 990 | 256 | [xla_npu/dependency/header/acl/error_codes/rt_error_codes.h](../xla_npu/dependency/header/acl/error_codes/rt_error_codes.h) | 171 | 0 | 0 | 0 | 0 | 0 | 171 |
| 991 | 232 | [xla_npu/dependency/header/acl_nn/acl_meta.h](../xla_npu/dependency/header/acl_nn/acl_meta.h) | 178 | 0 | 0 | 0 | 0 | 0 | 178 |
| 992 | 859 | [xla_npu/dependency/header/acl_nn/aclnn_base.h](../xla_npu/dependency/header/acl_nn/aclnn_base.h) | 59 | 0 | 0 | 0 | 0 | 0 | 59 |
| 993 | 74 | [xla_npu/dependency/header/acl_nn/opdev/aicpu/aicpu_any_value.h](../xla_npu/dependency/header/acl_nn/opdev/aicpu/aicpu_any_value.h) | 381 | 0 | 0 | 0 | 0 | 0 | 381 |
| 994 | 390 | [xla_npu/dependency/header/acl_nn/opdev/aicpu/aicpu_ext_info_handle.h](../xla_npu/dependency/header/acl_nn/opdev/aicpu/aicpu_ext_info_handle.h) | 132 | 0 | 0 | 0 | 0 | 0 | 132 |
| 995 | 691 | [xla_npu/dependency/header/acl_nn/opdev/aicpu/aicpu_uitls.h](../xla_npu/dependency/header/acl_nn/opdev/aicpu/aicpu_uitls.h) | 91 | 0 | 0 | 0 | 0 | 0 | 91 |
| 996 | 79 | [xla_npu/dependency/header/acl_nn/opdev/bfloat16.h](../xla_npu/dependency/header/acl_nn/opdev/bfloat16.h) | 355 | 0 | 0 | 0 | 0 | 0 | 355 |
| 997 | 237 | [xla_npu/dependency/header/acl_nn/opdev/data_type_utils.h](../xla_npu/dependency/header/acl_nn/opdev/data_type_utils.h) | 177 | 0 | 0 | 0 | 0 | 0 | 177 |
| 998 | 1192 | [xla_npu/dependency/header/acl_nn/opdev/fast_vector.h](../xla_npu/dependency/header/acl_nn/opdev/fast_vector.h) | 27 | 0 | 0 | 0 | 0 | 0 | 27 |
| 999 | 554 | [xla_npu/dependency/header/acl_nn/opdev/format_utils.h](../xla_npu/dependency/header/acl_nn/opdev/format_utils.h) | 107 | 0 | 0 | 0 | 0 | 0 | 107 |
| 1000 | 25 | [xla_npu/dependency/header/acl_nn/opdev/fp16_t.h](../xla_npu/dependency/header/acl_nn/opdev/fp16_t.h) | 885 | 0 | 0 | 0 | 0 | 0 | 885 |
| 1001 | 1193 | [xla_npu/dependency/header/acl_nn/opdev/framework_op.h](../xla_npu/dependency/header/acl_nn/opdev/framework_op.h) | 27 | 0 | 0 | 0 | 0 | 0 | 27 |
| 1002 | 812 | [xla_npu/dependency/header/acl_nn/opdev/make_op_executor.h](../xla_npu/dependency/header/acl_nn/opdev/make_op_executor.h) | 67 | 0 | 0 | 0 | 0 | 0 | 67 |
| 1003 | 970 | [xla_npu/dependency/header/acl_nn/opdev/object.h](../xla_npu/dependency/header/acl_nn/opdev/object.h) | 47 | 0 | 0 | 0 | 0 | 0 | 47 |
| 1004 | 117 | [xla_npu/dependency/header/acl_nn/opdev/op_cache.h](../xla_npu/dependency/header/acl_nn/opdev/op_cache.h) | 278 | 0 | 0 | 0 | 0 | 0 | 278 |
| 1005 | 77 | [xla_npu/dependency/header/acl_nn/opdev/op_cache_container.h](../xla_npu/dependency/header/acl_nn/opdev/op_cache_container.h) | 365 | 0 | 0 | 0 | 0 | 0 | 365 |
| 1006 | 170 | [xla_npu/dependency/header/acl_nn/opdev/op_common/data_type_utils.h](../xla_npu/dependency/header/acl_nn/opdev/op_common/data_type_utils.h) | 213 | 0 | 0 | 0 | 0 | 0 | 213 |
| 1007 | 1124 | [xla_npu/dependency/header/acl_nn/opdev/op_common/tiling_aicpu_task.h](../xla_npu/dependency/header/acl_nn/opdev/op_common/tiling_aicpu_task.h) | 37 | 0 | 0 | 0 | 0 | 0 | 37 |
| 1008 | 971 | [xla_npu/dependency/header/acl_nn/opdev/op_config.h](../xla_npu/dependency/header/acl_nn/opdev/op_config.h) | 47 | 0 | 0 | 0 | 0 | 0 | 47 |
| 1009 | 726 | [xla_npu/dependency/header/acl_nn/opdev/op_def.h](../xla_npu/dependency/header/acl_nn/opdev/op_def.h) | 87 | 0 | 0 | 0 | 0 | 0 | 87 |
| 1010 | 950 | [xla_npu/dependency/header/acl_nn/opdev/op_errno.h](../xla_npu/dependency/header/acl_nn/opdev/op_errno.h) | 48 | 0 | 0 | 0 | 0 | 0 | 48 |
| 1011 | 129 | [xla_npu/dependency/header/acl_nn/opdev/op_executor.h](../xla_npu/dependency/header/acl_nn/opdev/op_executor.h) | 259 | 0 | 0 | 0 | 0 | 0 | 259 |
| 1012 | 122 | [xla_npu/dependency/header/acl_nn/opdev/op_log.h](../xla_npu/dependency/header/acl_nn/opdev/op_log.h) | 267 | 0 | 0 | 0 | 0 | 0 | 267 |
| 1013 | 456 | [xla_npu/dependency/header/acl_nn/opdev/platform.h](../xla_npu/dependency/header/acl_nn/opdev/platform.h) | 118 | 0 | 0 | 0 | 0 | 0 | 118 |
| 1014 | 758 | [xla_npu/dependency/header/acl_nn/opdev/pool_allocator.h](../xla_npu/dependency/header/acl_nn/opdev/pool_allocator.h) | 82 | 0 | 0 | 0 | 0 | 0 | 82 |
| 1015 | 860 | [xla_npu/dependency/header/acl_nn/opdev/register/device_op_impl_registry.h](../xla_npu/dependency/header/acl_nn/opdev/register/device_op_impl_registry.h) | 59 | 0 | 0 | 0 | 0 | 0 | 59 |
| 1016 | 1125 | [xla_npu/dependency/header/acl_nn/opdev/shape_utils.h](../xla_npu/dependency/header/acl_nn/opdev/shape_utils.h) | 37 | 0 | 0 | 0 | 0 | 0 | 37 |
| 1017 | 861 | [xla_npu/dependency/header/acl_nn/opdev/tensor_view_utils.h](../xla_npu/dependency/header/acl_nn/opdev/tensor_view_utils.h) | 59 | 0 | 0 | 0 | 0 | 0 | 59 |
| 1018 | 800 | [xla_npu/dependency/header/acl_nn_op/acl_rfft1d.h](../xla_npu/dependency/header/acl_nn_op/acl_rfft1d.h) | 70 | 0 | 0 | 0 | 0 | 0 | 70 |
| 1019 | 987 | [xla_npu/dependency/header/acl_nn_op/aclnn_ada_layer_norm.h](../xla_npu/dependency/header/acl_nn_op/aclnn_ada_layer_norm.h) | 46 | 0 | 0 | 0 | 0 | 0 | 46 |
| 1020 | 930 | [xla_npu/dependency/header/acl_nn_op/aclnn_ada_layer_norm_quant.h](../xla_npu/dependency/header/acl_nn_op/aclnn_ada_layer_norm_quant.h) | 49 | 0 | 0 | 0 | 0 | 0 | 49 |
| 1021 | 951 | [xla_npu/dependency/header/acl_nn_op/aclnn_ada_layer_norm_v2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_ada_layer_norm_v2.h) | 48 | 0 | 0 | 0 | 0 | 0 | 48 |
| 1022 | 813 | [xla_npu/dependency/header/acl_nn_op/aclnn_adaptive_max_pool2d_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_adaptive_max_pool2d_backward.h) | 67 | 0 | 0 | 0 | 0 | 0 | 67 |
| 1023 | 814 | [xla_npu/dependency/header/acl_nn_op/aclnn_adaptive_max_pool3d_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_adaptive_max_pool3d_backward.h) | 67 | 0 | 0 | 0 | 0 | 0 | 67 |
| 1024 | 1364 | [xla_npu/dependency/header/acl_nn_op/aclnn_add_layer_norm.h](../xla_npu/dependency/header/acl_nn_op/aclnn_add_layer_norm.h) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1025 | 1228 | [xla_npu/dependency/header/acl_nn_op/aclnn_add_layer_norm_grad.h](../xla_npu/dependency/header/acl_nn_op/aclnn_add_layer_norm_grad.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1026 | 1365 | [xla_npu/dependency/header/acl_nn_op/aclnn_add_lora.h](../xla_npu/dependency/header/acl_nn_op/aclnn_add_lora.h) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1027 | 1229 | [xla_npu/dependency/header/acl_nn_op/aclnn_add_rms_norm.h](../xla_npu/dependency/header/acl_nn_op/aclnn_add_rms_norm.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1028 | 1230 | [xla_npu/dependency/header/acl_nn_op/aclnn_add_rms_norm_cast.h](../xla_npu/dependency/header/acl_nn_op/aclnn_add_rms_norm_cast.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1029 | 1231 | [xla_npu/dependency/header/acl_nn_op/aclnn_advance_step.h](../xla_npu/dependency/header/acl_nn_op/aclnn_advance_step.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1030 | 1232 | [xla_npu/dependency/header/acl_nn_op/aclnn_apply_adam_w_quant.h](../xla_npu/dependency/header/acl_nn_op/aclnn_apply_adam_w_quant.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1031 | 1233 | [xla_npu/dependency/header/acl_nn_op/aclnn_apply_fused_ema_adam.h](../xla_npu/dependency/header/acl_nn_op/aclnn_apply_fused_ema_adam.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1032 | 1234 | [xla_npu/dependency/header/acl_nn_op/aclnn_background_replace.h](../xla_npu/dependency/header/acl_nn_op/aclnn_background_replace.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1033 | 1235 | [xla_npu/dependency/header/acl_nn_op/aclnn_bidirection_lstm.h](../xla_npu/dependency/header/acl_nn_op/aclnn_bidirection_lstm.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1034 | 1236 | [xla_npu/dependency/header/acl_nn_op/aclnn_bidirection_lstmv2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_bidirection_lstmv2.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1035 | 1237 | [xla_npu/dependency/header/acl_nn_op/aclnn_blend_images_custom.h](../xla_npu/dependency/header/acl_nn_op/aclnn_blend_images_custom.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1036 | 1238 | [xla_npu/dependency/header/acl_nn_op/aclnn_clipped_swiglu.h](../xla_npu/dependency/header/acl_nn_op/aclnn_clipped_swiglu.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1037 | 1102 | [xla_npu/dependency/header/acl_nn_op/aclnn_confusion_transpose.h](../xla_npu/dependency/header/acl_nn_op/aclnn_confusion_transpose.h) | 39 | 0 | 0 | 0 | 0 | 0 | 39 |
| 1038 | 1239 | [xla_npu/dependency/header/acl_nn_op/aclnn_cross_entropy_loss.h](../xla_npu/dependency/header/acl_nn_op/aclnn_cross_entropy_loss.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1039 | 1240 | [xla_npu/dependency/header/acl_nn_op/aclnn_cross_entropy_loss_grad.h](../xla_npu/dependency/header/acl_nn_op/aclnn_cross_entropy_loss_grad.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1040 | 1241 | [xla_npu/dependency/header/acl_nn_op/aclnn_deep_norm.h](../xla_npu/dependency/header/acl_nn_op/aclnn_deep_norm.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1041 | 1242 | [xla_npu/dependency/header/acl_nn_op/aclnn_deep_norm_grad.h](../xla_npu/dependency/header/acl_nn_op/aclnn_deep_norm_grad.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1042 | 1243 | [xla_npu/dependency/header/acl_nn_op/aclnn_dequant_bias.h](../xla_npu/dependency/header/acl_nn_op/aclnn_dequant_bias.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1043 | 1244 | [xla_npu/dependency/header/acl_nn_op/aclnn_dequant_rope_quant_kvcache.h](../xla_npu/dependency/header/acl_nn_op/aclnn_dequant_rope_quant_kvcache.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1044 | 1245 | [xla_npu/dependency/header/acl_nn_op/aclnn_dequant_swiglu_quant.h](../xla_npu/dependency/header/acl_nn_op/aclnn_dequant_swiglu_quant.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1045 | 1246 | [xla_npu/dependency/header/acl_nn_op/aclnn_dequant_swiglu_quant_v2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_dequant_swiglu_quant_v2.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1046 | 1247 | [xla_npu/dependency/header/acl_nn_op/aclnn_dynamic_block_quant.h](../xla_npu/dependency/header/acl_nn_op/aclnn_dynamic_block_quant.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1047 | 1248 | [xla_npu/dependency/header/acl_nn_op/aclnn_dynamic_mx_quant.h](../xla_npu/dependency/header/acl_nn_op/aclnn_dynamic_mx_quant.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1048 | 1249 | [xla_npu/dependency/header/acl_nn_op/aclnn_expand_into_jagged_permute.h](../xla_npu/dependency/header/acl_nn_op/aclnn_expand_into_jagged_permute.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1049 | 1250 | [xla_npu/dependency/header/acl_nn_op/aclnn_fatrelu_mul.h](../xla_npu/dependency/header/acl_nn_op/aclnn_fatrelu_mul.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1050 | 1251 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_abs.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_abs.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1051 | 1252 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_acos.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_acos.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1052 | 1253 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_add_list.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_add_list.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1053 | 1254 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_add_scalar.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_add_scalar.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1054 | 1255 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_add_scalar_list.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_add_scalar_list.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1055 | 1256 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_addcdiv_list.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_addcdiv_list.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1056 | 1257 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_addcdiv_scalar.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_addcdiv_scalar.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1057 | 1258 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_addcdiv_scalar_list.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_addcdiv_scalar_list.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1058 | 1259 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_addcmul_list.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_addcmul_list.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1059 | 1260 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_addcmul_scalar.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_addcmul_scalar.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1060 | 1261 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_addcmul_scalar_list.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_addcmul_scalar_list.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1061 | 1262 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_asin.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_asin.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1062 | 1263 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_atan.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_atan.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1063 | 1264 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_copy.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_copy.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1064 | 1265 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_cos.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_cos.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1065 | 1266 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_cosh.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_cosh.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1066 | 1267 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_div_list.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_div_list.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1067 | 1268 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_div_scalar.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_div_scalar.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1068 | 1269 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_div_scalar_list.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_div_scalar_list.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1069 | 1270 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_erf.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_erf.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1070 | 1271 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_erfc.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_erfc.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1071 | 1272 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_exp.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_exp.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1072 | 1273 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_expm1.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_expm1.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1073 | 1274 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_lerp_list.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_lerp_list.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1074 | 1275 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_lerp_scalar.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_lerp_scalar.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1075 | 1276 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_log.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_log.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1076 | 1277 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_log10.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_log10.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1077 | 1278 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_log1p.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_log1p.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1078 | 1279 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_log2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_log2.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1079 | 1280 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_maximum_list.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_maximum_list.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1080 | 1281 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_maximum_scalar.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_maximum_scalar.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1081 | 1282 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_maximum_scalar_list.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_maximum_scalar_list.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1082 | 1283 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_minimum_list.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_minimum_list.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1083 | 1284 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_minimum_scalar.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_minimum_scalar.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1084 | 1285 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_minimum_scalar_list.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_minimum_scalar_list.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1085 | 1286 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_mul_list.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_mul_list.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1086 | 1287 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_mul_scalar.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_mul_scalar.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1087 | 1288 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_mul_scalar_list.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_mul_scalar_list.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1088 | 1289 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_neg.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_neg.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1089 | 1290 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_non_finite_check_and_unscale.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_non_finite_check_and_unscale.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1090 | 1291 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_norm.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_norm.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1091 | 1292 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_pow_list.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_pow_list.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1092 | 1293 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_pow_scalar.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_pow_scalar.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1093 | 1294 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_pow_scalar_and_tensor.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_pow_scalar_and_tensor.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1094 | 1295 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_pow_scalar_list.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_pow_scalar_list.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1095 | 1296 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_reciprocal.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_reciprocal.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1096 | 1297 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_round_off_number.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_round_off_number.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1097 | 1298 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_sigmoid.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_sigmoid.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1098 | 1299 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_sign.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_sign.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1099 | 1300 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_sin.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_sin.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1100 | 1301 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_sinh.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_sinh.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1101 | 1302 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_sqrt.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_sqrt.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1102 | 1303 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_sub_list.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_sub_list.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1103 | 1304 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_sub_scalar.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_sub_scalar.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1104 | 1305 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_sub_scalar_list.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_sub_scalar_list.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1105 | 1306 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_tan.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_tan.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1106 | 1307 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_tanh.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_tanh.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1107 | 1308 | [xla_npu/dependency/header/acl_nn_op/aclnn_foreach_zero_inplace.h](../xla_npu/dependency/header/acl_nn_op/aclnn_foreach_zero_inplace.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1108 | 901 | [xla_npu/dependency/header/acl_nn_op/aclnn_fused_infer_attention_score.h](../xla_npu/dependency/header/acl_nn_op/aclnn_fused_infer_attention_score.h) | 52 | 0 | 0 | 0 | 0 | 0 | 52 |
| 1109 | 838 | [xla_npu/dependency/header/acl_nn_op/aclnn_fused_infer_attention_score_v2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_fused_infer_attention_score_v2.h) | 64 | 0 | 0 | 0 | 0 | 0 | 64 |
| 1110 | 821 | [xla_npu/dependency/header/acl_nn_op/aclnn_fused_infer_attention_score_v3.h](../xla_npu/dependency/header/acl_nn_op/aclnn_fused_infer_attention_score_v3.h) | 66 | 0 | 0 | 0 | 0 | 0 | 66 |
| 1111 | 805 | [xla_npu/dependency/header/acl_nn_op/aclnn_fused_infer_attention_score_v4.h](../xla_npu/dependency/header/acl_nn_op/aclnn_fused_infer_attention_score_v4.h) | 69 | 0 | 0 | 0 | 0 | 0 | 69 |
| 1112 | 801 | [xla_npu/dependency/header/acl_nn_op/aclnn_fused_infer_attention_score_vx.h](../xla_npu/dependency/header/acl_nn_op/aclnn_fused_infer_attention_score_vx.h) | 70 | 0 | 0 | 0 | 0 | 0 | 70 |
| 1113 | 1309 | [xla_npu/dependency/header/acl_nn_op/aclnn_gather_pa_kv_cache.h](../xla_npu/dependency/header/acl_nn_op/aclnn_gather_pa_kv_cache.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1114 | 1310 | [xla_npu/dependency/header/acl_nn_op/aclnn_gelu_mul.h](../xla_npu/dependency/header/acl_nn_op/aclnn_gelu_mul.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1115 | 1311 | [xla_npu/dependency/header/acl_nn_op/aclnn_gemma_rms_norm.h](../xla_npu/dependency/header/acl_nn_op/aclnn_gemma_rms_norm.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1116 | 1312 | [xla_npu/dependency/header/acl_nn_op/aclnn_group_norm_swish.h](../xla_npu/dependency/header/acl_nn_op/aclnn_group_norm_swish.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1117 | 1313 | [xla_npu/dependency/header/acl_nn_op/aclnn_group_norm_swish_grad.h](../xla_npu/dependency/header/acl_nn_op/aclnn_group_norm_swish_grad.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1118 | 1314 | [xla_npu/dependency/header/acl_nn_op/aclnn_inplace_add_rms_norm.h](../xla_npu/dependency/header/acl_nn_op/aclnn_inplace_add_rms_norm.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1119 | 1315 | [xla_npu/dependency/header/acl_nn_op/aclnn_is_inf.h](../xla_npu/dependency/header/acl_nn_op/aclnn_is_inf.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1120 | 1316 | [xla_npu/dependency/header/acl_nn_op/aclnn_kv_rms_norm_rope_cache.h](../xla_npu/dependency/header/acl_nn_op/aclnn_kv_rms_norm_rope_cache.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1121 | 1317 | [xla_npu/dependency/header/acl_nn_op/aclnn_logit.h](../xla_npu/dependency/header/acl_nn_op/aclnn_logit.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1122 | 1318 | [xla_npu/dependency/header/acl_nn_op/aclnn_logit_grad.h](../xla_npu/dependency/header/acl_nn_op/aclnn_logit_grad.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1123 | 1319 | [xla_npu/dependency/header/acl_nn_op/aclnn_masked_softmax_with_rel_pos_bias.h](../xla_npu/dependency/header/acl_nn_op/aclnn_masked_softmax_with_rel_pos_bias.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1124 | 226 | [xla_npu/dependency/header/acl_nn_op/aclnn_math.h](../xla_npu/dependency/header/acl_nn_op/aclnn_math.h) | 183 | 0 | 0 | 0 | 0 | 0 | 183 |
| 1125 | 1320 | [xla_npu/dependency/header/acl_nn_op/aclnn_matmul.h](../xla_npu/dependency/header/acl_nn_op/aclnn_matmul.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1126 | 822 | [xla_npu/dependency/header/acl_nn_op/aclnn_max_pool3d_with_argmax.h](../xla_npu/dependency/header/acl_nn_op/aclnn_max_pool3d_with_argmax.h) | 66 | 0 | 0 | 0 | 0 | 0 | 66 |
| 1127 | 783 | [xla_npu/dependency/header/acl_nn_op/aclnn_max_pool3d_with_argmax_backward.h](../xla_npu/dependency/header/acl_nn_op/aclnn_max_pool3d_with_argmax_backward.h) | 75 | 0 | 0 | 0 | 0 | 0 | 75 |
| 1128 | 1321 | [xla_npu/dependency/header/acl_nn_op/aclnn_mla_preprocess.h](../xla_npu/dependency/header/acl_nn_op/aclnn_mla_preprocess.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1129 | 1322 | [xla_npu/dependency/header/acl_nn_op/aclnn_mla_prolog.h](../xla_npu/dependency/header/acl_nn_op/aclnn_mla_prolog.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1130 | 878 | [xla_npu/dependency/header/acl_nn_op/aclnn_mla_prolog_v2_weight_nz.h](../xla_npu/dependency/header/acl_nn_op/aclnn_mla_prolog_v2_weight_nz.h) | 56 | 0 | 0 | 0 | 0 | 0 | 56 |
| 1131 | 847 | [xla_npu/dependency/header/acl_nn_op/aclnn_mla_prolog_v3_weight_nz.h](../xla_npu/dependency/header/acl_nn_op/aclnn_mla_prolog_v3_weight_nz.h) | 62 | 0 | 0 | 0 | 0 | 0 | 62 |
| 1132 | 1323 | [xla_npu/dependency/header/acl_nn_op/aclnn_mm.h](../xla_npu/dependency/header/acl_nn_op/aclnn_mm.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1133 | 1324 | [xla_npu/dependency/header/acl_nn_op/aclnn_moe_compute_expert_tokens.h](../xla_npu/dependency/header/acl_nn_op/aclnn_moe_compute_expert_tokens.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1134 | 1325 | [xla_npu/dependency/header/acl_nn_op/aclnn_moe_finalize_routing.h](../xla_npu/dependency/header/acl_nn_op/aclnn_moe_finalize_routing.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1135 | 1326 | [xla_npu/dependency/header/acl_nn_op/aclnn_moe_gating_top_k.h](../xla_npu/dependency/header/acl_nn_op/aclnn_moe_gating_top_k.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1136 | 1327 | [xla_npu/dependency/header/acl_nn_op/aclnn_moe_gating_top_k_softmax.h](../xla_npu/dependency/header/acl_nn_op/aclnn_moe_gating_top_k_softmax.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1137 | 1328 | [xla_npu/dependency/header/acl_nn_op/aclnn_moe_gating_top_k_softmax_v2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_moe_gating_top_k_softmax_v2.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1138 | 1329 | [xla_npu/dependency/header/acl_nn_op/aclnn_moe_init_routing.h](../xla_npu/dependency/header/acl_nn_op/aclnn_moe_init_routing.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1139 | 1330 | [xla_npu/dependency/header/acl_nn_op/aclnn_moe_init_routing_quant.h](../xla_npu/dependency/header/acl_nn_op/aclnn_moe_init_routing_quant.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1140 | 1331 | [xla_npu/dependency/header/acl_nn_op/aclnn_moe_init_routing_quant_v2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_moe_init_routing_quant_v2.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1141 | 815 | [xla_npu/dependency/header/acl_nn_op/aclnn_moe_init_routing_v2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_moe_init_routing_v2.h) | 67 | 0 | 0 | 0 | 0 | 0 | 67 |
| 1142 | 1332 | [xla_npu/dependency/header/acl_nn_op/aclnn_moe_token_permute_with_ep.h](../xla_npu/dependency/header/acl_nn_op/aclnn_moe_token_permute_with_ep.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1143 | 1333 | [xla_npu/dependency/header/acl_nn_op/aclnn_moe_token_permute_with_ep_grad.h](../xla_npu/dependency/header/acl_nn_op/aclnn_moe_token_permute_with_ep_grad.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1144 | 1334 | [xla_npu/dependency/header/acl_nn_op/aclnn_moe_token_unpermute_with_ep.h](../xla_npu/dependency/header/acl_nn_op/aclnn_moe_token_unpermute_with_ep.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1145 | 1335 | [xla_npu/dependency/header/acl_nn_op/aclnn_moe_token_unpermute_with_ep_grad.h](../xla_npu/dependency/header/acl_nn_op/aclnn_moe_token_unpermute_with_ep_grad.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1146 | 1336 | [xla_npu/dependency/header/acl_nn_op/aclnn_moe_token_unpermute_with_routing_map_grad.h](../xla_npu/dependency/header/acl_nn_op/aclnn_moe_token_unpermute_with_routing_map_grad.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1147 | 1337 | [xla_npu/dependency/header/acl_nn_op/aclnn_mrgba_custom.h](../xla_npu/dependency/header/acl_nn_op/aclnn_mrgba_custom.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1148 | 1095 | [xla_npu/dependency/header/acl_nn_op/aclnn_non_max_suppression.h](../xla_npu/dependency/header/acl_nn_op/aclnn_non_max_suppression.h) | 40 | 0 | 0 | 0 | 0 | 0 | 40 |
| 1149 | 1338 | [xla_npu/dependency/header/acl_nn_op/aclnn_nsa_compress.h](../xla_npu/dependency/header/acl_nn_op/aclnn_nsa_compress.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1150 | 1339 | [xla_npu/dependency/header/acl_nn_op/aclnn_nsa_compress_grad.h](../xla_npu/dependency/header/acl_nn_op/aclnn_nsa_compress_grad.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1151 | 1340 | [xla_npu/dependency/header/acl_nn_op/aclnn_nsa_compress_with_cache.h](../xla_npu/dependency/header/acl_nn_op/aclnn_nsa_compress_with_cache.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1152 | 142 | [xla_npu/dependency/header/acl_nn_op/aclnn_ops_infer.h](../xla_npu/dependency/header/acl_nn_op/aclnn_ops_infer.h) | 245 | 0 | 0 | 0 | 0 | 0 | 245 |
| 1153 | 386 | [xla_npu/dependency/header/acl_nn_op/aclnn_ops_train.h](../xla_npu/dependency/header/acl_nn_op/aclnn_ops_train.h) | 133 | 0 | 0 | 0 | 0 | 0 | 133 |
| 1154 | 1052 | [xla_npu/dependency/header/acl_nn_op/aclnn_prompt_flash_attention.h](../xla_npu/dependency/header/acl_nn_op/aclnn_prompt_flash_attention.h) | 44 | 0 | 0 | 0 | 0 | 0 | 44 |
| 1155 | 911 | [xla_npu/dependency/header/acl_nn_op/aclnn_prompt_flash_attention_v2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_prompt_flash_attention_v2.h) | 51 | 0 | 0 | 0 | 0 | 0 | 51 |
| 1156 | 912 | [xla_npu/dependency/header/acl_nn_op/aclnn_prompt_flash_attention_v3.h](../xla_npu/dependency/header/acl_nn_op/aclnn_prompt_flash_attention_v3.h) | 51 | 0 | 0 | 0 | 0 | 0 | 51 |
| 1157 | 1341 | [xla_npu/dependency/header/acl_nn_op/aclnn_quant_matmul.h](../xla_npu/dependency/header/acl_nn_op/aclnn_quant_matmul.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1158 | 1133 | [xla_npu/dependency/header/acl_nn_op/aclnn_rand.h](../xla_npu/dependency/header/acl_nn_op/aclnn_rand.h) | 36 | 0 | 0 | 0 | 0 | 0 | 36 |
| 1159 | 1342 | [xla_npu/dependency/header/acl_nn_op/aclnn_repeat_interleave_grad.h](../xla_npu/dependency/header/acl_nn_op/aclnn_repeat_interleave_grad.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1160 | 1343 | [xla_npu/dependency/header/acl_nn_op/aclnn_ring_attention_update.h](../xla_npu/dependency/header/acl_nn_op/aclnn_ring_attention_update.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1161 | 1344 | [xla_npu/dependency/header/acl_nn_op/aclnn_rms_norm.h](../xla_npu/dependency/header/acl_nn_op/aclnn_rms_norm.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1162 | 1345 | [xla_npu/dependency/header/acl_nn_op/aclnn_rms_norm_grad.h](../xla_npu/dependency/header/acl_nn_op/aclnn_rms_norm_grad.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1163 | 1152 | [xla_npu/dependency/header/acl_nn_op/aclnn_rms_norm_quant.h](../xla_npu/dependency/header/acl_nn_op/aclnn_rms_norm_quant.h) | 33 | 0 | 0 | 0 | 0 | 0 | 33 |
| 1164 | 1346 | [xla_npu/dependency/header/acl_nn_op/aclnn_roi_align_rotated.h](../xla_npu/dependency/header/acl_nn_op/aclnn_roi_align_rotated.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1165 | 1347 | [xla_npu/dependency/header/acl_nn_op/aclnn_roi_align_rotated_grad.h](../xla_npu/dependency/header/acl_nn_op/aclnn_roi_align_rotated_grad.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1166 | 1348 | [xla_npu/dependency/header/acl_nn_op/aclnn_rotary_position_embedding_grad.h](../xla_npu/dependency/header/acl_nn_op/aclnn_rotary_position_embedding_grad.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1167 | 1349 | [xla_npu/dependency/header/acl_nn_op/aclnn_scatter_pa_cache.h](../xla_npu/dependency/header/acl_nn_op/aclnn_scatter_pa_cache.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1168 | 1350 | [xla_npu/dependency/header/acl_nn_op/aclnn_scatter_pa_kv_cache.h](../xla_npu/dependency/header/acl_nn_op/aclnn_scatter_pa_kv_cache.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1169 | 1351 | [xla_npu/dependency/header/acl_nn_op/aclnn_sparse_flash_attention_grad.h](../xla_npu/dependency/header/acl_nn_op/aclnn_sparse_flash_attention_grad.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1170 | 1352 | [xla_npu/dependency/header/acl_nn_op/aclnn_sparse_lightning_indexer_grad_kl_loss.h](../xla_npu/dependency/header/acl_nn_op/aclnn_sparse_lightning_indexer_grad_kl_loss.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1171 | 1353 | [xla_npu/dependency/header/acl_nn_op/aclnn_squared_relu.h](../xla_npu/dependency/header/acl_nn_op/aclnn_squared_relu.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1172 | 1354 | [xla_npu/dependency/header/acl_nn_op/aclnn_strided_slice_assign_v2.h](../xla_npu/dependency/header/acl_nn_op/aclnn_strided_slice_assign_v2.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1173 | 1355 | [xla_npu/dependency/header/acl_nn_op/aclnn_swi_glu.h](../xla_npu/dependency/header/acl_nn_op/aclnn_swi_glu.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1174 | 1356 | [xla_npu/dependency/header/acl_nn_op/aclnn_swi_glu_grad.h](../xla_npu/dependency/header/acl_nn_op/aclnn_swi_glu_grad.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1175 | 1357 | [xla_npu/dependency/header/acl_nn_op/aclnn_swin_attention_score_quant.h](../xla_npu/dependency/header/acl_nn_op/aclnn_swin_attention_score_quant.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1176 | 1358 | [xla_npu/dependency/header/acl_nn_op/aclnn_swin_transformer_ln_qkv_quant.h](../xla_npu/dependency/header/acl_nn_op/aclnn_swin_transformer_ln_qkv_quant.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1177 | 1359 | [xla_npu/dependency/header/acl_nn_op/aclnn_top_k_top_p_sample.h](../xla_npu/dependency/header/acl_nn_op/aclnn_top_k_top_p_sample.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1178 | 1360 | [xla_npu/dependency/header/acl_nn_op/aclnn_trans_quant_param.h](../xla_npu/dependency/header/acl_nn_op/aclnn_trans_quant_param.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1179 | 1361 | [xla_npu/dependency/header/acl_nn_op/aclnn_transform_bias_rescale_qkv.h](../xla_npu/dependency/header/acl_nn_op/aclnn_transform_bias_rescale_qkv.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1180 | 1362 | [xla_npu/dependency/header/acl_nn_op/aclnn_unfold_grad.h](../xla_npu/dependency/header/acl_nn_op/aclnn_unfold_grad.h) | 9 | 0 | 0 | 0 | 0 | 0 | 9 |
| 1181 | 1208 | [xla_npu/dependency/header/acl_nn_op/aclnn_util.h](../xla_npu/dependency/header/acl_nn_op/aclnn_util.h) | 20 | 0 | 0 | 0 | 0 | 0 | 20 |
| 1182 | 1214 | [xla_npu/dependency/header/ge/BUILD](../xla_npu/dependency/header/ge/BUILD) | 18 | 0 | 0 | 0 | 0 | 0 | 18 |
| 1183 | 394 | [xla_npu/dependency/header/ge/ge_api_error_codes.h](../xla_npu/dependency/header/ge/ge_api_error_codes.h) | 131 | 0 | 0 | 0 | 0 | 0 | 131 |
| 1184 | 30 | [xla_npu/dependency/header/ge/ge_api_types.h](../xla_npu/dependency/header/ge/ge_api_types.h) | 745 | 0 | 0 | 0 | 0 | 0 | 745 |
| 1185 | 880 | [xla_npu/dependency/header/ge/ge_continuous_tensor_list_api.h](../xla_npu/dependency/header/ge/ge_continuous_tensor_list_api.h) | 56 | 0 | 0 | 0 | 0 | 0 | 56 |
| 1186 | 688 | [xla_npu/dependency/header/ge/ge_data_flow_api.h](../xla_npu/dependency/header/ge/ge_data_flow_api.h) | 92 | 0 | 0 | 0 | 0 | 0 | 92 |
| 1187 | 724 | [xla_npu/dependency/header/ge/ge_error_codes.h](../xla_npu/dependency/header/ge/ge_error_codes.h) | 88 | 0 | 0 | 0 | 0 | 0 | 88 |
| 1188 | 806 | [xla_npu/dependency/header/ge/ge_feature_memory.h](../xla_npu/dependency/header/ge/ge_feature_memory.h) | 69 | 0 | 0 | 0 | 0 | 0 | 69 |
| 1189 | 87 | [xla_npu/dependency/header/ge/ge_graph_compile_summary.h](../xla_npu/dependency/header/ge/ge_graph_compile_summary.h) | 331 | 0 | 0 | 0 | 0 | 0 | 331 |
| 1190 | 131 | [xla_npu/dependency/header/ge/ge_ir_build.h](../xla_npu/dependency/header/ge/ge_ir_build.h) | 258 | 0 | 0 | 0 | 0 | 0 | 258 |
| 1191 | 1215 | [xla_npu/dependency/header/metadef/BUILD](../xla_npu/dependency/header/metadef/BUILD) | 18 | 0 | 0 | 0 | 0 | 0 | 18 |
| 1192 | 1103 | [xla_npu/dependency/header/metadef/exe_graph/runtime/base_type.h](../xla_npu/dependency/header/metadef/exe_graph/runtime/base_type.h) | 39 | 0 | 0 | 0 | 0 | 0 | 39 |
| 1193 | 454 | [xla_npu/dependency/header/metadef/exe_graph/runtime/exe_res_generation_context.h](../xla_npu/dependency/header/metadef/exe_graph/runtime/exe_res_generation_context.h) | 119 | 0 | 0 | 0 | 0 | 0 | 119 |
| 1194 | 1155 | [xla_npu/dependency/header/metadef/flow_graph/data_flow.h](../xla_npu/dependency/header/metadef/flow_graph/data_flow.h) | 32 | 0 | 0 | 0 | 0 | 0 | 32 |
| 1195 | 825 | [xla_npu/dependency/header/metadef/flow_graph/flow_attr.h](../xla_npu/dependency/header/metadef/flow_graph/flow_attr.h) | 66 | 0 | 0 | 0 | 0 | 0 | 66 |
| 1196 | 490 | [xla_npu/dependency/header/metadef/flow_graph/flow_graph.h](../xla_npu/dependency/header/metadef/flow_graph/flow_graph.h) | 115 | 0 | 0 | 0 | 0 | 0 | 115 |
| 1197 | 545 | [xla_npu/dependency/header/metadef/flow_graph/process_point.h](../xla_npu/dependency/header/metadef/flow_graph/process_point.h) | 109 | 0 | 0 | 0 | 0 | 0 | 109 |
| 1198 | 982 | [xla_npu/dependency/header/metadef/ge/framework/common/taskdown_common.h](../xla_npu/dependency/header/metadef/ge/framework/common/taskdown_common.h) | 47 | 0 | 0 | 0 | 0 | 0 | 47 |
| 1199 | 764 | [xla_npu/dependency/header/metadef/ge/ge_allocator.h](../xla_npu/dependency/header/metadef/ge/ge_allocator.h) | 81 | 0 | 0 | 0 | 0 | 0 | 81 |
| 1200 | 353 | [xla_npu/dependency/header/metadef/ge_common/ge_api_error_codes.h](../xla_npu/dependency/header/metadef/ge_common/ge_api_error_codes.h) | 144 | 0 | 0 | 0 | 0 | 0 | 144 |
| 1201 | 28 | [xla_npu/dependency/header/metadef/ge_common/ge_api_types.h](../xla_npu/dependency/header/metadef/ge_common/ge_api_types.h) | 792 | 0 | 0 | 0 | 0 | 0 | 792 |
| 1202 | 1061 | [xla_npu/dependency/header/metadef/ge_common/ge_error_codes.h](../xla_npu/dependency/header/metadef/ge_common/ge_error_codes.h) | 44 | 0 | 0 | 0 | 0 | 0 | 44 |
| 1203 | 785 | [xla_npu/dependency/header/metadef/graph/ascend_string.h](../xla_npu/dependency/header/metadef/graph/ascend_string.h) | 75 | 0 | 0 | 0 | 0 | 0 | 75 |
| 1204 | 741 | [xla_npu/dependency/header/metadef/graph/attr_value.h](../xla_npu/dependency/header/metadef/graph/attr_value.h) | 86 | 0 | 0 | 0 | 0 | 0 | 86 |
| 1205 | 1156 | [xla_npu/dependency/header/metadef/graph/compiler_def.h](../xla_npu/dependency/header/metadef/graph/compiler_def.h) | 32 | 0 | 0 | 0 | 0 | 0 | 32 |
| 1206 | 773 | [xla_npu/dependency/header/metadef/graph/ge_error_codes.h](../xla_npu/dependency/header/metadef/graph/ge_error_codes.h) | 78 | 0 | 0 | 0 | 0 | 0 | 78 |
| 1207 | 301 | [xla_npu/dependency/header/metadef/graph/gnode.h](../xla_npu/dependency/header/metadef/graph/gnode.h) | 159 | 0 | 0 | 0 | 0 | 0 | 159 |
| 1208 | 294 | [xla_npu/dependency/header/metadef/graph/graph.h](../xla_npu/dependency/header/metadef/graph/graph.h) | 161 | 0 | 0 | 0 | 0 | 0 | 161 |
| 1209 | 894 | [xla_npu/dependency/header/metadef/graph/graph_buffer.h](../xla_npu/dependency/header/metadef/graph/graph_buffer.h) | 53 | 0 | 0 | 0 | 0 | 0 | 53 |
| 1210 | 319 | [xla_npu/dependency/header/metadef/graph/inference_context.h](../xla_npu/dependency/header/metadef/graph/inference_context.h) | 152 | 0 | 0 | 0 | 0 | 0 | 152 |
| 1211 | 439 | [xla_npu/dependency/header/metadef/graph/operator_factory.h](../xla_npu/dependency/header/metadef/graph/operator_factory.h) | 123 | 0 | 0 | 0 | 0 | 0 | 123 |
| 1212 | 31 | [xla_npu/dependency/header/metadef/graph/operator_reg.h](../xla_npu/dependency/header/metadef/graph/operator_reg.h) | 737 | 0 | 0 | 0 | 0 | 0 | 737 |
| 1213 | 1138 | [xla_npu/dependency/header/metadef/graph/resource_context.h](../xla_npu/dependency/header/metadef/graph/resource_context.h) | 35 | 0 | 0 | 0 | 0 | 0 | 35 |
| 1214 | 63 | [xla_npu/dependency/header/metadef/graph/types.h](../xla_npu/dependency/header/metadef/graph/types.h) | 436 | 0 | 0 | 0 | 0 | 0 | 436 |
| 1215 | 1029 | [xla_npu/dependency/header/metadef/graph/utils/type_utils.h](../xla_npu/dependency/header/metadef/graph/utils/type_utils.h) | 45 | 0 | 0 | 0 | 0 | 0 | 45 |
| 1216 | 792 | [xla_npu/dependency/header/metadef/hcom/hcom_topo_info.h](../xla_npu/dependency/header/metadef/hcom/hcom_topo_info.h) | 74 | 0 | 0 | 0 | 0 | 0 | 74 |
| 1217 | 1083 | [xla_npu/dependency/header/metadef/op_common/common_infershape_fns.h](../xla_npu/dependency/header/metadef/op_common/common_infershape_fns.h) | 42 | 0 | 0 | 0 | 0 | 0 | 42 |
| 1218 | 172 | [xla_npu/dependency/header/metadef/op_common/data_type_utils.h](../xla_npu/dependency/header/metadef/op_common/data_type_utils.h) | 213 | 0 | 0 | 0 | 0 | 0 | 213 |
| 1219 | 1134 | [xla_npu/dependency/header/metadef/op_common/op_dev.h](../xla_npu/dependency/header/metadef/op_common/op_dev.h) | 36 | 0 | 0 | 0 | 0 | 0 | 36 |
| 1220 | 1135 | [xla_npu/dependency/header/metadef/op_common/op_error_code.h](../xla_npu/dependency/header/metadef/op_common/op_error_code.h) | 36 | 0 | 0 | 0 | 0 | 0 | 36 |
| 1221 | 175 | [xla_npu/dependency/header/metadef/platform/platform_info.h](../xla_npu/dependency/header/metadef/platform/platform_info.h) | 211 | 0 | 0 | 0 | 0 | 0 | 211 |
| 1222 | 240 | [xla_npu/dependency/header/metadef/platform/platform_info_def.h](../xla_npu/dependency/header/metadef/platform/platform_info_def.h) | 177 | 0 | 0 | 0 | 0 | 0 | 177 |
| 1223 | 428 | [xla_npu/dependency/header/metadef/platform/platform_infos_def.h](../xla_npu/dependency/header/metadef/platform/platform_infos_def.h) | 126 | 0 | 0 | 0 | 0 | 0 | 126 |
| 1224 | 747 | [xla_npu/dependency/header/metadef/register/ffts_node_calculater_registry.h](../xla_npu/dependency/header/metadef/register/ffts_node_calculater_registry.h) | 85 | 0 | 0 | 0 | 0 | 0 | 85 |
| 1225 | 826 | [xla_npu/dependency/header/metadef/register/hidden_input_func_registry.h](../xla_npu/dependency/header/metadef/register/hidden_input_func_registry.h) | 66 | 0 | 0 | 0 | 0 | 0 | 66 |
| 1226 | 1096 | [xla_npu/dependency/header/metadef/register/op_compile_info_base.h](../xla_npu/dependency/header/metadef/register/op_compile_info_base.h) | 40 | 0 | 0 | 0 | 0 | 0 | 40 |
| 1227 | 941 | [xla_npu/dependency/header/metadef/register/op_ct_impl_kernel_registry.h](../xla_npu/dependency/header/metadef/register/op_ct_impl_kernel_registry.h) | 49 | 0 | 0 | 0 | 0 | 0 | 49 |
| 1228 | 761 | [xla_npu/dependency/header/metadef/register/op_ct_impl_registry.h](../xla_npu/dependency/header/metadef/register/op_ct_impl_registry.h) | 82 | 0 | 0 | 0 | 0 | 0 | 82 |
| 1229 | 889 | [xla_npu/dependency/header/metadef/register/op_ct_impl_registry_api.h](../xla_npu/dependency/header/metadef/register/op_ct_impl_registry_api.h) | 54 | 0 | 0 | 0 | 0 | 0 | 54 |
| 1230 | 59 | [xla_npu/dependency/header/metadef/register/op_def.h](../xla_npu/dependency/header/metadef/register/op_def.h) | 450 | 0 | 0 | 0 | 0 | 0 | 450 |
| 1231 | 963 | [xla_npu/dependency/header/metadef/register/op_def_factory.h](../xla_npu/dependency/header/metadef/register/op_def_factory.h) | 48 | 0 | 0 | 0 | 0 | 0 | 48 |
| 1232 | 774 | [xla_npu/dependency/header/metadef/register/op_def_registry.h](../xla_npu/dependency/header/metadef/register/op_def_registry.h) | 78 | 0 | 0 | 0 | 0 | 0 | 78 |
| 1233 | 373 | [xla_npu/dependency/header/metadef/register/op_impl_registry.h](../xla_npu/dependency/header/metadef/register/op_impl_registry.h) | 138 | 0 | 0 | 0 | 0 | 0 | 138 |
| 1234 | 890 | [xla_npu/dependency/header/metadef/register/op_impl_registry_api.h](../xla_npu/dependency/header/metadef/register/op_impl_registry_api.h) | 54 | 0 | 0 | 0 | 0 | 0 | 54 |
| 1235 | 714 | [xla_npu/dependency/header/metadef/register/op_info_record_registry.h](../xla_npu/dependency/header/metadef/register/op_info_record_registry.h) | 89 | 0 | 0 | 0 | 0 | 0 | 89 |
| 1236 | 924 | [xla_npu/dependency/header/metadef/register/op_tiling_attr_utils.h](../xla_npu/dependency/header/metadef/register/op_tiling_attr_utils.h) | 50 | 0 | 0 | 0 | 0 | 0 | 50 |
| 1237 | 220 | [xla_npu/dependency/header/metadef/register/op_tiling_info.h](../xla_npu/dependency/header/metadef/register/op_tiling_info.h) | 186 | 0 | 0 | 0 | 0 | 0 | 186 |
| 1238 | 205 | [xla_npu/dependency/header/metadef/register/op_tiling_registry.h](../xla_npu/dependency/header/metadef/register/op_tiling_registry.h) | 192 | 0 | 0 | 0 | 0 | 0 | 192 |
| 1239 | 116 | [xla_npu/dependency/header/metadef/register/register.h](../xla_npu/dependency/header/metadef/register/register.h) | 280 | 0 | 0 | 0 | 0 | 0 | 280 |
| 1240 | 904 | [xla_npu/dependency/header/metadef/register/register_error_codes.h](../xla_npu/dependency/header/metadef/register/register_error_codes.h) | 52 | 0 | 0 | 0 | 0 | 0 | 52 |
| 1241 | 1030 | [xla_npu/dependency/header/metadef/register/register_fmk_types.h](../xla_npu/dependency/header/metadef/register/register_fmk_types.h) | 45 | 0 | 0 | 0 | 0 | 0 | 45 |
| 1242 | 793 | [xla_npu/dependency/header/metadef/register/register_pass.h](../xla_npu/dependency/header/metadef/register/register_pass.h) | 74 | 0 | 0 | 0 | 0 | 0 | 74 |
| 1243 | 797 | [xla_npu/dependency/header/metadef/register/register_types.h](../xla_npu/dependency/header/metadef/register/register_types.h) | 72 | 0 | 0 | 0 | 0 | 0 | 72 |
| 1244 | 54 | [xla_npu/dependency/header/metadef/register/scope/scope_fusion_pass_register.h](../xla_npu/dependency/header/metadef/register/scope/scope_fusion_pass_register.h) | 478 | 0 | 0 | 0 | 0 | 0 | 478 |
| 1245 | 108 | [xla_npu/dependency/header/metadef/register/tilingdata_base.h](../xla_npu/dependency/header/metadef/register/tilingdata_base.h) | 290 | 0 | 0 | 0 | 0 | 0 | 290 |
| 1246 | 212 | [xla_npu/dependency/header/metadef/register/tuning_tiling_reflection_utils.h](../xla_npu/dependency/header/metadef/register/tuning_tiling_reflection_utils.h) | 189 | 0 | 0 | 0 | 0 | 0 | 189 |
| 1247 | 516 | [xla_npu/dependency/header/metadef/register/tuning_tiling_registry.h](../xla_npu/dependency/header/metadef/register/tuning_tiling_registry.h) | 112 | 0 | 0 | 0 | 0 | 0 | 112 |
| 1248 | 530 | [xla_npu/dependency/header/metadef/utils/extern_math_util.h](../xla_npu/dependency/header/metadef/utils/extern_math_util.h) | 111 | 0 | 0 | 0 | 0 | 0 | 111 |
| 1249 | 895 | [xla_npu/dependency/stub/BUILD](../xla_npu/dependency/stub/BUILD) | 53 | 0 | 0 | 0 | 0 | 0 | 53 |
| 1250 | 370 | [xla_npu/dependency/stub/graph_stub.cc](../xla_npu/dependency/stub/graph_stub.cc) | 140 | 0 | 0 | 0 | 0 | 0 | 140 |
| 1251 | 1153 | [xla_npu/dependency/stub/operator_factory_stub.cc](../xla_npu/dependency/stub/operator_factory_stub.cc) | 33 | 0 | 0 | 0 | 0 | 0 | 33 |
| 1252 | 276 | [xla_npu/dependency/stub/operator_stub.cc](../xla_npu/dependency/stub/operator_stub.cc) | 167 | 0 | 0 | 0 | 0 | 0 | 167 |
| 1253 | 1142 | [xla_npu/dependency/tool/BUILD](../xla_npu/dependency/tool/BUILD) | 34 | 0 | 0 | 0 | 0 | 0 | 34 |
| 1254 | 576 | [xla_npu/dependency/tool/config_loader.cc](../xla_npu/dependency/tool/config_loader.cc) | 106 | 0 | 0 | 0 | 0 | 0 | 106 |
| 1255 | 715 | [xla_npu/dependency/tool/config_loader.h](../xla_npu/dependency/tool/config_loader.h) | 89 | 0 | 0 | 0 | 0 | 0 | 89 |
| 1256 | 451 | [xla_npu/dependency/tool/function_loader.cc](../xla_npu/dependency/tool/function_loader.cc) | 120 | 0 | 0 | 0 | 0 | 0 | 120 |
| 1257 | 862 | [xla_npu/pjrt/air/BUILD](../xla_npu/pjrt/air/BUILD) | 59 | 0 | 0 | 0 | 0 | 0 | 59 |
| 1258 | 1126 | [xla_npu/pjrt/air/host_memory_spaces.cc](../xla_npu/pjrt/air/host_memory_spaces.cc) | 37 | 0 | 0 | 0 | 0 | 0 | 37 |
| 1259 | 891 | [xla_npu/pjrt/air/host_memory_spaces.h](../xla_npu/pjrt/air/host_memory_spaces.h) | 54 | 0 | 0 | 0 | 0 | 0 | 54 |
| 1260 | 807 | [xla_npu/pjrt/air/mair/ir/BUILD](../xla_npu/pjrt/air/mair/ir/BUILD) | 69 | 0 | 0 | 0 | 0 | 0 | 69 |
| 1261 | 1062 | [xla_npu/pjrt/air/mair/ir/mair_ops.cc](../xla_npu/pjrt/air/mair/ir/mair_ops.cc) | 44 | 0 | 0 | 0 | 0 | 0 | 44 |
| 1262 | 1031 | [xla_npu/pjrt/air/mair/ir/mair_ops.h](../xla_npu/pjrt/air/mair/ir/mair_ops.h) | 45 | 0 | 0 | 0 | 0 | 0 | 45 |
| 1263 | 914 | [xla_npu/pjrt/air/mair/ir/mair_ops_base.td](../xla_npu/pjrt/air/mair/ir/mair_ops_base.td) | 51 | 0 | 0 | 0 | 0 | 0 | 51 |
| 1264 | 915 | [xla_npu/pjrt/air/mair/test/BUILD](../xla_npu/pjrt/air/mair/test/BUILD) | 51 | 0 | 0 | 0 | 0 | 0 | 51 |
| 1265 | 1219 | [xla_npu/pjrt/air/mair/test/CMakeLists.txt](../xla_npu/pjrt/air/mair/test/CMakeLists.txt) | 13 | 0 | 0 | 0 | 0 | 0 | 13 |
| 1266 | 882 | [xla_npu/pjrt/air/mair/test/test_mair_gen.h](../xla_npu/pjrt/air/mair/test/test_mair_gen.h) | 56 | 0 | 0 | 0 | 0 | 0 | 56 |
| 1267 | 886 | [xla_npu/pjrt/air/mair/transforms/BUILD](../xla_npu/pjrt/air/mair/transforms/BUILD) | 55 | 0 | 0 | 0 | 0 | 0 | 55 |
| 1268 | 1187 | [xla_npu/pjrt/air/mair/transforms/mair_passes.h](../xla_npu/pjrt/air/mair/transforms/mair_passes.h) | 28 | 0 | 0 | 0 | 0 | 0 | 28 |
| 1269 | 1207 | [xla_npu/pjrt/air/mair/transforms/mair_passes.td](../xla_npu/pjrt/air/mair/transforms/mair_passes.td) | 21 | 0 | 0 | 0 | 0 | 0 | 21 |
| 1270 | 965 | [xla_npu/pjrt/air/mair/translations/BUILD](../xla_npu/pjrt/air/mair/translations/BUILD) | 48 | 0 | 0 | 0 | 0 | 0 | 48 |
| 1271 | 1363 | [xla_npu/pjrt/air/mair/translations/CMakeLists.txt](../xla_npu/pjrt/air/mair/translations/CMakeLists.txt) | 8 | 0 | 0 | 0 | 0 | 0 | 8 |
| 1272 | 272 | [xla_npu/pjrt/air/mair/translations/convert_tensor.cc](../xla_npu/pjrt/air/mair/translations/convert_tensor.cc) | 168 | 0 | 0 | 0 | 0 | 0 | 168 |
| 1273 | 1127 | [xla_npu/pjrt/air/mair/translations/convert_tensor.h](../xla_npu/pjrt/air/mair/translations/convert_tensor.h) | 37 | 0 | 0 | 0 | 0 | 0 | 37 |
| 1274 | 499 | [xla_npu/pjrt/air/mair/translations/convert_type.cc](../xla_npu/pjrt/air/mair/translations/convert_type.cc) | 114 | 0 | 0 | 0 | 0 | 0 | 114 |
| 1275 | 1071 | [xla_npu/pjrt/air/mair/translations/convert_type.h](../xla_npu/pjrt/air/mair/translations/convert_type.h) | 43 | 0 | 0 | 0 | 0 | 0 | 43 |
| 1276 | 641 | [xla_npu/pjrt/air/mair/translations/status.cc](../xla_npu/pjrt/air/mair/translations/status.cc) | 98 | 0 | 0 | 0 | 0 | 0 | 98 |
| 1277 | 392 | [xla_npu/pjrt/air/mair/translations/status.h](../xla_npu/pjrt/air/mair/translations/status.h) | 132 | 0 | 0 | 0 | 0 | 0 | 132 |
| 1278 | 1158 | [xla_npu/pjrt/air/mair/translations/utils.h](../xla_npu/pjrt/air/mair/translations/utils.h) | 32 | 0 | 0 | 0 | 0 | 0 | 32 |
| 1279 | 491 | [xla_npu/pjrt/air/npu/abstract_npu_buffer.cc](../xla_npu/pjrt/air/npu/abstract_npu_buffer.cc) | 115 | 0 | 0 | 0 | 0 | 0 | 115 |
| 1280 | 658 | [xla_npu/pjrt/air/npu/abstract_npu_buffer.h](../xla_npu/pjrt/air/npu/abstract_npu_buffer.h) | 96 | 0 | 0 | 0 | 0 | 0 | 96 |
| 1281 | 1188 | [xla_npu/pjrt/air/npu/ge/BUILD](../xla_npu/pjrt/air/npu/ge/BUILD) | 28 | 0 | 0 | 0 | 0 | 0 | 28 |
| 1282 | 925 | [xla_npu/pjrt/air/npu/ge/graph_data.h](../xla_npu/pjrt/air/npu/ge/graph_data.h) | 50 | 0 | 0 | 0 | 0 | 0 | 50 |
| 1283 | 67 | [xla_npu/pjrt/air/npu/npu_client.h](../xla_npu/pjrt/air/npu/npu_client.h) | 416 | 0 | 0 | 0 | 0 | 0 | 416 |
| 1284 | 735 | [xla_npu/pjrt/air/npu/npu_device.cc](../xla_npu/pjrt/air/npu/npu_device.cc) | 87 | 0 | 0 | 0 | 0 | 0 | 87 |
| 1285 | 642 | [xla_npu/pjrt/air/npu/npu_device.h](../xla_npu/pjrt/air/npu/npu_device.h) | 98 | 0 | 0 | 0 | 0 | 0 | 98 |
| 1286 | 1194 | [xla_npu/pjrt/air/npu/npu_event.h](../xla_npu/pjrt/air/npu/npu_event.h) | 27 | 0 | 0 | 0 | 0 | 0 | 27 |
| 1287 | 88 | [xla_npu/pjrt/air/npu/raw_buffer.cc](../xla_npu/pjrt/air/npu/raw_buffer.cc) | 331 | 0 | 0 | 0 | 0 | 0 | 331 |
| 1288 | 266 | [xla_npu/pjrt/air/npu/raw_buffer.h](../xla_npu/pjrt/air/npu/raw_buffer.h) | 169 | 0 | 0 | 0 | 0 | 0 | 169 |
| 1289 | 696 | [xla_npu/pjrt/air/npu/service/BUILD](../xla_npu/pjrt/air/npu/service/BUILD) | 91 | 0 | 0 | 0 | 0 | 0 | 91 |
| 1290 | 1104 | [xla_npu/pjrt/air/npu/service/npu_buffer_assignment.cc](../xla_npu/pjrt/air/npu/service/npu_buffer_assignment.cc) | 39 | 0 | 0 | 0 | 0 | 0 | 39 |
| 1291 | 1032 | [xla_npu/pjrt/air/npu/service/npu_buffer_assignment.h](../xla_npu/pjrt/air/npu/service/npu_buffer_assignment.h) | 45 | 0 | 0 | 0 | 0 | 0 | 45 |
| 1292 | 374 | [xla_npu/pjrt/air/npu/service/npu_executable.h](../xla_npu/pjrt/air/npu/service/npu_executable.h) | 138 | 0 | 0 | 0 | 0 | 0 | 138 |
| 1293 | 83 | [xla_npu/pjrt/air/npu/tracked_npu_device_buffer.cc](../xla_npu/pjrt/air/npu/tracked_npu_device_buffer.cc) | 341 | 0 | 0 | 0 | 0 | 0 | 341 |
| 1294 | 168 | [xla_npu/pjrt/air/npu/tracked_npu_device_buffer.h](../xla_npu/pjrt/air/npu/tracked_npu_device_buffer.h) | 215 | 0 | 0 | 0 | 0 | 0 | 215 |
| 1295 | 114 | [xla_npu/pjrt/air/npu_tool.h](../xla_npu/pjrt/air/npu_tool.h) | 281 | 0 | 0 | 0 | 0 | 0 | 281 |
| 1296 | 1189 | [xla_npu/pjrt/air/pjrt_common.h](../xla_npu/pjrt/air/pjrt_common.h) | 28 | 0 | 0 | 0 | 0 | 0 | 28 |
| 1297 | 1209 | [xla_npu/pjrt/c/pjrt_c_api_npu.cc](../xla_npu/pjrt/c/pjrt_c_api_npu.cc) | 20 | 0 | 0 | 0 | 0 | 0 | 20 |
| 1298 | 1159 | [xla_npu/pjrt/c/pjrt_c_api_npu.h](../xla_npu/pjrt/c/pjrt_c_api_npu.h) | 32 | 0 | 0 | 0 | 0 | 0 | 32 |
| 1299 | 652 | [xla_npu/pjrt/c/pjrt_c_api_npu_internal.cc](../xla_npu/pjrt/c/pjrt_c_api_npu_internal.cc) | 97 | 0 | 0 | 0 | 0 | 0 | 97 |
| 1300 | 1195 | [xla_npu/pjrt/c/pjrt_c_api_npu_internal.h](../xla_npu/pjrt/c/pjrt_c_api_npu_internal.h) | 27 | 0 | 0 | 0 | 0 | 0 | 27 |
| 1301 | 1222 | [xla_npu/pjrt/c/pjrt_c_api_npu_version_script.lds](../xla_npu/pjrt/c/pjrt_c_api_npu_version_script.lds) | 12 | 0 | 0 | 0 | 0 | 0 | 12 |
| 1302 | 883 | [xla_npu/pjrt/npu/npu_pjrt_stream_executor_client.cc](../xla_npu/pjrt/npu/npu_pjrt_stream_executor_client.cc) | 56 | 0 | 0 | 0 | 0 | 0 | 56 |
| 1303 | 871 | [xla_npu/pjrt/plugin/BUILD](../xla_npu/pjrt/plugin/BUILD) | 58 | 0 | 0 | 0 | 0 | 0 | 58 |
| 1304 | 1098 | [xla_npu/pjrt/plugin/npu_device_description.cc](../xla_npu/pjrt/plugin/npu_device_description.cc) | 40 | 0 | 0 | 0 | 0 | 0 | 40 |
| 1305 | 863 | [xla_npu/pjrt/plugin/npu_device_description.h](../xla_npu/pjrt/plugin/npu_device_description.h) | 59 | 0 | 0 | 0 | 0 | 0 | 59 |
| 1306 | 1200 | [xla_npu/pjrt/plugin/npu_topology.cc](../xla_npu/pjrt/plugin/npu_topology.cc) | 24 | 0 | 0 | 0 | 0 | 0 | 24 |
| 1307 | 796 | [xla_npu/pjrt/plugin/npu_topology.h](../xla_npu/pjrt/plugin/npu_topology.h) | 73 | 0 | 0 | 0 | 0 | 0 | 73 |
| 1308 | 762 | [xla_npu/pjrt/plugin/npu_topology_description.cc](../xla_npu/pjrt/plugin/npu_topology_description.cc) | 82 | 0 | 0 | 0 | 0 | 0 | 82 |
| 1309 | 440 | [xla_npu/pjrt/plugin/npu_topology_description.h](../xla_npu/pjrt/plugin/npu_topology_description.h) | 123 | 0 | 0 | 0 | 0 | 0 | 123 |
| 1310 | 462 | [xla_npu/service/convert_to_afir/ascir_backend.h](../xla_npu/service/convert_to_afir/ascir_backend.h) | 118 | 0 | 0 | 0 | 0 | 0 | 118 |
| 1311 | 97 | [xla_npu/service/convert_to_afir/ascir_backend_test.cc](../xla_npu/service/convert_to_afir/ascir_backend_test.cc) | 310 | 0 | 0 | 0 | 0 | 0 | 310 |
| 1312 | 827 | [xla_npu/service/convert_to_afir/convert_to_afir.h](../xla_npu/service/convert_to_afir/convert_to_afir.h) | 66 | 0 | 0 | 0 | 0 | 0 | 66 |
| 1313 | 223 | [xla_npu/service/convert_to_afir/hlo_to_afir_tool.cc](../xla_npu/service/convert_to_afir/hlo_to_afir_tool.cc) | 185 | 0 | 0 | 0 | 0 | 0 | 185 |
| 1314 | 361 | [xla_npu/service/convert_to_afir/python_gil_guard.cc](../xla_npu/service/convert_to_afir/python_gil_guard.cc) | 142 | 0 | 0 | 0 | 0 | 0 | 142 |
| 1315 | 716 | [xla_npu/service/convert_to_afir/python_gil_guard.h](../xla_npu/service/convert_to_afir/python_gil_guard.h) | 89 | 0 | 0 | 0 | 0 | 0 | 89 |
| 1316 | 1113 | [xla_npu/service/convert_to_npuir/BUILD](../xla_npu/service/convert_to_npuir/BUILD) | 38 | 0 | 0 | 0 | 0 | 0 | 38 |
| 1317 | 1196 | [xla_npu/service/convert_to_npuir/npuir_backend.cc](../xla_npu/service/convert_to_npuir/npuir_backend.cc) | 26 | 0 | 0 | 0 | 0 | 0 | 26 |
| 1318 | 1114 | [xla_npu/service/convert_to_npuir/npuir_backend.h](../xla_npu/service/convert_to_npuir/npuir_backend.h) | 38 | 0 | 0 | 0 | 0 | 0 | 38 |
| 1319 | 942 | [xla_npu/service/kernel/BUILD](../xla_npu/service/kernel/BUILD) | 49 | 0 | 0 | 0 | 0 | 0 | 49 |
| 1320 | 358 | [xla_npu/service/npu_backend_config.proto](../xla_npu/service/npu_backend_config.proto) | 143 | 0 | 0 | 0 | 0 | 0 | 143 |
| 1321 | 673 | [xla_npu/service/npu_buffer_allocations.h](../xla_npu/service/npu_buffer_allocations.h) | 94 | 0 | 0 | 0 | 0 | 0 | 94 |
| 1322 | 795 | [xla_npu/service/npu_compile_module_to_llvm_ir.h](../xla_npu/service/npu_compile_module_to_llvm_ir.h) | 74 | 0 | 0 | 0 | 0 | 0 | 74 |
| 1323 | 111 | [xla_npu/service/npu_executable.h](../xla_npu/service/npu_executable.h) | 284 | 0 | 0 | 0 | 0 | 0 | 284 |
| 1324 | 124 | [xla_npu/service/npu_execution_stream_assignment.cc](../xla_npu/service/npu_execution_stream_assignment.cc) | 266 | 0 | 0 | 0 | 0 | 0 | 266 |
| 1325 | 697 | [xla_npu/service/npu_execution_stream_assignment.h](../xla_npu/service/npu_execution_stream_assignment.h) | 91 | 0 | 0 | 0 | 0 | 0 | 91 |
| 1326 | 160 | [xla_npu/service/npu_hlo_fusion_analysis.cc](../xla_npu/service/npu_hlo_fusion_analysis.cc) | 220 | 0 | 0 | 0 | 0 | 0 | 220 |
| 1327 | 400 | [xla_npu/service/npu_hlo_fusion_analysis.h](../xla_npu/service/npu_hlo_fusion_analysis.h) | 130 | 0 | 0 | 0 | 0 | 0 | 130 |
| 1328 | 780 | [xla_npu/service/npu_hlo_schedule.cc](../xla_npu/service/npu_hlo_schedule.cc) | 76 | 0 | 0 | 0 | 0 | 0 | 76 |
| 1329 | 983 | [xla_npu/service/npu_hlo_schedule.h](../xla_npu/service/npu_hlo_schedule.h) | 47 | 0 | 0 | 0 | 0 | 0 | 47 |
| 1330 | 393 | [xla_npu/service/npu_hlo_to_ir_bindings.cc](../xla_npu/service/npu_hlo_to_ir_bindings.cc) | 132 | 0 | 0 | 0 | 0 | 0 | 132 |
| 1331 | 757 | [xla_npu/service/npu_hlo_to_ir_bindings.h](../xla_npu/service/npu_hlo_to_ir_bindings.h) | 83 | 0 | 0 | 0 | 0 | 0 | 83 |
| 1332 | 94 | [xla_npu/service/npu_ir_emission_utils.cc](../xla_npu/service/npu_ir_emission_utils.cc) | 320 | 0 | 0 | 0 | 0 | 0 | 320 |
| 1333 | 235 | [xla_npu/service/npu_ir_emission_utils.h](../xla_npu/service/npu_ir_emission_utils.h) | 178 | 0 | 0 | 0 | 0 | 0 | 178 |
| 1334 | 173 | [xla_npu/service/npu_ir_emitter.cc](../xla_npu/service/npu_ir_emitter.cc) | 213 | 0 | 0 | 0 | 0 | 0 | 213 |
| 1335 | 432 | [xla_npu/service/npu_ir_emitter.h](../xla_npu/service/npu_ir_emitter.h) | 125 | 0 | 0 | 0 | 0 | 0 | 125 |
| 1336 | 689 | [xla_npu/service/npu_ir_emitter_context.cc](../xla_npu/service/npu_ir_emitter_context.cc) | 92 | 0 | 0 | 0 | 0 | 0 | 92 |
| 1337 | 320 | [xla_npu/service/npu_ir_emitter_context.h](../xla_npu/service/npu_ir_emitter_context.h) | 152 | 0 | 0 | 0 | 0 | 0 | 152 |
| 1338 | 1033 | [xla_npu/service/npu_latency_hiding_scheduler.cc](../xla_npu/service/npu_latency_hiding_scheduler.cc) | 45 | 0 | 0 | 0 | 0 | 0 | 45 |
| 1339 | 1160 | [xla_npu/service/npu_latency_hiding_scheduler.h](../xla_npu/service/npu_latency_hiding_scheduler.h) | 32 | 0 | 0 | 0 | 0 | 0 | 32 |
| 1340 | 674 | [xla_npu/service/npu_stream_executor_util.cc](../xla_npu/service/npu_stream_executor_util.cc) | 94 | 0 | 0 | 0 | 0 | 0 | 94 |
| 1341 | 1003 | [xla_npu/service/npu_stream_executor_util.h](../xla_npu/service/npu_stream_executor_util.h) | 46 | 0 | 0 | 0 | 0 | 0 | 46 |
| 1342 | 864 | [xla_npu/service/npu_transfer_manager.cc](../xla_npu/service/npu_transfer_manager.cc) | 59 | 0 | 0 | 0 | 0 | 0 | 59 |
| 1343 | 798 | [xla_npu/service/npu_transfer_manager.h](../xla_npu/service/npu_transfer_manager.h) | 72 | 0 | 0 | 0 | 0 | 0 | 72 |
| 1344 | 156 | [xla_npu/service/transform/npu_copy_fusion.cc](../xla_npu/service/transform/npu_copy_fusion.cc) | 226 | 0 | 0 | 0 | 0 | 0 | 226 |
| 1345 | 852 | [xla_npu/service/transform/npu_copy_fusion.h](../xla_npu/service/transform/npu_copy_fusion.h) | 61 | 0 | 0 | 0 | 0 | 0 | 61 |
| 1346 | 66 | [xla_npu/service/transform/npu_dynamic_slice_fusion_rewriter.cc](../xla_npu/service/transform/npu_dynamic_slice_fusion_rewriter.cc) | 418 | 0 | 0 | 0 | 0 | 0 | 418 |
| 1347 | 770 | [xla_npu/service/transform/npu_dynamic_slice_fusion_rewriter.h](../xla_npu/service/transform/npu_dynamic_slice_fusion_rewriter.h) | 79 | 0 | 0 | 0 | 0 | 0 | 79 |
| 1348 | 418 | [xla_npu/service/transform/npu_fusion_dispatch_pipeline.cc](../xla_npu/service/transform/npu_fusion_dispatch_pipeline.cc) | 127 | 0 | 0 | 0 | 0 | 0 | 127 |
| 1349 | 355 | [xla_npu/service/transform/npu_fusion_dynamic_memcpy_rewriter.cc](../xla_npu/service/transform/npu_fusion_dynamic_memcpy_rewriter.cc) | 144 | 0 | 0 | 0 | 0 | 0 | 144 |
| 1350 | 1074 | [xla_npu/service/transform/npu_fusion_dynamic_memcpy_rewriter.h](../xla_npu/service/transform/npu_fusion_dynamic_memcpy_rewriter.h) | 43 | 0 | 0 | 0 | 0 | 0 | 43 |
| 1351 | 1105 | [xla_npu/service/transform/npu_fusion_pipeline.h](../xla_npu/service/transform/npu_fusion_pipeline.h) | 39 | 0 | 0 | 0 | 0 | 0 | 39 |
| 1352 | 120 | [xla_npu/service/transform/npu_gemm_fusion_swap_operands.cc](../xla_npu/service/transform/npu_gemm_fusion_swap_operands.cc) | 271 | 0 | 0 | 0 | 0 | 0 | 271 |
| 1353 | 943 | [xla_npu/service/transform/npu_hlo_printer_pass.h](../xla_npu/service/transform/npu_hlo_printer_pass.h) | 49 | 0 | 0 | 0 | 0 | 0 | 49 |
| 1354 | 1099 | [xla_npu/service/transform/npu_pipeline.h](../xla_npu/service/transform/npu_pipeline.h) | 40 | 0 | 0 | 0 | 0 | 0 | 40 |
| 1355 | 191 | [xla_npu/service/transform/npu_rename_fusions.cc](../xla_npu/service/transform/npu_rename_fusions.cc) | 201 | 0 | 0 | 0 | 0 | 0 | 201 |
| 1356 | 1034 | [xla_npu/service/transform/npu_rename_fusions.h](../xla_npu/service/transform/npu_rename_fusions.h) | 45 | 0 | 0 | 0 | 0 | 0 | 45 |
| 1357 | 966 | [xla_npu/stream_executor/npu_event.h](../xla_npu/stream_executor/npu_event.h) | 48 | 0 | 0 | 0 | 0 | 0 | 48 |
| 1358 | 479 | [xla_npu/stream_executor/npu_executor.h](../xla_npu/stream_executor/npu_executor.h) | 116 | 0 | 0 | 0 | 0 | 0 | 116 |
| 1359 | 389 | [xla_npu/stream_executor/npu_platform.cc](../xla_npu/stream_executor/npu_platform.cc) | 133 | 0 | 0 | 0 | 0 | 0 | 133 |
| 1360 | 786 | [xla_npu/stream_executor/npu_platform.h](../xla_npu/stream_executor/npu_platform.h) | 75 | 0 | 0 | 0 | 0 | 0 | 75 |
| 1361 | 1201 | [xla_npu/stream_executor/npu_platform_id.cc](../xla_npu/stream_executor/npu_platform_id.cc) | 24 | 0 | 0 | 0 | 0 | 0 | 24 |
| 1362 | 1161 | [xla_npu/stream_executor/npu_platform_id.h](../xla_npu/stream_executor/npu_platform_id.h) | 32 | 0 | 0 | 0 | 0 | 0 | 32 |
| 1363 | 808 | [xla_npu/stream_executor/npu_stream.h](../xla_npu/stream_executor/npu_stream.h) | 69 | 0 | 0 | 0 | 0 | 0 | 69 |
| 1364 | 803 | [xla_npu/stream_executor/npu_stream_factory.cc](../xla_npu/stream_executor/npu_stream_factory.cc) | 70 | 0 | 0 | 0 | 0 | 0 | 70 |
| 1365 | 837 | [xla_npu/stream_executor/npu_stream_factory.h](../xla_npu/stream_executor/npu_stream_factory.h) | 65 | 0 | 0 | 0 | 0 | 0 | 65 |

## 9. 复现统计

在仓库根目录运行下列只读脚本；固定提交确保之后 HEAD 变化不会改变报告快照。它输出完整文件排名及递归目录排名，可重算正文和附录中的累计数。已删除文件仍由历史自动收集。

```python
import collections
import subprocess

SNAPSHOT = "23271f4d188fcd8f93a6871872b6483363b04fec"
def git(*args):
    return subprocess.check_output(["git", *args])

roots = git("rev-list", "--max-parents=0", SNAPSHOT).decode().splitlines()
assert len(roots) == 1
assert not git("rev-list", "--merges", SNAPSHOT).strip()
root = roots[0]
files = collections.defaultdict(lambda: [0, 0, 0, 0])
for commit in git("rev-list", "--reverse", SNAPSHOT).decode().splitlines():
    raw = git("diff-tree", "--root", "--no-commit-id", "--numstat",
              "--no-renames", "-r", "-z", commit, "--", "xla_npu/")
    for record in raw.split(b"\0"):
        if not record:
            continue
        a, d, path = record.decode().split("\t", 2)
        assert a != "-", "binary file: report separately"
        a, d = int(a), int(d)
        if commit == root:
            assert d == 0
            files[path][0] += a
        else:
            files[path][1] += a
            files[path][2] += d
            files[path][3] += 1

def ranked(items):
    return sorted(items, key=lambda kv: (-(kv[1][1] + kv[1][2]), kv[0]))

print("FILE: path, initial, added, deleted, commits, churn, total")
for path, (initial, added, deleted, count) in ranked(files.items()):
    print(path, initial, added, deleted, count,
          added + deleted, initial + added + deleted, sep="\t")

dirs = collections.defaultdict(lambda: [0, 0, 0])
for path, values in files.items():
    parts = path.split("/")
    for end in range(2, len(parts)):
        directory = "/".join(parts[:end])
        for i in range(3):
            dirs[directory][i] += values[i]
print("DIRECTORY: path, initial, added, deleted, churn, total")
for path, (initial, added, deleted) in ranked(dirs.items()):
    print(path, initial, added, deleted, added + deleted,
          initial + added + deleted, sep="\t")
```

辅助校验命令：

```bash
git diff --numstat --no-renames d32d52c 23271f4 -- xla_npu/
git diff --check d32d52c 23271f4 -- xla_npu/
git show --format= --numstat --no-renames dbe12ec -- xla_npu/
git blame -L 160,179 23271f4 -- xla_npu/service/transform/npu_tril_utils.h
```

第二条命令预期返回非零，因为确实存在报告中列出的空白问题。
