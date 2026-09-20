# GatherV2Op 实现总结

## 完成的工作

### 1. 在 mair_ops.td 中添加 GatherV2Op 定义

**文件**：[mair_ops.td](file:///home/code/xla/xla/xla_npu/pjrt/air/mair/ir/mair_ops.td#L573-L588)

```td
def Air_GatherV2Op : Air_Op<"GatherV2", [Pure]> {
  let summary = "GatherV2 operation";
  let description = [{
    Gathers slices from params axis according to indices.
    Similar to TensorFlow's tf.gather and PyTorch's torch.gather.
  }];
  let arguments = (ins
    Air_Tensor:$x,
    Air_Tensor:$indices,
    Air_Tensor:$axis,
    Optional<Air_Tensor>:$batch_dims
  );
  let results = (outs Air_Tensor);
}
```

**参数说明**：
- `x`: 输入张量
- `indices`: 索引张量
- `axis`: 指定收集的轴（标量张量）
- `batch_dims`: 可选的 batch 维度数

### 2. 实现 ConvertGatherOp 转换逻辑

**文件**：[mair_passes.cc](file:///home/code/xla/xla/xla_npu/pjrt/air/mair/transforms/mair_passes.cc#L846-L971)

**转换策略**：

针对简单的单轴 gather 情况（满足以下条件）：
1. `start_index_map` 只有一个元素
2. `collapsed_slice_dims` 只有一个元素
3. `collapsed_slice_dims[0] == start_index_map[0]`
4. `slice_sizes` 在 collapsed 维度上为 1

**转换步骤**：

1. **提取维度信息**：
   ```cpp
   auto offsetDims = dimensionNumbers.getOffsetDims();
   auto collapsedSliceDims = dimensionNumbers.getCollapsedSliceDims();
   auto startIndexMap = dimensionNumbers.getStartIndexMap();
   auto indexVectorDim = dimensionNumbers.getIndexVectorDim();
   ```

2. **创建 axis 常量**：
   ```cpp
   int64_t axis = startIndexMap[0];
   auto axisConst = rewriter.create<ConstantOp>(op.getLoc(), axisType, axisAttr);
   ```

3. **处理 indices（可能需要 squeeze）**：
   ```cpp
   // If index_vector_dim is the last dimension and has size 1, squeeze it
   if (indexVectorDim == indicesRank - 1 && indicesShape.back() == 1) {
     // Squeeze the last dimension
     indices = rewriter.create<ReshapeOp>(...);
   }
   ```

4. **创建 GatherV2Op**：
   ```cpp
   auto gatherResult = rewriter.create<GatherV2Op>(
       op.getLoc(), op.getType(), operand, indices, 
       axisConst.getResult(), nullptr);
   ```

**调试日志**：
- 打印转换前的 MLIR
- 打印维度信息（operand shape, indices shape, offset_dims, collapsed_slice_dims, start_index_map, index_vector_dim, slice_sizes）
- 打印转换后的 MLIR

### 3. 在 export_graphdef.cc 中添加 GatherV2 导出

**文件**：[export_graphdef.cc](file:///home/code/xla/xla/xla_npu/pjrt/air/mair/translations/export_graphdef.cc#L64)

**添加的映射**：
```cpp
{"GatherV2", {"x", "indices", "axis", "batch_dims"}},
```

**说明**：
- MLIR Op 到 GE Op 的转换是通用的，通过 `processOperationInputs` 函数使用 `inputNameMap` 来映射输入名称
- 添加这个映射后，GatherV2Op 会自动被正确导出为 Ascend 的 GatherV2 操作

## 转换示例

### 用户的例子

**输入**：
```mlir
%13 = "stablehlo.gather"(%arg0, %5) <{
  dimension_numbers = #stablehlo.gather<
    offset_dims = [2],
    collapsed_slice_dims = [0],
    start_index_map = [0],
    index_vector_dim = 2
  >,
  slice_sizes = array<i64: 1, 896>
}> : (tensor<151936x896xf32>, tensor<1x8x1xi32>) -> tensor<1x8x896xf32>
```

**转换步骤**：

1. **分析维度信息**：
   - operand: `[151936, 896]`
   - indices: `[1, 8, 1]`
   - start_index_map: `[0]` → axis=0
   - collapsed_slice_dims: `[0]`
   - index_vector_dim: 2（最后一个维度）
   - slice_sizes: `[1, 896]`

2. **Squeeze indices**：
   - `[1, 8, 1]` → `[1, 8]`（去掉最后一个维度）

3. **创建 GatherV2**：
   ```mlir
   %axis = "mair.Const"() {value = 0 : i64} : () -> tensor<i64>
   %result = "mair.GatherV2"(%arg0, %indices_squeezed, %axis) 
     : (tensor<151936x896xf32>, tensor<1x8xi32>, tensor<i64>) -> tensor<1x8x896xf32>
   ```

**语义解释**：
```python
# Pseudocode
result = zeros([1, 8, 896])
for i in range(1):
  for j in range(8):
    index = indices[i, j]  # Scalar index
    result[i, j, :] = operand[index, :]  # Gather slice [896]
```

## 限制和未来工作

### 当前限制

1. **只支持简单的单轴 gather**：
   - `start_index_map` 只有一个元素
   - `collapsed_slice_dims` 只有一个元素
   - 不支持多轴 gather

2. **不支持复杂的 gather 场景**：
   - 多个 batch 维度
   - 多个 collapsed 维度
   - 复杂的维度映射

### 未来工作

1. **支持更复杂的 gather 场景**：
   - 使用 GatherNd 处理多轴 gather
   - 组合多个操作处理复杂的维度映射

2. **优化性能**：
   - 减少 Reshape 操作
   - 直接使用更高效的 Ascend 操作

3. **添加更多测试用例**：
   - 多轴 gather
   - 不同维度组合
   - 边界情况

## 测试建议

建议创建以下测试用例：

1. **简单的单轴 gather**（当前支持）：
   ```mlir
   stablehlo.gather with start_index_map = [0], collapsed_slice_dims = [0]
   ```

2. **不同轴的 gather**：
   ```mlir
   stablehlo.gather with start_index_map = [1], collapsed_slice_dims = [1]
   ```

3. **多索引维度**：
   ```mlir
   stablehlo.gather with indices shape [batch, M, N, 1]
   ```

4. **边界情况**：
   - 空索引
   - 单元素索引
   - 大批量索引

## 总结

成功实现了 GatherV2Op 的定义和转换逻辑，支持简单的单轴 gather 场景。对于用户的例子，转换逻辑可以正确处理：

- ✓ 识别单轴 gather
- ✓ 创建 axis 常量
- ✓ Squeeze indices
- ✓ 创建 GatherV2Op
- ✓ 添加详细的调试日志

对于更复杂的 gather 场景，需要进一步扩展实现。
