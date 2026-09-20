# DotGeneralOp 到 Ascend Op 的优化转换

## 问题分析

从日志和错误信息分析，发现 Ascend 的 MatMul 操作对 batch 维度的处理存在问题：

**原始错误**：
```
OpName:[MatMul215] "[InferShape] The k-axis of a(8) and b(14) tensors must be the same"
```

**输入形状**：
- lhs: `[14, 8, 64]`
- rhs: `[1, 14, 64, 8]`

**转换后**：
- lhs: `[14, 8, 64]`
- rhs: `[14, 64, 8]`

**问题**：Ascend MatMul 将 `[14, 8, 64]` 解释为 K=8，将 `[14, 64, 8]` 解释为 K=14，导致 K 轴不匹配。

## 解决方案

### Ascend MatMul 操作对比

通过分析 Ascend 的 Op 定义，发现有以下几种 MatMul 操作：

1. **MatMul**：基本的矩阵乘法，可能不支持 batch 维度
   - 输入：x1, x2, bias (optional)
   - 属性：transpose_x1, transpose_x2
   - 适用于：2D 矩阵乘法 `[M, K] x [K, N] -> [M, N]`

2. **BatchMatMul**：专门支持 batch 维度的矩阵乘法
   - 输入：x1, x2
   - 属性：adj_x1, adj_x2
   - 适用于：batch 矩阵乘法 `[batch..., M, K] x [batch..., K, N] -> [batch..., M, N]`

3. **MatMulV2**：增强版本，支持更多数据类型
   - 输入：x1, x2, bias (optional), offset_w (optional)
   - 属性：transpose_x1, transpose_x2, offset_x
   - 适用于：需要更多数据类型支持的场景

### 优化策略

根据 StableHLO `dot_general` 的输入特征，选择最合适的 Ascend Op：

| 场景 | StableHLO dot_general | Ascend Op | 输入形状 |
|------|----------------------|-----------|----------|
| 无 batch 维度 | `contracting_dims = [1] x [0]` | MatMul | `[M, K] x [K, N]` |
| 有 batch 维度 | `batching_dims = [0] x [1]` | BatchMatMul | `[B, M, K] x [B, K, N]` |

## 实现细节

### 1. 添加 BatchMatMulOp 定义

在 `mair_ops.td` 中添加：

```td
def Air_BatchMatMulOp : Air_Op<"BatchMatMul", [Pure]> {
  let summary = "Batch matrix multiplication operation";
  let description = [{
    Performs batch matrix multiplication on two input tensors.
    Supports batch dimensions: [batch..., M, K] x [batch..., K, N] -> [batch..., M, N]
  }];

  let arguments = (ins
    Air_Tensor:$x1,
    Air_Tensor:$x2,
    DefaultValuedAttr<BoolAttr, "false">:$adj_x1,
    DefaultValuedAttr<BoolAttr, "false">:$adj_x2
  );

  let results = (outs
    Air_Tensor:$output
  );
}
```

### 2. 修改 ConvertMatMulOp

根据是否有 batch 维度选择不同的操作：

```cpp
if (!lhsBatchingDims.empty()) {
  // Has batch dimensions; use BatchMatMul
  lhsReshapeShape = {lhsBatchSize, lhsNonContractSize, lhsContractSize};
  rhsReshapeShape = {rhsBatchSize, rhsContractSize, rhsNonContractSize};
  
  matmulResultShape = {lhsBatchSize, lhsNonContractSize, rhsNonContractSize};
  matmulResult = rewriter.create<BatchMatMulOp>(
      op.getLoc(), matmulResultType, lhsReshaped, rhsReshaped, 
      false, false).getResult();
} else {
  // No batch dimensions; use MatMul
  lhsReshapeShape = {lhsNonContractSize, lhsContractSize};
  rhsReshapeShape = {rhsContractSize, rhsNonContractSize};
  
  matmulResultShape = {lhsNonContractSize, rhsNonContractSize};
  matmulResult = rewriter.create<MatMulOp>(
      op.getLoc(), matmulResultType, lhsReshaped, rhsReshaped, 
      nullptr, false, false).getResult();
}
```

### 3. 转换流程

#### 例子 1：有 batch 维度

**输入**：
```mlir
stablehlo.dot_general %299, %296, 
  batching_dims = [0] x [1], 
  contracting_dims = [2] x [2] 
  : (tensor<14x8x64xf32>, tensor<1x14x64x8xf32>) -> tensor<14x8x1x8xf32>
```

**转换步骤**：
1. 维度识别：
   - lhs: `[14, 8, 64]` → batch=14, M=8, K=64
   - rhs: `[1, 14, 64, 8]` → batch=14, K=64, N=8

2. Transpose：
   - lhs: `[14, 8, 64]` → `[14, 8, 64]` (无需转置)
   - rhs: `[1, 14, 64, 8]` → `[14, 64, 1, 8]` → `[14, 64, 8]`

3. Reshape：
   - lhs: `[14, 8, 64]` → `[14, 8, 64]`
   - rhs: `[14, 64, 8]` → `[14, 64, 8]`

4. BatchMatMul：
   - `[14, 8, 64]` x `[14, 64, 8]` → `[14, 8, 8]`

5. Reshape：
   - `[14, 8, 8]` → `[14, 8, 1, 8]`

#### 例子 2：无 batch 维度

**输入**：
```mlir
stablehlo.dot_general %24, %arg13, 
  contracting_dims = [2] x [0] 
  : (tensor<1x8x896xf32>, tensor<896x128xf32>) -> tensor<1x8x128xf32>
```

**转换步骤**：
1. 维度识别：
   - lhs: `[1, 8, 896]` → M=8, K=896
   - rhs: `[896, 128]` → K=896, N=128

2. Reshape：
   - lhs: `[1, 8, 896]` → `[8, 896]`
   - rhs: `[896, 128]` → `[896, 128]`

3. MatMul：
   - `[8, 896]` x `[896, 128]` → `[8, 128]`

4. Reshape：
   - `[8, 128]` → `[1, 8, 128]`

## 优势

1. **语义正确**：使用 BatchMatMul 正确处理 batch 维度
2. **性能优化**：避免不必要的维度展平和恢复操作
3. **代码清晰**：根据输入特征选择最合适的操作
4. **可扩展性**：易于添加更多 MatMul 变体的支持

## 修改的文件

1. **mair_ops.td**：添加 BatchMatMulOp 定义
2. **mair_passes.cc**：修改 ConvertMatMulOp，根据 batch 维度选择不同的操作

## 测试建议

建议创建以下测试用例：

1. **无 batch 维度的 dot_general** → 使用 MatMul
2. **有 batch 维度的 dot_general** → 使用 BatchMatMul
3. **多个 batch 维度的 dot_general** → 验证 BatchMatMul 的多 batch 支持
4. **边界情况**：维度大小为 1 的情况

## 总结

通过分析 Ascend 的不同 MatMul 操作，我们优化了 StableHLO `dot_general` 到 Ascend Op 的转换：

- **无 batch 维度**：使用 MatMul，保持原有的 2D 矩阵乘法语义
- **有 batch 维度**：使用 BatchMatMul，正确处理 batch 维度

这种优化不仅解决了 K 轴不匹配的问题，还提高了转换的效率和正确性。
