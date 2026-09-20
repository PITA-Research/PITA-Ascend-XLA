# StableHLO GatherOp 到 Ascend Op 的转换分析

## 问题背景

用户提供的 stablehlo.gather 例子：
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

## StableHLO Gather 语义分析

### 输入
- **operand**: `tensor<151936x896xf32>` - 被收集的张量
- **start_indices**: `tensor<1x8x1xi32>` - 起始索引

### 参数
- **offset_dims = [2]**: 切片维度在结果中的位置
- **collapsed_slice_dims = [0]**: 被折叠的切片维度
- **start_index_map = [0]**: 索引映射到 operand 的维度
- **index_vector_dim = 2**: 索引向量的维度（最后一个维度）
- **slice_sizes = [1, 896]**: 每个切片的大小

### 输出
- **result**: `tensor<1x8x896xf32>`

### 语义解释

1. **索引解释**：
   - start_indices 形状: `[1, 8, 1]`
   - index_vector_dim = 2 表示最后一个维度是索引向量
   - 所以有 `1 * 8 = 8` 个索引，每个索引是标量（因为最后一个维度大小为 1）

2. **切片收集**：
   - 每个索引从 operand 的第 0 维收集一个切片
   - slice_sizes = [1, 896] 表示每个切片的大小
   - collapsed_slice_dims = [0] 表示切片的第 0 维被折叠
   - 所以实际收集的切片大小是 [896]（第 0 维被折叠）

3. **结果组装**：
   - offset_dims = [2] 表示切片维度在结果的位置 2
   - 结果形状: `[1, 8, 896]`
   - 其中 [1, 8] 来自 start_indices 的 batch 维度
   - 896 来自切片

### 计算过程

```python
# Pseudocode
result = zeros([1, 8, 896])
for i in range(1):
  for j in range(8):
    index = start_indices[i, j, 0]  # Scalar index
    slice = operand[index:index+1, :]  # [1, 896]
    slice_collapsed = squeeze(slice, dim=0)  # [896] (collapsed_slice_dims = [0])
    result[i, j, :] = slice_collapsed
```

## Ascend Gather 操作对比

### 1. GatherV2

```python
def GatherV2(x: Tensor, indices: Tensor, axis: int, batch_dims: int = 0):
    """
    REG_OP(GatherV2)
    .INPUT(x, Tensor)
    .INPUT(indices, Tensor)
    .INPUT(axis, Tensor)
    .OPTIONAL_INPUT(batch_dims, Tensor)
    .OUTPUT(y, Tensor)
    .ATTR(validate_indices, Bool, false)
    """
```

**特点**：
- 支持指定轴（axis）
- 支持 batch_dims
- 最接近 TensorFlow 的 tf.gather

**适用性**：★★★★★
- 可以处理简单的 gather 操作
- 需要将 stablehlo.gather 的复杂语义简化为 axis-based gather

### 2. GatherV2D

```python
def GatherV2D(x: Tensor, indices: Tensor, *, axis: int = 0, batch_dims: int = 0):
    """
    REG_OP(GatherV2D)
    .INPUT(x, Tensor)
    .INPUT(indices, Tensor)
    .ATTR(axis, Int, 0)
    .ATTR(batch_dims, Int, 0)
    .OUTPUT(y, Tensor)
    """
```

**特点**：
- axis 是属性而不是输入
- 其他与 GatherV2 相同

**适用性**：★★★★☆
- 与 GatherV2 类似，但 axis 必须是编译时常量

### 3. GatherNd

```python
def GatherNd(x: Tensor, indices: Tensor):
    """
    REG_OP(GatherNd)
    .INPUT(x, Tensor)
    .INPUT(indices, Tensor)
    .OUTPUT(y, Tensor)
    """
```

**特点**：
- 支持多维索引
- indices 的最后一个维度是索引向量

**适用性**：★★★☆☆
- 适合需要多维索引的场景
- 但 stablehlo.gather 的语义更复杂

### 4. GatherElements

```python
def GatherElements(x: Tensor, indices: Tensor, *, dim: int = 0):
    """
    REG_OP(GatherElements)
    .INPUT(x, Tensor)
    .INPUT(indices, Tensor)
    .ATTR(dim, Int, 0)
    .OUTPUT(y, Tensor)
    """
```

**特点**：
- 按元素收集
- 类似 PyTorch 的 torch.gather

**适用性**：★★☆☆☆
- 不太适合 stablehlo.gather 的语义

## 推荐方案

### 方案 1：使用 GatherV2（推荐）

**理由**：
1. 最接近 stablehlo.gather 的语义
2. 支持 batch_dims，可以处理 batch 维度
3. 灵活性高，axis 可以是动态输入

**转换策略**：

对于用户的例子：
```mlir
stablehlo.gather operand<151936x896>, indices<1x8x1>
  -> GatherV2(x=operand, indices=indices_squeezed, axis=0, batch_dims=0)
```

**步骤**：
1. Squeeze indices: `[1, 8, 1]` -> `[1, 8]`
2. 使用 GatherV2，axis=0，batch_dims=0
3. 结果: `[1, 8, 896]`

### 方案 2：组合多个操作

如果 GatherV2 无法完全匹配，可以组合多个操作：

```mlir
// 1. Reshape indices
%indices_reshaped = reshape(%indices) : tensor<1x8x1xi32> -> tensor<1x8xi32>

// 2. Expand dims for operand
%operand_expanded = expand_dims(%operand, axis=0) : tensor<151936x896xf32> -> tensor<1x151936x896xf32>

// 3. GatherV2 with batch_dims
%gathered = GatherV2(%operand_expanded, %indices_reshaped, axis=1, batch_dims=1)

// 4. Squeeze if needed
%result = squeeze(%gathered) : tensor<1x8x896xf32>
```

## 当前实现的问题

当前的 ConvertGatherOp 实现过于简单：

```cpp
class ConvertGatherOp : public OpConversionPattern<stablehlo::GatherOp> {
  LogicalResult matchAndRewrite(
      stablehlo::GatherOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    rewriter.replaceOpWithNewOp<GatherOp>(op, op.getType(), 
                                          op.getOperand(),
                                          op.getStartIndices());
    return success();
  }
};
```

**问题**：
1. 没有处理 dimension_numbers
2. 没有处理 slice_sizes
3. 没有选择合适的 Ascend Gather 操作
4. AIR GatherOp 定义过于简单，缺少必要的参数

## 建议的改进

### 1. 扩展 AIR GatherOp 定义

```td
def Air_GatherV2Op : Air_Op<"GatherV2", [Pure]> {
  let arguments = (ins
    Air_Tensor:$x,
    Air_Tensor:$indices,
    Air_Tensor:$axis,
    Optional<Air_Tensor>:$batch_dims
  );
  let results = (outs Air_Tensor);
}
```

### 2. 实现 ConvertGatherOp

```cpp
class ConvertGatherOp : public OpConversionPattern<stablehlo::GatherOp> {
  LogicalResult matchAndRewrite(
      stablehlo::GatherOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    
    auto dimensionNumbers = op.getDimensionNumbers();
    auto sliceSizes = op.getSliceSizes();
    
    // Analyze parameters
    auto offsetDims = dimensionNumbers.getOffsetDims();
    auto collapsedSliceDims = dimensionNumbers.getCollapsedSliceDims();
    auto startIndexMap = dimensionNumbers.getStartIndexMap();
    auto indexVectorDim = dimensionNumbers.getIndexVectorDim();
    
    // Simple case: single-axis gather
    if (startIndexMap.size() == 1 && 
        collapsedSliceDims.size() == 1 &&
        collapsedSliceDims[0] == startIndexMap[0]) {
      
      // Use GatherV2
      int64_t axis = startIndexMap[0];
      
      // Create the axis constant
      auto axisType = RankedTensorType::get({}, rewriter.getI64Type());
      auto axisAttr = DenseIntElementsAttr::get(axisType, {axis});
      auto axisConst = rewriter.create<ConstantOp>(op.getLoc(), axisType, axisAttr);
      
      // Process indices (may need squeeze)
      Value indices = op.getStartIndices();
      if (indexVectorDim == static_cast<int64_t>(op.getStartIndices().getType().getRank()) - 1) {
        // Squeeze the last dimension
        auto indicesType = dyn_cast<RankedTensorType>(indices.getType());
        auto indicesShape = indicesType.getShape();
        SmallVector<int64_t> newShape(indicesShape.begin(), indicesShape.end() - 1);
        
        auto newIndicesType = RankedTensorType::get(newShape, indicesType.getElementType());
        auto shapeType = RankedTensorType::get({static_cast<int64_t>(newShape.size())}, rewriter.getI64Type());
        auto shapeAttr = DenseIntElementsAttr::get(shapeType, newShape);
        auto shapeConst = rewriter.create<ConstantOp>(op.getLoc(), shapeType, shapeAttr);
        
        indices = rewriter.create<ReshapeOp>(op.getLoc(), newIndicesType, indices, shapeConst,
                                             rewriter.getI64IntegerAttr(0),
                                             rewriter.getI64IntegerAttr(-1)).getResult();
      }
      
      // Create GatherV2
      rewriter.replaceOpWithNewOp<GatherV2Op>(op, op.getType(), 
                                              op.getOperand(), indices, 
                                              axisConst.getResult(), nullptr);
      return success();
    }
    
    // Complex cases require a more complex conversion
    return rewriter.notifyMatchFailure(op, "Complex gather not yet supported");
  }
};
```

## 总结

1. **推荐使用 GatherV2**：最接近 stablehlo.gather 的语义
2. **需要扩展 AIR GatherOp 定义**：添加必要的参数
3. **需要实现更复杂的转换逻辑**：处理 dimension_numbers 和 slice_sizes
4. **可能需要组合多个操作**：对于复杂的 gather 场景

## 下一步

1. 在 mair_ops.td 中添加 GatherV2Op 定义
2. 在 export_graphdef.cc 中添加 GatherV2 的导出逻辑
3. 实现 ConvertGatherOp 的完整转换逻辑
4. 添加测试用例验证转换正确性
