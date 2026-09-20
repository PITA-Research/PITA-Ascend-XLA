/* Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
 
#ifndef MAIR_OPS
#define MAIR_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/BuiltinAttributes.td"
include "mair_ops_base.td"

//===--------------------------------------------------------------------------===//
// ConstantOp
//===--------------------------------------------------------------------------===//

def Air_ConstantOp : Air_Op<"Const", [
    ConstantLike,
    Pure
    ]>{
        let summary = "Constant operation";
        let description = [{
            Represents a constant value in the MAIR dialect.
        }];

        let arguments = (ins
            AnyAttr:$value
        );

        let results = (outs
           AnyType:$output
     );
}

//===--------------------------------------------------------------------------===//
// AddOp
//===--------------------------------------------------------------------------===//

def Air_AddOp : Air_Op<"Add",[
    Pure, 
    SameOperandsAndResultType
    ]>{
        let summary = "element-wise addition operation";
        let description = [{
            The "add" operation performs element-wise addition between two tensors.
        }];

        let arguments = (ins
            Air_Tensor:$lhs,
            Air_Tensor:$rhs
        );

        let results = (outs Air_Tensor);
    }


//===--------------------------------------------------------------------------===//
// BroadcastOp
//===--------------------------------------------------------------------------===//

def Air_BroadcastOp : Air_Op<"BroadcastTo", [Pure]>{
    let summary = "broadcast to a tensor type.";
    let description = "broadcast to a tensor type.";

    let arguments = (ins
        Air_Tensor:$input,
        TensorOf<[I32, I64]>:$shape
    );

    let results = (outs Air_Tensor);
}


//===--------------------------------------------------------------------------===//
// MultiplyOp
//===--------------------------------------------------------------------------===//

def Air_MulOp : Air_Op<"Mul", [Pure]>{
    let summary = "Returns x * y element-wise.";
    let description = [{
        Returns x * y element-wise. 
    }];

    let arguments = (ins
        Air_Tensor:$x,
        Air_Tensor:$y
    );

    let results = (outs Air_Tensor);
}


//===--------------------------------------------------------------------------===//
// TanhOp
//===--------------------------------------------------------------------------===//

def Air_TanhOp : Air_Op<"Tanh", [Pure]>{
    let summary = "Tanh operator.";
    let description = "Tanh operator.";

    let arguments = (ins
        Air_Tensor:$x
    );

    let results = (outs Air_Tensor);
}

//===----------------------------------------------------------------------===//
// ascend graph op definitions
//===----------------------------------------------------------------------===//
def Air_GraphOp : Air_Op<"graph"> {
  let summary = [{The `ascend.graph` operation contains a region with a
    single block that lists the operations in a Ascend graph.}];


  let description = [{
    The operations are topologically sorted in-order (no cycles are allowed in
    the values).
  }];

  let results = (outs
    Variadic<AnyType>:$results
  );

  let regions = (region SizedRegion<1>:$body);

  let extraClassDeclaration = [{
    Block &GetBody() { return getOperation()->getRegion(0).front(); }
  }];

  let hasCanonicalizer = 1;
}


//===--------------------------------------------------------------------------===//
// HcomAllReduceOp
//===--------------------------------------------------------------------------===//

def Air_HcomAllReduceOp : Air_Op<"HcomAllReduce", [Pure]>{
    let summary = "HcomAllReduce operator.";
    let description = "HcomAllReduce operator.";

    let arguments = (ins
        Air_Tensor:$x, // Currently only supports allreduce on a tensor
        I32:$allreduce_type // 0: SUM, 1: MAX, 2: MIN, 3: PROD
    );

    let results = (outs Air_Tensor);
}

//===--------------------------------------------------------------------------===//
// HcomAllGatherOp
//===--------------------------------------------------------------------------===//

def Air_HcomAllGatherOp : Air_Op<"HcomAllGather", [Pure]>{
    let summary = "HcomAllGather operator.";
    let description = "HcomAllGather operator.";

    let arguments = (ins
        Air_Tensor:$x,
        I64:$rank_size
    );

    let results = (outs Air_Tensor);
}

//===--------------------------------------------------------------------------===//
// ConcatV2Op
//===--------------------------------------------------------------------------===//
def Air_ConcatV2Op : Air_Op<"ConcatV2", [Pure]> {
    let summary = "Concatenates tensors along one dimension.";

    let arguments = (ins
    Arg<Variadic<TensorOf<[Air_BasicType]>>, [{List of `N` Tensors to concatenate. Their ranks and types must match,
    and their sizes must match in all dimensions except `concat_dim`.}]>:$x,
    Arg<TensorOf<[Air_BasicType]>, [{0-D.  The dimension along which to concatenate.}]>:$concat_dim,
    DefaultValuedAttr<I64Attr, "1">:$N
    );

    let results = (outs
    Res<TensorOf<[Air_BasicType]>, [{A `Tensor` with the concatenation of values stacked along the
    `concat_dim` dimension.  This tensor's shape matches that of `values` except
    in `concat_dim` where it has the sum of the sizes.}]>:$y
    );
}

//===--------------------------------------------------------------------------===//
// DivOp
//===--------------------------------------------------------------------------===//
def Air_DivOp : Air_Op<"Div", [Pure, SameOperandsAndResultType]> {
  let summary = "Element-wise division operation.";
  let description = [{
    Performs element-wise division of two input tensors.
  }];
  let arguments = (ins
    Air_Tensor:$lhs,
    Air_Tensor:$rhs
  );
  let results = (outs Air_Tensor);
}

//===--------------------------------------------------------------------------===//
// MatMulOp
//===--------------------------------------------------------------------------===//
def Air_MatMulOp : Air_Op<"MatMul", [Pure]> {
  let summary = "Standard matrix multiplication operation";
  let description = [{
    Performs standard matrix multiplication on two input tensors, summing over contracting dimensions.
    For example: tensor<AxB> * tensor<BxC> -> tensor<AxC>
  }];

  let arguments = (ins
    Air_Tensor:$x1,
    Air_Tensor:$x2,
    Optional<Air_Tensor>:$bias,
    DefaultValuedAttr<BoolAttr, "false">:$transpose_x1,
    DefaultValuedAttr<BoolAttr, "false">:$transpose_x2
  );

  let results = (outs
    Air_Tensor:$output
  );
}

//===--------------------------------------------------------------------------===//
// BatchMatMulOp
//===--------------------------------------------------------------------------===//
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

//===--------------------------------------------------------------------------===//
// ExpOp
//===--------------------------------------------------------------------------===//
def Air_ExpOp : Air_Op<"Exp", [Pure, SameOperandsAndResultType]> {
  let summary = "Element-wise exponential operation.";
  let description = [{
    Computes the exponential value of each element in the input tensor.
  }];
  let arguments = (ins Air_Tensor:$operand);
  let results = (outs Air_Tensor:$result);
}

//===--------------------------------------------------------------------------===//
// MaxOp
//===--------------------------------------------------------------------------===//
def Air_MaxOp : Air_Op<"Maximum", [Pure, SameOperandsAndResultType]> {
  let summary = "Element-wise maximum operation.";
  let description = [{
    Computes the element-wise maximum of two input tensors.
  }];
  let arguments = (ins
    Air_Tensor:$lhs,
    Air_Tensor:$rhs
  );
  let results = (outs Air_Tensor);
}

//===--------------------------------------------------------------------------===//
// NegOp
//===--------------------------------------------------------------------------===//
def Air_NegOp : Air_Op<"Neg", [Pure, SameOperandsAndResultType]> {
  let summary = "Element-wise negation operation.";
  let description = [{
    Computes the element-wise negation of the input tensor.
  }];
  let arguments = (ins Air_Tensor:$operand);
  let results = (outs Air_Tensor:$result);
}

//===--------------------------------------------------------------------------===//
// ReduceSumOp
//===--------------------------------------------------------------------------===//
def Air_ReduceSumOp : Air_Op<"ReduceSum", [Pure]> {
  let summary = "Reduction sum operation.";
  let description = [{
    Applies sum reduction to elements along specified dimensions of the input tensor.
  }];
  let arguments = (ins
    Air_Tensor:$inputs,
    Air_Tensor:$axes,
    DefaultValuedAttr<BoolAttr, "false">:$keep_dims
  );
  let results = (outs Variadic<AnyType>:$outputs);
}

//===--------------------------------------------------------------------------===//
// ReduceMaxOp
//===--------------------------------------------------------------------------===//
def Air_ReduceMaxOp : Air_Op<"ReduceMax", [Pure]> {
  let summary = "Reduction max operation.";
  let description = [{
    Applies max reduction to elements along specified dimensions of the input tensor.
  }];
  let arguments = (ins
    Air_Tensor:$inputs,
    Air_Tensor:$axes,
    DefaultValuedAttr<BoolAttr, "false">:$keep_dims
  );
  let results = (outs Variadic<AnyType>:$outputs);
}

//===--------------------------------------------------------------------------===//
// ReduceMinOp
//===--------------------------------------------------------------------------===//
def Air_ReduceMinOp : Air_Op<"ReduceMin", [Pure]> {
  let summary = "Reduction min operation.";
  let description = [{
    Applies min reduction to elements along specified dimensions of the input tensor.
  }];
  let arguments = (ins
    Air_Tensor:$inputs,
    Air_Tensor:$axes,
    DefaultValuedAttr<BoolAttr, "false">:$keep_dims
  );
  let results = (outs Variadic<AnyType>:$outputs);
}

//===--------------------------------------------------------------------------===//
// ReshapeOp
//===--------------------------------------------------------------------------===//
def Air_ReshapeOp : Air_Op<"Reshape", [Pure]> {
  let summary = "Reshape operation.";
  let description = [{
    Reshapes a tensor to the specified shape while preserving total element count.
  }];
  let arguments = (ins
    Air_Tensor:$x,
    Air_Tensor:$shape,
    DefaultValuedAttr<I64Attr, "0">:$axis,
    DefaultValuedAttr<I64Attr, "-1">:$num_axes
  );

  let results = (outs Air_Tensor:$result);
}

//===--------------------------------------------------------------------------===//
// SubOp
//===--------------------------------------------------------------------------===//
def Air_SubOp : Air_Op<"Sub", [Pure, SameOperandsAndResultType]> {
  let summary = "Element-wise subtraction operation.";
  let description = [{
    Performs element-wise subtraction of two input tensors.
  }];
  let arguments = (ins
    Air_Tensor:$lhs,
    Air_Tensor:$rhs
  );
  let results = (outs Air_Tensor);
}

//===--------------------------------------------------------------------------===//
// CastOp
//===--------------------------------------------------------------------------===//
def Air_CastOp : Air_Op<"Cast", [Pure]> {
  let summary = "Type cast operation with explicit destination type.";
  let description = [{
    The `your_dialect.cast` operation converts values of one type to another.
    The destination type is specified by the `dst_type` attribute.
  }];

  let arguments = (ins
    AnyType:$x,
    TypeAttr:$dst_type
  );

  let results = (outs
    AnyType:$y
  );
}

//===--------------------------------------------------------------------------===//
// SqueezeV3Op
//===--------------------------------------------------------------------------===//
def Air_SqueezeV3Op : Air_Op<"SqueezeV3", [Pure]> {
  let summary = "Removes dimensions of size 1 from the shape of a tensor.";
  let description = [{
    The SqueezeV3 operation removes specified dimensions of size 1 from the shape of a tensor.
    If the optional 'axes' input is provided, only the dimensions listed in 'axes' will be removed,
    and those dimensions must have a size of 1. If 'axes' is not provided (is None),
    all dimensions of size 1 will be removed automatically.
  }];

  let arguments = (ins
    Air_Tensor:$x,
    Optional<Air_Tensor>:$axes
  );

  let results = (outs
    Air_Tensor:$output
  );
}

//===--------------------------------------------------------------------------===//
// UnsqueezeV3Op
//===--------------------------------------------------------------------------===//
def Air_UnsqueezeV3Op : Air_Op<"UnsqueezeV3", [Pure]> {
  let summary = "Inserts dimensions of size 1 into the shape of a tensor.";
  let description = [{
    The UnsqueezeV3 operation inserts new dimensions of size 1 at specified positions in the shape of a tensor.
    The positions are given by the 'axes' input tensor, which is a 1-D tensor of type int64.
  }];

  let arguments = (ins
    Air_Tensor:$x,
    Air_Tensor:$axes
  );

  let results = (outs
    Air_Tensor:$output
  );
}

//===--------------------------------------------------------------------------===//
// AndOp
//===--------------------------------------------------------------------------===//
def Air_AndOp : Air_Op<"LogicalAnd", [Pure, SameOperandsAndResultType]> {
  let summary = "Element-wise logical AND operation.";
  let description = [{
    Computes element-wise logical AND of two input tensors.
  }];
  let arguments = (ins
    Air_Tensor:$x1,
    Air_Tensor:$x2
  );
  let results = (outs Air_Tensor);
}

//===--------------------------------------------------------------------------===//
// EqualOp
//===--------------------------------------------------------------------------===//
def Air_EqualOp : Air_Op<"Equal", [Pure]> {
  let summary = "Element-wise equal comparison operation.";
  let description = [{
    Returns the truth value of (x1 == x2) element-wise.
  }];
  let arguments = (ins
    Air_Tensor:$x1,
    Air_Tensor:$x2
  );
  let results = (outs Air_Tensor);
}

//===--------------------------------------------------------------------------===//
// NotEqualOp
//===--------------------------------------------------------------------------===//
def Air_NotEqualOp : Air_Op<"NotEqual", [Pure]> {
  let summary = "Element-wise not equal comparison operation.";
  let description = [{
    Returns the truth value of (x1 != x2) element-wise.
  }];
  let arguments = (ins
    Air_Tensor:$x1,
    Air_Tensor:$x2
  );
  let results = (outs Air_Tensor);
}

//===--------------------------------------------------------------------------===//
// GreaterOp
//===--------------------------------------------------------------------------===//
def Air_GreaterOp : Air_Op<"Greater", [Pure]> {
  let summary = "Element-wise greater than comparison operation.";
  let description = [{
    Returns the truth value of (x1 > x2) element-wise.
  }];
  let arguments = (ins
    Air_Tensor:$x1,
    Air_Tensor:$x2
  );
  let results = (outs Air_Tensor);
}

//===--------------------------------------------------------------------------===//
// GreaterEqualOp
//===--------------------------------------------------------------------------===//
def Air_GreaterEqualOp : Air_Op<"GreaterEqual", [Pure]> {
  let summary = "Element-wise greater than or equal comparison operation.";
  let description = [{
    Returns the truth value of (x1 >= x2) element-wise.
  }];
  let arguments = (ins
    Air_Tensor:$x1,
    Air_Tensor:$x2
  );
  let results = (outs Air_Tensor);
}

//===--------------------------------------------------------------------------===//
// LessOp
//===--------------------------------------------------------------------------===//
def Air_LessOp : Air_Op<"Less", [Pure]> {
  let summary = "Element-wise less than comparison operation.";
  let description = [{
    Returns the truth value of (x1 < x2) element-wise.
  }];
  let arguments = (ins
    Air_Tensor:$x1,
    Air_Tensor:$x2
  );
  let results = (outs Air_Tensor);
}

//===--------------------------------------------------------------------------===//
// LessEqualOp
//===--------------------------------------------------------------------------===//
def Air_LessEqualOp : Air_Op<"LessEqual", [Pure]> {
  let summary = "Element-wise less than or equal comparison operation.";
  let description = [{
    Returns the truth value of (x1 <= x2) element-wise.
  }];
  let arguments = (ins
    Air_Tensor:$x1,
    Air_Tensor:$x2
  );
  let results = (outs Air_Tensor);
}

//===--------------------------------------------------------------------------===//
// CosOp
//===--------------------------------------------------------------------------===//
def Air_CosOp : Air_Op<"Cos", [Pure, SameOperandsAndResultType]> {
  let summary = "Element-wise cosine operation.";
  let description = [{
    Computes cosine of input elements element-wise.
  }];
  let arguments = (ins Air_Tensor:$x);
  let results = (outs Air_Tensor);
}

//===--------------------------------------------------------------------------===//
// GatherOp
//===--------------------------------------------------------------------------===//
def Air_GatherOp : Air_Op<"Gather", [Pure]> {
  let summary = "Gather operation.";
  let description = [{
    Gather slices from params according to indices.
  }];
  let arguments = (ins
    Air_Tensor:$x,
    Air_Tensor:$indices
  );
  let results = (outs Air_Tensor);
}

//===--------------------------------------------------------------------------===//
// GatherV2Op
//===--------------------------------------------------------------------------===//
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

//===--------------------------------------------------------------------------===//
// PowOp
//===--------------------------------------------------------------------------===//
def Air_PowOp : Air_Op<"Pow", [Pure, SameOperandsAndResultType]> {
  let summary = "Element-wise power operation.";
  let description = [{
    Computes x1 raised to the power of x2 element-wise.
  }];
  let arguments = (ins
    Air_Tensor:$x1,
    Air_Tensor:$x2
  );
  let results = (outs Air_Tensor);
}

//===--------------------------------------------------------------------------===//
// RsqrtOp
//===--------------------------------------------------------------------------===//
def Air_RsqrtOp : Air_Op<"Rsqrt", [Pure, SameOperandsAndResultType]> {
  let summary = "Element-wise reciprocal square root operation.";
  let description = [{
    Computes reciprocal square root of input elements element-wise.
  }];
  let arguments = (ins Air_Tensor:$x);
  let results = (outs Air_Tensor);
}

//===--------------------------------------------------------------------------===//
// SelectOp
//===--------------------------------------------------------------------------===//
def Air_SelectOp : Air_Op<"Select", [Pure]> {
  let summary = "Select operation.";
  let description = [{
    Selects elements from x1 or x2 based on condition.
  }];
  let arguments = (ins
    Air_Tensor:$condition,
    Air_Tensor:$x1,
    Air_Tensor:$x2
  );
  let results = (outs Air_Tensor);
}

//===--------------------------------------------------------------------------===//
// SinOp
//===--------------------------------------------------------------------------===//
def Air_SinOp : Air_Op<"Sin", [Pure, SameOperandsAndResultType]> {
  let summary = "Element-wise sine operation.";
  let description = [{
    Computes sine of input elements element-wise.
  }];
  let arguments = (ins Air_Tensor:$x);
  let results = (outs Air_Tensor);
}

//===--------------------------------------------------------------------------===//
// SliceOp
//===--------------------------------------------------------------------------===//
def Air_SliceOp : Air_Op<"Slice", [Pure]> {
  let summary = "Slice operation.";
  let description = [{
    Extracts a slice from a tensor.
  }];
  let arguments = (ins
    Air_Tensor:$x,
    Air_Tensor:$offsets,
    Air_Tensor:$size
  );
  let results = (outs Air_Tensor);
}

//===--------------------------------------------------------------------------===//
// TransposeOp
//===--------------------------------------------------------------------------===//
def Air_TransposeOp : Air_Op<"Transpose", [Pure]> {
  let summary = "Transpose operation.";
  let description = [{
    Permutes the dimensions of a tensor according to the given permutation.
  }];
  let arguments = (ins
    Air_Tensor:$x,
    Air_Tensor:$perm
  );
  let results = (outs Air_Tensor);
}

//===--------------------------------------------------------------------------===//
// ReduceAllOp
//===--------------------------------------------------------------------------===//
def Air_ReduceAllOp : Air_Op<"ReduceAll", [Pure]> {
  let summary = "ReduceAll operation.";
  let description = [{
    Computes the logical AND of elements along specified dimensions.
  }];
  let arguments = (ins
    Air_Tensor:$x,
    Air_Tensor:$axes,
    DefaultValuedAttr<BoolAttr, "false">:$keep_dims
  );
  let results = (outs Variadic<AnyType>:$outputs);
}

//===--------------------------------------------------------------------------===//
// ReduceAnyOp
//===--------------------------------------------------------------------------===//
def Air_ReduceAnyOp : Air_Op<"ReduceAny", [Pure]> {
  let summary = "ReduceAny operation.";
  let description = [{
    Computes the logical OR of elements along specified dimensions.
  }];
  let arguments = (ins
    Air_Tensor:$x,
    Air_Tensor:$axes,
    DefaultValuedAttr<BoolAttr, "false">:$keep_dims
  );
  let results = (outs Variadic<AnyType>:$outputs);
}

#endif // MAIR_OPS