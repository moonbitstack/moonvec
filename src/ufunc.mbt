// Copyright 2026 Leo Cheng
// SPDX-License-Identifier: Apache-2.0

///|
/// The backend a caller gets when it does not name one.
fn pick_kernel(kernel : &Kernel?) -> &Kernel {
  match kernel {
    Some(kernel) => kernel
    None => cpu
  }
}

///|
/// Applies a binary operation over two arrays, broadcasting them to a common shape.
///
/// `dtype` is the type the operands are read as — the promotion of their two dtypes — and
/// `out_dtype` what the result is stored as, which differs for comparisons.
fn binary_map(
  a : NDArray,
  b : NDArray,
  dtype : DType,
  out_dtype : DType,
  op : BinaryOp,
  kernel : &Kernel,
) -> NDArray raise VecError {
  let shape = broadcast_shape(a.shape, b.shape)
  let out = alloc(shape, out_dtype)
  let plan = plan_elementwise(shape, [a, b])
  plan.run(fn(bases, len) raise VecError {
    kernel.binary(
      op,
      dtype,
      a.buffer,
      bases[1],
      plan.run_strides[1],
      b.buffer,
      bases[2],
      plan.run_strides[2],
      out.buffer,
      bases[0],
      plan.run_strides[0],
      len,
    )
  })
  out
}

///|
/// Applies a unary operation, writing a new array of `out_dtype`.
fn unary_map(
  a : NDArray,
  dtype : DType,
  out_dtype : DType,
  op : UnaryOp,
  kernel : &Kernel,
) -> NDArray raise VecError {
  let out = alloc(a.shape, out_dtype)
  let plan = plan_elementwise(a.shape, [a])
  plan.run(fn(bases, len) raise VecError {
    kernel.unary(
      op,
      dtype,
      a.buffer,
      bases[1],
      plan.run_strides[1],
      out.buffer,
      bases[0],
      plan.run_strides[0],
      len,
    )
  })
  out
}

///|
/// The dtype an elementwise arithmetic operation produces, and the one its operands are read as.
fn arithmetic_dtype(a : NDArray, b : NDArray, op : BinaryOp) -> DType {
  let dtype = match op {
    Divide => division_dtype(a.dtype(), b.dtype())
    _ => promote(a.dtype(), b.dtype())
  }
  match op {
    // The division family and power count booleans rather than treating them as true and false, so
    // their result is an integer.
    Divide | FloorDivide | Remainder | Power =>
      if dtype.is_bool() {
        I8
      } else {
        dtype
      }
    _ => dtype
  }
}

///|
/// The dtype a unary mathematical function produces: an integer array is read as doubles, a float
/// array keeps its width, and a complex array keeps its complex type.
fn math_dtype(dtype : DType) -> DType {
  if dtype.is_real() {
    dtype
  } else {
    F64
  }
}

///|
/// Runs a binary operation with broadcasting, promotion and the default kernel.
fn binary_with(
  a : NDArray,
  b : NDArray,
  op : BinaryOp,
  kernel : &Kernel,
) -> NDArray raise VecError {
  let dtype = arithmetic_dtype(a, b, op)
  binary_map(a, b, dtype, dtype, op, kernel)
}

///|
/// Runs a comparison, which reads its operands at the promoted dtype and answers in booleans.
fn compare_with(
  a : NDArray,
  b : NDArray,
  op : BinaryOp,
  kernel : &Kernel,
) -> NDArray raise VecError {
  let dtype = promote(a.dtype(), b.dtype())
  binary_map(a, b, dtype, Bool, op, kernel)
}

///|
/// Runs a unary mathematical function at `math_dtype`.
fn math_unary(
  a : NDArray,
  op : UnaryOp,
  kernel : &Kernel,
) -> NDArray raise VecError {
  let dtype = math_dtype(a.dtype())
  unary_map(a, dtype, dtype, op, kernel)
}

///|
/// The shape an elementwise operation of these two arrays has, without doing the work.
pub fn broadcast(
  shape_a : Array[Int32],
  shape_b : Array[Int32],
) -> Array[Int32] raise VecError {
  broadcast_shape(shape_a, shape_b)
}

///|
/// The same array seen with a larger shape, without copying, by giving stretched axes stride 0.
///
/// The new shape must broadcast up to `shape`, so a size-one axis may grow and a leading axis may
/// be added; anything else cannot be expressed as a view and is refused rather than copied
/// silently.
pub fn broadcast_to(
  array : NDArray,
  shape : Array[Int32],
) -> NDArray raise VecError {
  let reached = broadcast_shape(array.shape, shape)
  if reached != shape {
    raise ShapeMismatch(want=array.shape, got=shape)
  }
  let strides = broadcast_strides(array.shape, array.strides, shape)
  NDArray::new(buffer=array.buffer, shape~, strides~, offset=array.offset)
}
