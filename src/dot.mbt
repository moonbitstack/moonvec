// Copyright 2026 Leo Cheng
// SPDX-License-Identifier: Apache-2.0

///|
/// Matrix multiplication, with NumPy's rules for the shapes that are not both matrices.
///
/// The last axis of `a` contracts with the second-to-last of `b`. A 1-D operand is taken as a row or
/// a column and the axis it borrowed is removed from the result, so a vector times a vector is a
/// 0-dimensional array. Leading axes broadcast against each other, which makes a stack of matrices
/// multiply as a stack.
pub fn matmul(
  a : NDArray,
  b : NDArray,
  kernel? : &Kernel,
) -> NDArray raise VecError {
  if a.ndim() == 0 || b.ndim() == 0 {
    raise ShapeMismatch(want=a.shape(), got=b.shape())
  }
  let padded_a = if a.ndim() == 1 { reshape(a, [1, a.shape()[0]]) } else { a }
  let padded_b = if b.ndim() == 1 { reshape(b, [b.shape()[0], 1]) } else { b }
  let ndim_a = padded_a.ndim()
  let ndim_b = padded_b.ndim()
  let inner = padded_a.shape()[ndim_a - 1]
  if inner != padded_b.shape()[ndim_b - 2] {
    raise ShapeMismatch(want=padded_a.shape(), got=padded_b.shape())
  }
  let batch = broadcast_shape(
    padded_a.shape()[0:ndim_a - 2].to_owned(),
    padded_b.shape()[0:ndim_b - 2].to_owned(),
  )
  let rows = padded_a.shape()[ndim_a - 2]
  let columns = padded_b.shape()[ndim_b - 1]
  let left = if padded_a.is_contiguous() { padded_a } else { padded_a.copy() }
  let right = if padded_b.is_contiguous() { padded_b } else { padded_b.copy() }
  let dtype = promote(a.dtype(), b.dtype())
  let out_shape = batch.copy()
  out_shape.push(rows)
  out_shape.push(columns)
  let out = alloc(out_shape, dtype)
  let out_strides = contiguous_strides(out_shape)
  let plan = plan_loop(
    batch,
    [
      out_strides[0:batch.length()].to_owned(),
      broadcast_strides(
        padded_a.shape()[0:ndim_a - 2].to_owned(),
        padded_a.strides()[0:ndim_a - 2].to_owned(),
        batch,
      ),
      broadcast_strides(
        padded_b.shape()[0:ndim_b - 2].to_owned(),
        padded_b.strides()[0:ndim_b - 2].to_owned(),
        batch,
      ),
    ],
    [0, left.offset(), right.offset()],
  )
  let backend = pick_kernel(kernel)
  plan.run(fn(bases, len) raise VecError {
    for step in 0..<len {
      backend.matmul(
        dtype,
        left.buffer,
        bases[1] + step * plan.run_strides[1],
        right.buffer,
        bases[2] + step * plan.run_strides[2],
        out.buffer,
        bases[0] + step * plan.run_strides[0],
        rows,
        inner,
        columns,
      )
    }
  })
  let dropped : Array[Int32] = []
  for i in 0..<batch.length() {
    dropped.push(out_shape[i])
  }
  if a.ndim() > 1 {
    dropped.push(rows)
  }
  if b.ndim() > 1 {
    dropped.push(columns)
  }
  NDArray::new(buffer=out.buffer, shape=dropped)
}

///|
/// The dot product of two arrays: `a`'s last axis against `b`'s second-to-last, with the axes that
/// are left over kept in order.
///
/// For 1-D and 2-D operands this is `matmul`. For higher ranks it differs, as in NumPy: a stack times
/// a matrix contracts every row of the stack against the same matrix.
pub fn dot(
  a : NDArray,
  b : NDArray,
  kernel? : &Kernel,
) -> NDArray raise VecError {
  if a.ndim() <= 2 && b.ndim() <= 2 {
    return matmul(a, b, kernel?)
  }
  if a.ndim() == 0 || b.ndim() == 0 {
    raise ShapeMismatch(want=a.shape(), got=b.shape())
  }
  let inner = a.shape()[a.ndim() - 1]
  let contracted = b.ndim() - 2
  if inner != b.shape()[contracted] {
    raise ShapeMismatch(want=a.shape(), got=b.shape())
  }
  let left_rows = element_count(a.shape()[0:a.ndim() - 1].to_owned())
  let left = reshape(a, [left_rows, inner])
  let moved = move_axis(b, contracted, 0)
  let right_columns = element_count(moved.shape()[1:].to_owned())
  let columns = moved.shape()[moved.ndim() - 1]
  let right = reshape(moved, [inner, right_columns])
  let product = matmul(left, right, kernel?)
  let shape = a.shape()[0:a.ndim() - 1].to_owned()
  let batch_dims = b.ndim() - 2
  for i in 0..<batch_dims {
    shape.push(b.shape()[i])
  }
  shape.push(columns)
  reshape(product, shape)
}

///|
/// The outer product: `a` as a column, `b` as a row, their product a matrix. Both are read flat.
pub fn outer(
  a : NDArray,
  b : NDArray,
  kernel? : &Kernel,
) -> NDArray raise VecError {
  let column = reshape(ravel(a), [a.size(), 1])
  let row = reshape(ravel(b), [1, b.size()])
  multiply(column, row, kernel?)
}

///|
/// The inner product: a sum over the last axis of both, so the result has the shape of the operands
/// without their last axis, added together.
pub fn inner(
  a : NDArray,
  b : NDArray,
  kernel? : &Kernel,
) -> NDArray raise VecError {
  if a.ndim() == 0 || b.ndim() == 0 {
    raise ShapeMismatch(want=a.shape(), got=b.shape())
  }
  let last = a.shape()[a.ndim() - 1]
  if last != b.shape()[b.ndim() - 1] {
    raise ShapeMismatch(want=a.shape(), got=b.shape())
  }
  let right = transpose(b)
  dot(a, right, kernel?)
}

///|
/// The sum of the diagonal, or of a diagonal chosen by `offset`; a 2-D only operation.
pub fn trace(a : NDArray, offset~ : Int32 = 0) -> NDArray raise VecError {
  if a.ndim() != 2 {
    raise ShapeMismatch(want=[a.ndim()], got=[2])
  }
  let shape = a.shape()
  let rows = shape[0]
  let columns = shape[1]
  let mut start_row = if offset < 0 { -offset } else { 0 }
  let mut start_column = if offset > 0 { offset } else { 0 }
  let mut length = if offset >= 0 {
    let from_row = rows
    let from_column = columns - offset
    if from_row < from_column {
      from_row
    } else {
      from_column
    }
  } else {
    let from_row = rows + offset
    let from_column = columns
    if from_row < from_column {
      from_row
    } else {
      from_column
    }
  }
  if length < 0 {
    length = 0
    start_row = 0
    start_column = 0
  }
  let dtype = if a.dtype().is_bool() { I64 } else { a.dtype() }
  let mut total = reduce_identity(Sum, dtype)
  for i in 0..<length {
    total = apply_binary(
      Add,
      dtype,
      total,
      a.buffer.read(
        a.offset() +
        (start_row + i) * a.strides()[0] +
        (start_column + i) * a.strides()[1],
      ),
    )
  }
  let out = alloc([], dtype)
  out.buffer.write(0, total)
  out
}
