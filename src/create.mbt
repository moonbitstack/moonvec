// Copyright 2026 Leo Cheng
// SPDX-License-Identifier: Apache-2.0

///|
/// An array of `shape` filled with zeros.
pub fn zeros(
  shape : Array[Int32],
  dtype~ : DType = F64,
) -> NDArray raise VecError {
  filled(shape, Boolean(false), dtype)
}

///|
/// An array of `shape` filled with ones.
pub fn ones(
  shape : Array[Int32],
  dtype~ : DType = F64,
) -> NDArray raise VecError {
  filled(shape, Number::cast(Signed(1L), dtype), dtype)
}

///|
/// An array of `shape` filled with `value`, which is converted to `dtype`.
pub fn full(
  shape : Array[Int32],
  value : Number,
  dtype~ : DType = F64,
) -> NDArray raise VecError {
  filled(shape, value.cast(dtype), dtype)
}

///|
/// An array with the same shape as `prototype`, filled with `value`; the dtype follows `prototype`
/// unless `dtype` says otherwise.
pub fn full_like(
  prototype : NDArray,
  value : Number,
  dtype? : DType,
) -> NDArray raise VecError {
  let dtype = match dtype {
    Some(dtype) => dtype
    None => prototype.dtype()
  }
  filled(prototype.shape(), value.cast(dtype), dtype)
}

///|
/// The common body of the `*_like` constructors.
pub fn zeros_like(
  prototype : NDArray,
  dtype? : DType,
) -> NDArray raise VecError {
  full_like(prototype, Boolean(false), dtype?)
}

///|
/// An array of ones shaped like `prototype`.
pub fn ones_like(prototype : NDArray, dtype? : DType) -> NDArray raise VecError {
  full_like(prototype, Signed(1L), dtype?)
}

///|
fn filled(
  shape : Array[Int32],
  value : Number,
  dtype : DType,
) -> NDArray raise VecError {
  let out = alloc(shape, dtype)
  out.buffer.fill(value)
  out
}

///|
/// Evenly spaced values in `[start, stop)`, stepping by `step`.
///
/// The count is computed from the span and the step, so `arange(0, 1, 0.3)` stops at 0.9 rather
/// than at 1.2. A step whose sign does not lead towards `stop` gives an empty array
/// instead of looping forever.
pub fn arange(
  start~ : Double,
  stop~ : Double,
  step~ : Double = 1.0,
  dtype~ : DType = F64,
) -> NDArray raise VecError {
  if step == 0.0 {
    raise Domain(message="arange needs a step that is not zero")
  }
  let span = (stop - start) / step
  let count = if span <= 0.0 {
    0
  } else {
    let rounded = span.ceil()
    if rounded > 2147483647.0 {
      raise Domain(
        message="arange would need more elements than an array can hold",
      )
    }
    rounded.to_int()
  }
  let out = alloc([count], dtype)
  for i in 0..<count {
    out.buffer.write(i, Real(start + step * i.to_double()).cast(dtype))
  }
  out
}

///|
/// `num` values from `start` to `stop`; with `endpoint` (the default) the last value is exactly
/// `stop`, so the spacing is `(stop - start) / (num - 1)`. With one value the result is `start`,
/// not `stop`.
pub fn linspace(
  start~ : Double,
  stop~ : Double,
  num~ : Int32 = 50,
  endpoint~ : Bool = true,
  dtype~ : DType = F64,
) -> NDArray raise VecError {
  if num < 0 {
    raise Domain(message="linspace needs a count that is not negative")
  }
  let out = alloc([num], dtype)
  if num == 0 {
    return out
  }
  let divisor = if endpoint { if num == 1 { 1 } else { num - 1 } } else { num }
  let step = (stop - start) / divisor.to_double()
  for i in 0..<num {
    let value = if num == 1 {
      start
    } else if i == num - 1 && endpoint {
      stop
    } else {
      start + step * i.to_double()
    }
    out.buffer.write(i, Real(value).cast(dtype))
  }
  out
}

///|
/// A matrix with ones on the `k`-th diagonal and zeros elsewhere.
///
/// `k` counts diagonals from the main one: positive towards the upper right, negative towards the
/// lower left. `m` defaults to `n`, so the result is square.
pub fn eye(
  n~ : Int32,
  m? : Int32,
  k~ : Int32 = 0,
  dtype~ : DType = F64,
) -> NDArray raise VecError {
  let columns = match m {
    Some(m) => m
    None => n
  }
  if n < 0 || columns < 0 {
    raise Domain(message="eye needs non-negative dimensions")
  }
  let out = zeros([n, columns], dtype~)
  for row in 0..<n {
    let column = row + k
    if column >= 0 && column < columns {
      out.buffer.write(row * columns + column, Number::cast(Signed(1L), dtype))
    }
  }
  out
}

///|
/// An array of `shape` from values in row-major order.
pub fn from_numbers(
  values : Array[Number],
  shape : Array[Int32],
  dtype? : DType,
) -> NDArray raise VecError {
  let count = element_count(shape)
  if count != values.length() {
    raise ShapeMismatch(want=shape, got=[values.length()])
  }
  let dtype = match dtype {
    Some(dtype) => dtype
    None => infer_dtype(values)
  }
  let out = alloc(shape, dtype)
  for i in 0..<values.length() {
    out.buffer.write(i, values[i].cast(dtype))
  }
  out
}

///|
/// An array of `shape` from doubles in row-major order.
pub fn from_doubles(
  values : Array[Double],
  shape : Array[Int32],
  dtype~ : DType = F64,
) -> NDArray raise VecError {
  let count = element_count(shape)
  if count != values.length() {
    raise ShapeMismatch(want=shape, got=[values.length()])
  }
  let out = alloc(shape, dtype)
  for i in 0..<values.length() {
    out.buffer.write(i, Real(values[i]).cast(dtype))
  }
  out
}

///|
/// A 1-D array from a list of values.
pub fn array(
  values : Array[Double],
  dtype~ : DType = F64,
) -> NDArray raise VecError {
  from_doubles(values, [values.length()], dtype~)
}

///|
/// A 2-D array from rows that all have the same length.
pub fn matrix(
  rows : Array[Array[Double]],
  dtype~ : DType = F64,
) -> NDArray raise VecError {
  let height = rows.length()
  let width = if height == 0 { 0 } else { rows[0].length() }
  let flat : Array[Double] = []
  for row in rows {
    if row.length() != width {
      raise ShapeMismatch(want=[height, width], got=[row.length()])
    }
    for value in row {
      flat.push(value)
    }
  }
  from_doubles(flat, [height, width], dtype~)
}

///|
/// An array from raw bytes, which must be a whole number of elements of `dtype`.
pub fn from_bytes(
  bytes : Bytes,
  shape : Array[Int32],
  dtype~ : DType,
) -> NDArray raise VecError {
  let buffer = Buffer::from_bytes(bytes, dtype~)
  let count = element_count(shape)
  if buffer.len() != count {
    raise ShapeMismatch(want=shape, got=[buffer.len()])
  }
  NDArray::new(buffer~, shape~)
}

///|
/// The dtype that holds every value without loss: booleans for a boolean list, `int64` for
/// integers, `complex128` for complex, `float64` for a mixture.
fn infer_dtype(values : Array[Number]) -> DType {
  let mut result = Bool
  let mut first = true
  for value in values {
    let dtype = match value {
      Boolean(_) => Bool
      Signed(_) => I64
      Unsigned(_) => U64
      Real(_) => F64
      Complex(_) => C128
    }
    if first {
      result = dtype
      first = false
    } else {
      result = promote(result, dtype)
    }
  }
  result
}
