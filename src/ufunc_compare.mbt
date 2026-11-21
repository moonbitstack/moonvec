// Copyright 2026 Leo Cheng
// SPDX-License-Identifier: Apache-2.0

///|
/// Elementwise `a == b`, answering with a boolean array.
pub fn equal(
  a : NDArray,
  b : NDArray,
  kernel? : &Kernel,
) -> NDArray raise VecError {
  compare_with(a, b, Equal, pick_kernel(kernel))
}

///|
pub fn not_equal(
  a : NDArray,
  b : NDArray,
  kernel? : &Kernel,
) -> NDArray raise VecError {
  compare_with(a, b, NotEqual, pick_kernel(kernel))
}

///|
/// Elementwise `a < b`; complex arrays are not ordered and are refused.
pub fn less(
  a : NDArray,
  b : NDArray,
  kernel? : &Kernel,
) -> NDArray raise VecError {
  compare_with(a, b, Less, pick_kernel(kernel))
}

///|
pub fn less_equal(
  a : NDArray,
  b : NDArray,
  kernel? : &Kernel,
) -> NDArray raise VecError {
  compare_with(a, b, LessEqual, pick_kernel(kernel))
}

///|
pub fn greater(
  a : NDArray,
  b : NDArray,
  kernel? : &Kernel,
) -> NDArray raise VecError {
  compare_with(a, b, Greater, pick_kernel(kernel))
}

///|
pub fn greater_equal(
  a : NDArray,
  b : NDArray,
  kernel? : &Kernel,
) -> NDArray raise VecError {
  compare_with(a, b, GreaterEqual, pick_kernel(kernel))
}

///|
/// Elementwise equality that also holds for two NaNs in the same place, which `equal` does not.
pub fn equal_nan(a : NDArray, b : NDArray) -> NDArray raise VecError {
  let both_nan = logical_and(isnan(a), isnan(b))
  let plain = equal(a, b)
  select(both_nan, full_like(plain, Boolean(true)), plain)
}

///|
pub fn logical_and(
  a : NDArray,
  b : NDArray,
  kernel? : &Kernel,
) -> NDArray raise VecError {
  let dtype = promote(a.dtype(), b.dtype())
  binary_map(a, b, dtype, Bool, LogicalAnd, pick_kernel(kernel))
}

///|
pub fn logical_or(
  a : NDArray,
  b : NDArray,
  kernel? : &Kernel,
) -> NDArray raise VecError {
  let dtype = promote(a.dtype(), b.dtype())
  binary_map(a, b, dtype, Bool, LogicalOr, pick_kernel(kernel))
}

///|
/// True where exactly one side is true.
pub fn logical_xor(
  a : NDArray,
  b : NDArray,
  kernel? : &Kernel,
) -> NDArray raise VecError {
  let dtype = promote(a.dtype(), b.dtype())
  binary_map(a, b, dtype, Bool, LogicalXor, pick_kernel(kernel))
}

///|
/// Elementwise `not`, answering with a boolean array; zero is false, everything else true.
pub fn logical_not(a : NDArray, kernel? : &Kernel) -> NDArray raise VecError {
  unary_map(a, a.dtype(), Bool, LogicalNot, pick_kernel(kernel))
}

///|
/// Elementwise `~`, flipping every bit of an integer array.
pub fn invert(a : NDArray, kernel? : &Kernel) -> NDArray raise VecError {
  let one = full_like(a, Signed(1L))
  bitwise_xor(a, one, kernel?)
}

///|
/// True where the value is not a number; an integer array is never.
pub fn isnan(a : NDArray) -> NDArray raise VecError {
  if !a.dtype().is_float() {
    return full_like(a, Boolean(false), dtype=Bool)
  }
  let out = alloc(a.shape(), Bool)
  let mut at = 0
  for_each_offset(a, fn(offset) raise VecError {
    out.buffer.write(at, Boolean(a.buffer.read(offset).to_double().is_nan()))
    at = at + 1
  })
  out
}

///|
/// True where the value is an infinity.
pub fn isinf(a : NDArray) -> NDArray raise VecError {
  if !a.dtype().is_float() {
    return full_like(a, Boolean(false), dtype=Bool)
  }
  let out = alloc(a.shape(), Bool)
  let mut at = 0
  for_each_offset(a, fn(offset) raise VecError {
    out.buffer.write(at, Boolean(a.buffer.read(offset).to_double().is_inf()))
    at = at + 1
  })
  out
}

///|
/// True where the value is neither infinite nor a NaN; integers are always finite.
pub fn isfinite(a : NDArray) -> NDArray raise VecError {
  let infinite = isinf(a)
  let missing = isnan(a)
  logical_not(logical_or(infinite, missing))
}

///|
/// True where the value is negative, including `-0.0`.
pub fn signbit(a : NDArray) -> NDArray raise VecError {
  if a.dtype().is_complex() {
    raise Domain(message="signbit is not defined on complex numbers")
  }
  let out = alloc(a.shape(), Bool)
  let mut at = 0
  for_each_offset(a, fn(offset) raise VecError {
    let value = a.buffer.read(offset)
    let negative = match a.dtype() {
      F16 | F32 | F64 => is_negative(value.to_double())
      U8 | U16 | U32 | U64 | Bool => false
      _ => value.to_int64() < 0L
    }
    out.buffer.write(at, Boolean(negative))
    at = at + 1
  })
  out
}
