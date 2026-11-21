// Copyright 2026 Leo Cheng
// SPDX-License-Identifier: Apache-2.0

///|
/// A copy of the array with every element converted to `dtype`.
///
/// The conversion is the one C makes: an integer that does not fit wraps, a float is truncated
/// towards zero when it becomes an integer, and a NaN becomes zero, because there is no integer it
/// could honestly become.
pub fn astype(a : NDArray, dtype~ : DType) -> NDArray raise VecError {
  if a.dtype() == dtype {
    return a.copy()
  }
  let out = alloc(a.shape(), dtype)
  let mut at = 0
  for_each_offset(a, fn(offset) raise VecError {
    out.buffer.write(at, a.buffer.read(offset))
    at = at + 1
  })
  out
}
