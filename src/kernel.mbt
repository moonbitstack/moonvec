// Copyright 2026 Leo Cheng
// SPDX-License-Identifier: Apache-2.0

///|
/// What an array backend has to provide: the arithmetic over runs of contiguous elements.
///
/// Everything else — broadcasting, striding, dtype promotion, allocation — belongs to the array
/// layer above, so a backend only has to be fast on long flat runs and never sees a shape. A run is
/// always within one buffer and one dtype, and the offsets are in elements.
///
/// Implement this trait to add a backend: a C stub, a compute shader, a vector unit. The array
/// layer takes it as a parameter and does not change.
pub(open) trait Kernel {
  ///|
  /// The name the backend reports, for diagnostics.
  fn name(Self) -> String
  ///|
  /// `out[i] = a[i] op b[i]` for `len` elements; `dtype` is the type both operands are read as, and
  /// the output buffer decides what the result is stored as.
  fn binary(
    Self,
    op : BinaryOp,
    dtype : DType,
    a : Buffer,
    a_at : Int32,
    a_step : Int32,
    b : Buffer,
    b_at : Int32,
    b_step : Int32,
    out : Buffer,
    out_at : Int32,
    out_step : Int32,
    len : Int32,
  ) -> Unit raise VecError
  ///|
  /// `out[i] = op a[i]` for `len` elements.
  fn unary(
    Self,
    op : UnaryOp,
    dtype : DType,
    a : Buffer,
    a_at : Int32,
    a_step : Int32,
    out : Buffer,
    out_at : Int32,
    out_step : Int32,
    len : Int32,
  ) -> Unit raise VecError
  ///|
  /// Multiplies an `m` by `k` block by a `k` by `n` block, both row-major and contiguous, into an
  /// `m` by `n` block. The accumulation happens in `dtype`, so an integer product stays exact.
  fn matmul(
    Self,
    dtype : DType,
    a : Buffer,
    a_at : Int32,
    b : Buffer,
    b_at : Int32,
    out : Buffer,
    out_at : Int32,
    m : Int32,
    k : Int32,
    n : Int32,
  ) -> Unit raise VecError
  ///|
  /// Folds `len` elements into `into` with the operation's pair, returning the new accumulator.
  /// Starting from an explicit value rather than the identity lets a caller fold an axis in slices.
  fn reduce(
    Self,
    op : ReduceOp,
    dtype : DType,
    a : Buffer,
    a_at : Int32,
    a_step : Int32,
    len : Int32,
    into : Number,
  ) -> Number raise VecError
}

///|
/// The only backend there is for now: plain MoonBit, with a vector path where the elements are
/// doubles or floats and the operation is one of the four arithmetic ones.
pub struct Cpu {
  /// Elements per vector register: two doubles or four floats.
  lanes : Int32
}

///|
/// The default backend, sized for the vector registers the build target provides.
pub fn Cpu::new() -> Cpu {
  { lanes: lanes_f64, }
}

///|
/// The backend every function uses when the caller does not name one.
pub let cpu : Cpu = Cpu::new()

///|
impl Kernel for Cpu with fn name(_self) {
  "cpu"
}

///|
impl Kernel for Cpu with fn binary(
  _self,
  op,
  dtype,
  a,
  a_at,
  a_step,
  b,
  b_at,
  b_step,
  out,
  out_at,
  out_step,
  len,
) {
  // The vector path reads doubles and floats straight out of the buffers, so it only applies when
  // every buffer really holds that dtype; a mixed pair is converted element by element instead.
  let same_dtype = a.dtype() == dtype &&
    b.dtype() == dtype &&
    out.dtype() == dtype
  if a_step == 1 && b_step == 1 && out_step == 1 && same_dtype {
    match dtype {
      F64 =>
        if binary_vector_f64(op, a, a_at, b, b_at, out, out_at, len) {
          return
        }
      F32 =>
        if binary_vector_f32(op, a, a_at, b, b_at, out, out_at, len) {
          return
        }
      _ => ()
    }
    for i in 0..<len {
      out.write(
        out_at + i,
        apply_binary(op, dtype, a.read(a_at + i), b.read(b_at + i)),
      )
    }
    return
  }
  for i in 0..<len {
    let value = apply_binary(
      op,
      dtype,
      a.read(a_at + i * a_step),
      b.read(b_at + i * b_step),
    )
    out.write(out_at + i * out_step, value)
  }
}

///|
impl Kernel for Cpu with fn unary(
  _self,
  op,
  dtype,
  a,
  a_at,
  a_step,
  out,
  out_at,
  out_step,
  len,
) {
  if a_step == 1 && out_step == 1 && a.dtype() == dtype && out.dtype() == dtype {
    match dtype {
      F64 => if unary_vector_f64(op, a, a_at, out, out_at, len) { return }
      F32 => if unary_vector_f32(op, a, a_at, out, out_at, len) { return }
      _ => ()
    }
    for i in 0..<len {
      out.write(out_at + i, apply_unary(op, dtype, a.read(a_at + i)))
    }
    return
  }
  for i in 0..<len {
    out.write(
      out_at + i * out_step,
      apply_unary(op, dtype, a.read(a_at + i * a_step)),
    )
  }
}

///|
impl Kernel for Cpu with fn matmul(
  _self,
  dtype,
  a,
  a_at,
  b,
  b_at,
  out,
  out_at,
  m,
  k,
  n,
) {
  let accumulators = Array::make(m * n, reduce_identity(Sum, dtype))
  for i in 0..<m {
    for inner in 0..<k {
      let left = a.read(a_at + i * k + inner)
      for j in 0..<n {
        let right = b.read(b_at + inner * n + j)
        let product = apply_binary(Multiply, dtype, left, right)
        let slot = i * n + j
        accumulators[slot] = apply_binary(
          Add,
          dtype,
          accumulators[slot],
          product,
        )
      }
    }
  }
  let total = m * n
  for i in 0..<total {
    out.write(out_at + i, accumulators[i])
  }
}

///|
impl Kernel for Cpu with fn reduce(_self, op, dtype, a, a_at, a_step, len, into) {
  match (op, dtype) {
    (Sum, F64) =>
      if a_step == 1 && len >= 8 && a.dtype() == F64 {
        return Real(reduce_sum_f64(a, a_at, len) + into.to_double())
      }
    _ => ()
  }
  let mut total = into
  for i in 0..<len {
    total = apply_reduce(op, dtype, total, a.read(a_at + i * a_step))
  }
  total
}
