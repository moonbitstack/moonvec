// Copyright 2026 Leo Cheng
// SPDX-License-Identifier: Apache-2.0

///|
/// Four doubles per vector, two lanes per register; the width of the widest vector MoonBit exposes
/// today is 128 bits, so a `float64` loop runs two elements at a time and a `float32` loop four.
let lanes_f64 : Int32 = 2

///|
let lanes_f32 : Int32 = 4

///|
/// Runs the four arithmetic operations over doubles with vector instructions; returns false when the
/// operation has no vector form, so the caller falls back to the scalar loop.
///
/// The tail shorter than one vector runs in the scalar loop, which is also the reference the vector
/// path is tested against.
fn binary_vector_f64(
  op : BinaryOp,
  a : Buffer,
  a_at : Int32,
  b : Buffer,
  b_at : Int32,
  out : Buffer,
  out_at : Int32,
  len : Int32,
) -> Bool raise VecError {
  let vector_len = len - len % lanes_f64
  if vector_len == 0 {
    return false
  }
  match op {
    Add =>
      apply_vector_f64(
        a, a_at, b, b_at, out, out_at, vector_len, @v128.f64x2_add,
      )
    Subtract =>
      apply_vector_f64(
        a, a_at, b, b_at, out, out_at, vector_len, @v128.f64x2_sub,
      )
    Multiply =>
      apply_vector_f64(
        a, a_at, b, b_at, out, out_at, vector_len, @v128.f64x2_mul,
      )
    Divide =>
      apply_vector_f64(
        a, a_at, b, b_at, out, out_at, vector_len, @v128.f64x2_div,
      )
    _ => return false
  }
  let mut i = vector_len
  while i < len {
    out.write(
      out_at + i,
      apply_binary(op, F64, a.read(a_at + i), b.read(b_at + i)),
    )
    i = i + 1
  }
  true
}

///|
fn apply_vector_f64(
  a : Buffer,
  a_at : Int32,
  b : Buffer,
  b_at : Int32,
  out : Buffer,
  out_at : Int32,
  vector_len : Int32,
  combine : (V128, V128) -> V128,
) -> Unit {
  let mut i = 0
  while i < vector_len {
    let x = @v128.v128_load(a.data, (a_at + i) * 8)
    let y = @v128.v128_load(b.data, (b_at + i) * 8)
    @v128.v128_store(out.data, (out_at + i) * 8, combine(x, y))
    i = i + lanes_f64
  }
}

///|
fn binary_vector_f32(
  op : BinaryOp,
  a : Buffer,
  a_at : Int32,
  b : Buffer,
  b_at : Int32,
  out : Buffer,
  out_at : Int32,
  len : Int32,
) -> Bool raise VecError {
  let vector_len = len - len % lanes_f32
  if vector_len == 0 {
    return false
  }
  match op {
    Add =>
      apply_vector_f32(
        a, a_at, b, b_at, out, out_at, vector_len, @v128.f32x4_add,
      )
    Subtract =>
      apply_vector_f32(
        a, a_at, b, b_at, out, out_at, vector_len, @v128.f32x4_sub,
      )
    Multiply =>
      apply_vector_f32(
        a, a_at, b, b_at, out, out_at, vector_len, @v128.f32x4_mul,
      )
    Divide =>
      apply_vector_f32(
        a, a_at, b, b_at, out, out_at, vector_len, @v128.f32x4_div,
      )
    _ => return false
  }
  let mut i = vector_len
  while i < len {
    out.write(
      out_at + i,
      apply_binary(op, F32, a.read(a_at + i), b.read(b_at + i)),
    )
    i = i + 1
  }
  true
}

///|
fn apply_vector_f32(
  a : Buffer,
  a_at : Int32,
  b : Buffer,
  b_at : Int32,
  out : Buffer,
  out_at : Int32,
  vector_len : Int32,
  combine : (V128, V128) -> V128,
) -> Unit {
  let mut i = 0
  while i < vector_len {
    let x = @v128.v128_load(a.data, (a_at + i) * 4)
    let y = @v128.v128_load(b.data, (b_at + i) * 4)
    @v128.v128_store(out.data, (out_at + i) * 4, combine(x, y))
    i = i + lanes_f32
  }
}

///|
/// The unary operations that map to one vector instruction.
fn unary_vector_f64(
  op : UnaryOp,
  a : Buffer,
  a_at : Int32,
  out : Buffer,
  out_at : Int32,
  len : Int32,
) -> Bool raise VecError {
  let vector_len = len - len % lanes_f64
  if vector_len == 0 {
    return false
  }
  let mut i = 0
  match op {
    Negate =>
      while i < vector_len {
        let x = @v128.v128_load(a.data, (a_at + i) * 8)
        @v128.v128_store(out.data, (out_at + i) * 8, @v128.f64x2_neg(x))
        i = i + lanes_f64
      }
    Absolute =>
      while i < vector_len {
        let x = @v128.v128_load(a.data, (a_at + i) * 8)
        @v128.v128_store(out.data, (out_at + i) * 8, @v128.f64x2_abs(x))
        i = i + lanes_f64
      }
    Sqrt =>
      while i < vector_len {
        let x = @v128.v128_load(a.data, (a_at + i) * 8)
        @v128.v128_store(out.data, (out_at + i) * 8, @v128.f64x2_sqrt(x))
        i = i + lanes_f64
      }
    _ => return false
  }
  while i < len {
    out.write(out_at + i, apply_unary(op, F64, a.read(a_at + i)))
    i = i + 1
  }
  true
}

///|
fn unary_vector_f32(
  op : UnaryOp,
  a : Buffer,
  a_at : Int32,
  out : Buffer,
  out_at : Int32,
  len : Int32,
) -> Bool raise VecError {
  let vector_len = len - len % lanes_f32
  if vector_len == 0 {
    return false
  }
  let mut i = 0
  match op {
    Negate =>
      while i < vector_len {
        let x = @v128.v128_load(a.data, (a_at + i) * 4)
        @v128.v128_store(out.data, (out_at + i) * 4, @v128.f32x4_neg(x))
        i = i + lanes_f32
      }
    Absolute =>
      while i < vector_len {
        let x = @v128.v128_load(a.data, (a_at + i) * 4)
        @v128.v128_store(out.data, (out_at + i) * 4, @v128.f32x4_abs(x))
        i = i + lanes_f32
      }
    Sqrt =>
      while i < vector_len {
        let x = @v128.v128_load(a.data, (a_at + i) * 4)
        @v128.v128_store(out.data, (out_at + i) * 4, @v128.f32x4_sqrt(x))
        i = i + lanes_f32
      }
    _ => return false
  }
  while i < len {
    out.write(out_at + i, apply_unary(op, F32, a.read(a_at + i)))
    i = i + 1
  }
  true
}

///|
/// Sums a run of doubles two lanes at a time, then folds the lanes and the tail.
///
/// Floating point addition is not associative, so this does not promise the bits a strictly
/// sequential loop would give; a sum is compared against the reference within a tolerance, while
/// every elementwise operation does promise the same bits on every path.
fn reduce_sum_f64(
  a : Buffer,
  a_at : Int32,
  len : Int32,
) -> Double raise VecError {
  let vector_len = len - len % lanes_f64
  let mut acc = @v128.f64x2_splat(0.0)
  let mut i = 0
  while i < vector_len {
    acc = @v128.f64x2_add(acc, @v128.v128_load(a.data, (a_at + i) * 8))
    i = i + lanes_f64
  }
  let mut total = @v128.f64x2_extract_lane(acc, 0)
  let mut lane = 1
  while lane < lanes_f64 {
    total = total + @v128.f64x2_extract_lane(acc, lane)
    lane = lane + 1
  }
  while i < len {
    total = total + a.read(a_at + i).to_double()
    i = i + 1
  }
  total
}
