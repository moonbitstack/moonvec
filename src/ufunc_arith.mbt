// Copyright 2026 Leo Cheng
// SPDX-License-Identifier: Apache-2.0

///|
/// Elementwise sum; the two shapes broadcast.
pub fn add(
  a : NDArray,
  b : NDArray,
  kernel? : &Kernel,
) -> NDArray raise VecError {
  binary_with(a, b, Add, pick_kernel(kernel))
}

///|
/// Elementwise difference.
pub fn subtract(
  a : NDArray,
  b : NDArray,
  kernel? : &Kernel,
) -> NDArray raise VecError {
  binary_with(a, b, Subtract, pick_kernel(kernel))
}

///|
/// Elementwise product.
pub fn multiply(
  a : NDArray,
  b : NDArray,
  kernel? : &Kernel,
) -> NDArray raise VecError {
  binary_with(a, b, Multiply, pick_kernel(kernel))
}

///|
/// True division: two integer arrays give `float64`, so `1 / 2` is `0.5` rather than `0`.
pub fn divide(
  a : NDArray,
  b : NDArray,
  kernel? : &Kernel,
) -> NDArray raise VecError {
  binary_with(a, b, Divide, pick_kernel(kernel))
}

///|
/// Division rounded towards negative infinity; both operands keep their integer type.
pub fn floor_divide(
  a : NDArray,
  b : NDArray,
  kernel? : &Kernel,
) -> NDArray raise VecError {
  binary_with(a, b, FloorDivide, pick_kernel(kernel))
}

///|
/// The remainder with the sign of the divisor, so `-3 % 2` is `1`.
pub fn remainder(
  a : NDArray,
  b : NDArray,
  kernel? : &Kernel,
) -> NDArray raise VecError {
  binary_with(a, b, Remainder, pick_kernel(kernel))
}

///|
/// The same as `remainder`, under the name Python uses for the operator.
pub fn mod(
  a : NDArray,
  b : NDArray,
  kernel? : &Kernel,
) -> NDArray raise VecError {
  binary_with(a, b, Remainder, pick_kernel(kernel))
}

///|
/// Elementwise power; an integer base with a negative integer exponent is refused.
pub fn power(
  a : NDArray,
  b : NDArray,
  kernel? : &Kernel,
) -> NDArray raise VecError {
  binary_with(a, b, Power, pick_kernel(kernel))
}

///|
/// Elementwise larger of the two; a NaN on either side gives NaN.
pub fn maximum(
  a : NDArray,
  b : NDArray,
  kernel? : &Kernel,
) -> NDArray raise VecError {
  binary_with(a, b, Maximum, pick_kernel(kernel))
}

///|
/// Elementwise smaller of the two; a NaN on either side gives NaN.
pub fn minimum(
  a : NDArray,
  b : NDArray,
  kernel? : &Kernel,
) -> NDArray raise VecError {
  binary_with(a, b, Minimum, pick_kernel(kernel))
}

///|
/// Elementwise arctangent of `a / b`, choosing the quadrant from both signs.
pub fn arctan2(
  a : NDArray,
  b : NDArray,
  kernel? : &Kernel,
) -> NDArray raise VecError {
  let dtype = math_dtype(promote(a.dtype(), b.dtype()))
  binary_map(a, b, dtype, dtype, ArcTangent2, pick_kernel(kernel))
}

///|
/// Elementwise `sqrt(a * a + b * b)`, without the overflow the naive form has.
pub fn hypot(
  a : NDArray,
  b : NDArray,
  kernel? : &Kernel,
) -> NDArray raise VecError {
  let dtype = math_dtype(promote(a.dtype(), b.dtype()))
  binary_map(a, b, dtype, dtype, Hypotenuse, pick_kernel(kernel))
}

///|
/// Elementwise bitwise and; both operands must be integers.
pub fn bitwise_and(
  a : NDArray,
  b : NDArray,
  kernel? : &Kernel,
) -> NDArray raise VecError {
  binary_with(a, b, BitAnd, pick_kernel(kernel))
}

///|
/// Elementwise bitwise or; both operands are read as integers.
pub fn bitwise_or(
  a : NDArray,
  b : NDArray,
  kernel? : &Kernel,
) -> NDArray raise VecError {
  binary_with(a, b, BitOr, pick_kernel(kernel))
}

///|
/// Elementwise bitwise exclusive or; both operands are read as integers.
pub fn bitwise_xor(
  a : NDArray,
  b : NDArray,
  kernel? : &Kernel,
) -> NDArray raise VecError {
  binary_with(a, b, BitXor, pick_kernel(kernel))
}

///|
/// Elementwise left shift, with the second array as the count.
pub fn left_shift(
  a : NDArray,
  b : NDArray,
  kernel? : &Kernel,
) -> NDArray raise VecError {
  binary_with(a, b, LeftShift, pick_kernel(kernel))
}

///|
/// Elementwise right shift, with the second array as the count.
pub fn right_shift(
  a : NDArray,
  b : NDArray,
  kernel? : &Kernel,
) -> NDArray raise VecError {
  binary_with(a, b, RightShift, pick_kernel(kernel))
}

///|
/// Elementwise negation.
pub fn negative(a : NDArray, kernel? : &Kernel) -> NDArray raise VecError {
  let dtype = a.dtype()
  unary_map(a, dtype, dtype, Negate, pick_kernel(kernel))
}

///|
/// A copy with every value's sign flipped, which for `bool` gives `-1` for true.
pub fn positive(a : NDArray) -> NDArray raise VecError {
  a.copy()
}

///|
/// The magnitude; a complex array answers in its real dtype.
pub fn absolute(a : NDArray, kernel? : &Kernel) -> NDArray raise VecError {
  let out_dtype = if a.dtype().is_complex() {
    if a.dtype() == C64 {
      F32
    } else {
      F64
    }
  } else {
    a.dtype()
  }
  unary_map(a, a.dtype(), out_dtype, Absolute, pick_kernel(kernel))
}

///|
/// The magnitude of a real array, as floating point; the same as `absolute` for floats.
pub fn fabs(a : NDArray, kernel? : &Kernel) -> NDArray raise VecError {
  let dtype = if a.dtype().is_float() { a.dtype() } else { F64 }
  unary_map(a, dtype, dtype, Absolute, pick_kernel(kernel))
}

///|
/// `-1`, `0` or `1` by sign; a complex argument gives the value divided by its magnitude.
pub fn sign(a : NDArray, kernel? : &Kernel) -> NDArray raise VecError {
  let dtype = a.dtype()
  unary_map(a, dtype, dtype, Sign, pick_kernel(kernel))
}

///|
/// Elementwise square.
pub fn square(a : NDArray, kernel? : &Kernel) -> NDArray raise VecError {
  let dtype = a.dtype()
  unary_map(a, dtype, dtype, Square, pick_kernel(kernel))
}

///|
/// Elementwise reciprocal, which for integers gives `float64`.
pub fn reciprocal(a : NDArray, kernel? : &Kernel) -> NDArray raise VecError {
  let dtype = math_dtype(a.dtype())
  unary_map(a, dtype, dtype, Reciprocal, pick_kernel(kernel))
}

///|
/// Sums two arrays with `+`; a shape mismatch aborts, because an operator cannot raise. Call `add`
/// to get the error value instead.
pub impl Add for NDArray with fn add(self, other) {
  match (Ok(add(self, other)) catch { error => Err(error) }) {
    Ok(value) => value
    Err(error) => abort(error.message())
  }
}

///|
pub impl Sub for NDArray with fn sub(self, other) {
  match (Ok(subtract(self, other)) catch { error => Err(error) }) {
    Ok(value) => value
    Err(error) => abort(error.message())
  }
}

///|
pub impl Mul for NDArray with fn mul(self, other) {
  match (Ok(multiply(self, other)) catch { error => Err(error) }) {
    Ok(value) => value
    Err(error) => abort(error.message())
  }
}

///|
pub impl Div for NDArray with fn div(self, other) {
  match (Ok(divide(self, other)) catch { error => Err(error) }) {
    Ok(value) => value
    Err(error) => abort(error.message())
  }
}

///|
pub impl Mod for NDArray with fn mod(self, other) {
  match (Ok(remainder(self, other)) catch { error => Err(error) }) {
    Ok(value) => value
    Err(error) => abort(error.message())
  }
}

///|
pub impl Neg for NDArray with fn neg(self) {
  match (Ok(negative(self)) catch { error => Err(error) }) {
    Ok(value) => value
    Err(error) => abort(error.message())
  }
}

///|
/// Adds a scalar to every element, keeping the array's dtype; `1` added to an `int8` array stays
/// `int8`, because a scalar does not widen the array.
pub fn add_scalar(
  a : NDArray,
  value : Double,
  kernel? : &Kernel,
) -> NDArray raise VecError {
  scalar_operation(a, value, Add, kernel?)
}

///|
/// Subtracts a scalar from every element, keeping the array's dtype.
pub fn subtract_scalar(
  a : NDArray,
  value : Double,
  kernel? : &Kernel,
) -> NDArray raise VecError {
  scalar_operation(a, value, Subtract, kernel?)
}

///|
/// Multiplies every element by a scalar, keeping the array's dtype.
pub fn multiply_scalar(
  a : NDArray,
  value : Double,
  kernel? : &Kernel,
) -> NDArray raise VecError {
  scalar_operation(a, value, Multiply, kernel?)
}

///|
/// Divides by a scalar; an integer array gives `float64`, as with `divide`.
pub fn divide_scalar(
  a : NDArray,
  value : Double,
  kernel? : &Kernel,
) -> NDArray raise VecError {
  let scalar = full(a.shape(), Real(value), dtype=a.dtype())
  binary_with(a, scalar, Divide, pick_kernel(kernel))
}

///|
/// Raises every element to a scalar power, by the same rules as `power`.
pub fn power_scalar(
  a : NDArray,
  value : Double,
  kernel? : &Kernel,
) -> NDArray raise VecError {
  scalar_operation(a, value, Power, kernel?)
}

///|
/// The larger of each element and a scalar, elementwise.
pub fn maximum_scalar(
  a : NDArray,
  value : Double,
  kernel? : &Kernel,
) -> NDArray raise VecError {
  scalar_operation(a, value, Maximum, kernel?)
}

///|
/// The smaller of each element and a scalar, elementwise.
pub fn minimum_scalar(
  a : NDArray,
  value : Double,
  kernel? : &Kernel,
) -> NDArray raise VecError {
  scalar_operation(a, value, Minimum, kernel?)
}

///|
/// The body of the scalar helpers: the scalar is cast into the array's own dtype first, so the
/// result keeps that dtype.
fn scalar_operation(
  a : NDArray,
  value : Double,
  op : BinaryOp,
  kernel? : &Kernel,
) -> NDArray raise VecError {
  let dtype = a.dtype()
  let scalar = full([], Real(value), dtype~)
  binary_map(a, scalar, dtype, dtype, op, pick_kernel(kernel))
}

///|
/// Clamps every element into `[min, max]`; either bound may be left out, and the bounds broadcast
/// like any other operand.
pub fn clip(
  a : NDArray,
  min? : Double,
  max? : Double,
  kernel? : &Kernel,
) -> NDArray raise VecError {
  let mut out = a
  match min {
    Some(low) => out = maximum_scalar(out, low, kernel?)
    None => ()
  }
  match max {
    Some(high) => out = minimum_scalar(out, high, kernel?)
    None => ()
  }
  out
}

///|
/// Chooses elementwise from `x` where `condition` holds and from `y` elsewhere; all three broadcast
/// together. It is named `select` because `where` is a keyword in MoonBit.
pub fn select(
  condition : NDArray,
  x : NDArray,
  y : NDArray,
) -> NDArray raise VecError {
  if !condition.dtype().is_bool() {
    raise Domain(message="select needs a boolean array as its condition")
  }
  let shape = broadcast_shape(
    broadcast_shape(condition.shape(), x.shape()),
    y.shape(),
  )
  let out = alloc(shape, promote(x.dtype(), y.dtype()))
  let out_dtype = out.dtype()
  let plan = plan_loop(
    shape,
    [
      contiguous_strides(shape),
      broadcast_strides(condition.shape(), condition.strides(), shape),
      broadcast_strides(x.shape(), x.strides(), shape),
      broadcast_strides(y.shape(), y.strides(), shape),
    ],
    [0, condition.offset(), x.offset(), y.offset()],
  )
  plan.run(fn(bases, len) raise VecError {
    for i in 0..<len {
      let take_x = condition.buffer
        .read(bases[1] + i * plan.run_strides[1])
        .to_bool()
      let value = if take_x {
        x.buffer.read(bases[2] + i * plan.run_strides[2])
      } else {
        y.buffer.read(bases[3] + i * plan.run_strides[3])
      }
      out.buffer.write(
        bases[0] + i * plan.run_strides[0],
        Number::cast(value, out_dtype),
      )
    }
  })
  out
}
