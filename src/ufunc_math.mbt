// Copyright 2026 Leo Cheng
// SPDX-License-Identifier: Apache-2.0

///|
/// Elementwise square root; a negative input gives NaN, as IEEE 754 says, rather than an error.
pub fn sqrt(a : NDArray, kernel? : &Kernel) -> NDArray raise VecError {
  math_unary(a, Sqrt, pick_kernel(kernel))
}

///|
/// Elementwise cube root, defined for negative inputs too: `cbrt(-8)` is `-2`.
pub fn cbrt(a : NDArray, kernel? : &Kernel) -> NDArray raise VecError {
  math_unary(a, Cbrt, pick_kernel(kernel))
}

///|
/// `e` raised to each element.
pub fn exp(a : NDArray, kernel? : &Kernel) -> NDArray raise VecError {
  math_unary(a, Exp, pick_kernel(kernel))
}

///|
/// Two raised to each element.
pub fn exp2(a : NDArray, kernel? : &Kernel) -> NDArray raise VecError {
  math_unary(a, Exp2, pick_kernel(kernel))
}

///|
/// `exp(a) - 1`, accurate near zero.
pub fn expm1(a : NDArray, kernel? : &Kernel) -> NDArray raise VecError {
  math_unary(a, Expm1, pick_kernel(kernel))
}

///|
/// Natural logarithm; zero gives minus infinity and a negative input gives NaN.
pub fn log(a : NDArray, kernel? : &Kernel) -> NDArray raise VecError {
  math_unary(a, Log, pick_kernel(kernel))
}

///|
/// Base-two logarithm, with the same treatment of non-positive input as `log`.
pub fn log2(a : NDArray, kernel? : &Kernel) -> NDArray raise VecError {
  math_unary(a, Log2, pick_kernel(kernel))
}

///|
/// Base-ten logarithm, with the same treatment of non-positive input as `log`.
pub fn log10(a : NDArray, kernel? : &Kernel) -> NDArray raise VecError {
  math_unary(a, Log10, pick_kernel(kernel))
}

///|
/// `log(1 + a)`, accurate near zero.
pub fn log1p(a : NDArray, kernel? : &Kernel) -> NDArray raise VecError {
  math_unary(a, Log1p, pick_kernel(kernel))
}

///|
/// Sine of each element, in radians.
pub fn sin(a : NDArray, kernel? : &Kernel) -> NDArray raise VecError {
  math_unary(a, Sin, pick_kernel(kernel))
}

///|
/// Cosine of each element, in radians.
pub fn cos(a : NDArray, kernel? : &Kernel) -> NDArray raise VecError {
  math_unary(a, Cos, pick_kernel(kernel))
}

///|
/// Tangent of each element, in radians.
pub fn tan(a : NDArray, kernel? : &Kernel) -> NDArray raise VecError {
  math_unary(a, Tan, pick_kernel(kernel))
}

///|
/// Inverse sine of each element, in radians; outside `[-1, 1]` it is NaN.
pub fn arcsin(a : NDArray, kernel? : &Kernel) -> NDArray raise VecError {
  math_unary(a, ArcSin, pick_kernel(kernel))
}

///|
/// Inverse cosine of each element, in radians; outside `[-1, 1]` it is NaN.
pub fn arccos(a : NDArray, kernel? : &Kernel) -> NDArray raise VecError {
  math_unary(a, ArcCos, pick_kernel(kernel))
}

///|
/// Inverse tangent of each element, in radians.
pub fn arctan(a : NDArray, kernel? : &Kernel) -> NDArray raise VecError {
  math_unary(a, ArcTan, pick_kernel(kernel))
}

///|
/// Hyperbolic sine of each element.
pub fn sinh(a : NDArray, kernel? : &Kernel) -> NDArray raise VecError {
  math_unary(a, Sinh, pick_kernel(kernel))
}

///|
/// Hyperbolic cosine of each element.
pub fn cosh(a : NDArray, kernel? : &Kernel) -> NDArray raise VecError {
  math_unary(a, Cosh, pick_kernel(kernel))
}

///|
/// Hyperbolic tangent of each element.
pub fn tanh(a : NDArray, kernel? : &Kernel) -> NDArray raise VecError {
  math_unary(a, Tanh, pick_kernel(kernel))
}

///|
/// Inverse hyperbolic sine of each element.
pub fn arcsinh(a : NDArray, kernel? : &Kernel) -> NDArray raise VecError {
  math_unary(a, ArcSinh, pick_kernel(kernel))
}

///|
/// Inverse hyperbolic cosine of each element; below one it is NaN.
pub fn arccosh(a : NDArray, kernel? : &Kernel) -> NDArray raise VecError {
  math_unary(a, ArcCosh, pick_kernel(kernel))
}

///|
/// Inverse hyperbolic tangent of each element; outside `(-1, 1)` it is NaN.
pub fn arctanh(a : NDArray, kernel? : &Kernel) -> NDArray raise VecError {
  math_unary(a, ArcTanh, pick_kernel(kernel))
}

///|
/// Rounds down.
pub fn floor(a : NDArray, kernel? : &Kernel) -> NDArray raise VecError {
  math_unary(a, Floor, pick_kernel(kernel))
}

///|
/// Rounds up.
pub fn ceil(a : NDArray, kernel? : &Kernel) -> NDArray raise VecError {
  math_unary(a, Ceil, pick_kernel(kernel))
}

///|
/// Drops the fractional part, towards zero.
pub fn trunc(a : NDArray, kernel? : &Kernel) -> NDArray raise VecError {
  math_unary(a, Trunc, pick_kernel(kernel))
}

///|
/// Rounds to the nearest integer with a tie going to the even one, the way IEEE 754 does.
pub fn rint(a : NDArray, kernel? : &Kernel) -> NDArray raise VecError {
  math_unary(a, Rint, pick_kernel(kernel))
}

///|
/// Rounds to the nearest integer, a tie going away from zero; `round` in Python 3 goes to even, so
/// this is `rint`'s sibling, not a replacement for it.
pub fn round(a : NDArray) -> NDArray raise VecError {
  let out = alloc(a.shape(), a.dtype())
  let mut at = 0
  for_each_offset(a, fn(offset) raise VecError {
    out.buffer.write(
      at,
      Real(round_half_away(a.buffer.read(offset).to_double())),
    )
    at = at + 1
  })
  out
}

///|
/// Converts radians to degrees.
pub fn degrees(a : NDArray, kernel? : &Kernel) -> NDArray raise VecError {
  math_unary(a, Degrees, pick_kernel(kernel))
}

///|
/// Converts degrees to radians.
pub fn radians(a : NDArray, kernel? : &Kernel) -> NDArray raise VecError {
  math_unary(a, Radians, pick_kernel(kernel))
}

///|
/// The real part; a real array is returned as it is.
pub fn real(a : NDArray) -> NDArray raise VecError {
  match a.dtype() {
    C64 => complex_part(a, F32, 0)
    C128 => complex_part(a, F64, 0)
    _ => a.copy()
  }
}

///|
/// The imaginary part; a real array gives zeros.
pub fn imag(a : NDArray) -> NDArray raise VecError {
  match a.dtype() {
    C64 => complex_part(a, F32, 4)
    C128 => complex_part(a, F64, 8)
    _ => zeros(a.shape(), dtype=a.dtype())
  }
}

///|
/// The angle of each complex value in radians; a real array is `0` where positive and `pi` where
/// negative.
pub fn angle(a : NDArray) -> NDArray raise VecError {
  let out_dtype = if a.dtype() == C64 || a.dtype() == F32 { F32 } else { F64 }
  let out = alloc(a.shape(), out_dtype)
  let mut at = 0
  for_each_offset(a, fn(offset) raise VecError {
    let value = a.buffer.read(offset)
    let radians = if a.dtype().is_complex() {
      value.to_complex().arg()
    } else if value.to_double() < 0.0 {
      3.141592653589793
    } else {
      0.0
    }
    out.buffer.write(at, Real(radians))
    at = at + 1
  })
  out
}

///|
/// Elementwise conjugate; a real array is returned as it is.
pub fn conj(a : NDArray, kernel? : &Kernel) -> NDArray raise VecError {
  unary_map(a, a.dtype(), a.dtype(), Conjugate, pick_kernel(kernel))
}

///|
/// Reads one component out of a complex buffer into a real array.
fn complex_part(
  a : NDArray,
  out_dtype : DType,
  skew : Int32,
) -> NDArray raise VecError {
  let out = alloc(a.shape(), out_dtype)
  let size = a.dtype().item_size()
  let mut at = 0
  for_each_offset(a, fn(offset) raise VecError {
    let at_bytes = offset * size + skew
    let value = if out_dtype == F32 {
      load_f32(a.buffer.data, at_bytes)
    } else {
      load_f64(a.buffer.data, at_bytes)
    }
    out.buffer.write(at, Real(value))
    at = at + 1
  })
  out
}

///|
/// Rounds a double to the nearest integer, a tie going away from zero.
fn round_half_away(value : Double) -> Double {
  if value >= 0.0 {
    (value + 0.5).floor()
  } else {
    (value - 0.5).ceil()
  }
}

///|
/// Applies a function of one double to every element, keeping the array's float width.
///
/// The special functions come from the mathematical library rather than being written again here, so
/// they do not go through the kernel: they are one implementation, and a backend would have to match
/// it exactly to be worth having.
fn map_real(a : NDArray, f : (Double) -> Double) -> NDArray raise VecError {
  let dtype = math_dtype(a.dtype())
  if dtype.is_complex() {
    raise Domain(message="this function is not defined on complex numbers")
  }
  let out = alloc(a.shape(), dtype)
  let mut at = 0
  for_each_offset(a, fn(offset) raise VecError {
    out.buffer.write(at, Real(f(a.buffer.read(offset).to_double())))
    at = at + 1
  })
  out
}

///|
/// The gamma function, from the special-function library.
pub fn gamma(a : NDArray) -> NDArray raise VecError {
  map_real(a, fn(x) {
    if x.is_nan() {
      x
    } else if x == @double.infinity {
      x
    } else {
      @special.gamma(x)
    }
  })
}

///|
/// The natural logarithm of the absolute value of gamma, which stays finite where `gamma` overflows.
pub fn lgamma(a : NDArray) -> NDArray raise VecError {
  map_real(a, fn(x) {
    if x.is_nan() || x == @double.infinity {
      x
    } else {
      @special.lgamma(x)
    }
  })
}

///|
/// The logarithmic derivative of gamma.
pub fn digamma(a : NDArray) -> NDArray raise VecError {
  map_real(a, @special.digamma)
}

///|
/// The error function.
pub fn erf(a : NDArray) -> NDArray raise VecError {
  map_real(a, fn(x) {
    if x.is_nan() {
      x
    } else if x == @double.infinity {
      1.0
    } else if x == @double.neg_infinity {
      -1.0
    } else {
      @special.erf(x)
    }
  })
}

///|
/// The complementary error function, `1 - erf(x)`, accurate for large arguments.
pub fn erfc(a : NDArray) -> NDArray raise VecError {
  map_real(a, fn(x) {
    if x == @double.infinity {
      0.0
    } else if x == @double.neg_infinity {
      2.0
    } else {
      @special.erfc(x)
    }
  })
}

///|
/// The inverse of the error function.
pub fn erfinv(a : NDArray) -> NDArray raise VecError {
  map_real(a, @special.erfinv)
}

///|
/// The inverse of the complementary error function.
pub fn erfcinv(a : NDArray) -> NDArray raise VecError {
  map_real(a, @special.erfcinv)
}

///|
/// The beta function, elementwise over two broadcast operands.
pub fn beta(a : NDArray, b : NDArray) -> NDArray raise VecError {
  let dtype = math_dtype(promote(a.dtype(), b.dtype()))
  if dtype.is_complex() {
    raise Domain(message="beta is not defined on complex numbers")
  }
  let shape = broadcast_shape(a.shape(), b.shape())
  let out = alloc(shape, dtype)
  let plan = plan_loop(
    shape,
    [
      contiguous_strides(shape),
      broadcast_strides(a.shape(), a.strides(), shape),
      broadcast_strides(b.shape(), b.strides(), shape),
    ],
    [0, a.offset(), b.offset()],
  )
  plan.run(fn(bases, len) raise VecError {
    for i in 0..<len {
      let left = a.buffer.read(bases[1] + i * plan.run_strides[1]).to_double()
      let right = b.buffer.read(bases[2] + i * plan.run_strides[2]).to_double()
      out.buffer.write(
        bases[0] + i * plan.run_strides[0],
        Real(@special.beta(left, right)),
      )
    }
  })
  out
}

///|
/// The Riemann zeta function.
pub fn zeta(a : NDArray) -> NDArray raise VecError {
  map_real(a, @special.zeta)
}

///|
/// The Bessel function of the first kind of order `order`.
pub fn bessel_j(order~ : Double, a : NDArray) -> NDArray raise VecError {
  map_real(a, fn(x) { @special.bessel_j(order, x) })
}

///|
/// The Bessel function of the second kind of order `order`.
pub fn bessel_y(order~ : Double, a : NDArray) -> NDArray raise VecError {
  map_real(a, fn(x) { @special.bessel_y(order, x) })
}

///|
/// The modified Bessel function of the first kind of order `order`.
pub fn bessel_i(order~ : Double, a : NDArray) -> NDArray raise VecError {
  map_real(a, fn(x) { @special.bessel_i(order, x) })
}

///|
/// The modified Bessel function of the second kind of order `order`.
pub fn bessel_k(order~ : Double, a : NDArray) -> NDArray raise VecError {
  map_real(a, fn(x) { @special.bessel_k(order, x) })
}
