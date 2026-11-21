// Copyright 2026 Leo Cheng
// SPDX-License-Identifier: Apache-2.0

///|
/// A complex number, held as a pair of doubles.
///
/// `complex64` arrays store the two parts as single precision floats; the value type always widens
/// to doubles so that arithmetic in a loop does not lose the bits the storage kept.
pub(all) struct Complex {
  re : Double
  im : Double
}

///|
pub fn Complex::new(re : Double, im : Double) -> Complex {
  { re, im, }
}

///|
/// The modulus, computed as a hypotenuse so that large components do not overflow.
pub fn Complex::abs(self : Complex) -> Double {
  @math.hypot(self.re, self.im)
}

///|
/// The angle in radians, in `(-pi, pi]`, the argument `atan2` returns.
pub fn Complex::arg(self : Complex) -> Double {
  @math.atan2(self.im, self.re)
}

///|
pub fn Complex::conj(self : Complex) -> Complex {
  { re: self.re, im: -self.im, }
}

///|
pub fn Complex::neg(self : Complex) -> Complex {
  { re: -self.re, im: -self.im, }
}

///|
pub fn Complex::exp(self : Complex) -> Complex {
  let r = @math.exp(self.re)
  { re: r * @math.cos(self.im), im: r * @math.sin(self.im), }
}

///|
/// The principal logarithm: `ln|z| + i arg z`.
pub fn Complex::log(self : Complex) -> Complex {
  { re: @math.ln(self.abs()), im: self.arg(), }
}

///|
/// The principal square root, on the branch where the result's real part is non-negative.
pub fn Complex::sqrt(self : Complex) -> Complex {
  if self.re == 0.0 && self.im == 0.0 {
    return { re: 0.0, im: 0.0, }
  }
  let r = self.abs()
  if self.re >= 0.0 {
    let t = ((r + self.re) / 2.0).sqrt()
    { re: t, im: self.im / (2.0 * t), }
  } else {
    let t = ((r - self.re) / 2.0).sqrt()
    let im = if self.im < 0.0 { -t } else { t }
    { re: self.im.abs() / (2.0 * t), im, }
  }
}

///|
/// `self` raised to `exponent`, through `exp(exponent * log self)`.
pub fn Complex::pow(self : Complex, exponent : Complex) -> Complex {
  if self.re == 0.0 && self.im == 0.0 {
    if exponent.re == 0.0 && exponent.im == 0.0 {
      return { re: 1.0, im: 0.0, }
    }
    return { re: 0.0, im: 0.0, }
  }
  (exponent * self.log()).exp()
}

///|
pub fn Complex::sin(self : Complex) -> Complex {
  {
    re: @math.sin(self.re) * @math.cosh(self.im),
    im: @math.cos(self.re) * @math.sinh(self.im),
  }
}

///|
pub fn Complex::cos(self : Complex) -> Complex {
  {
    re: @math.cos(self.re) * @math.cosh(self.im),
    im: -@math.sin(self.re) * @math.sinh(self.im),
  }
}

///|
pub fn Complex::tan(self : Complex) -> Complex {
  self.sin() / self.cos()
}

///|
pub fn Complex::sinh(self : Complex) -> Complex {
  {
    re: @math.sinh(self.re) * @math.cos(self.im),
    im: @math.cosh(self.re) * @math.sin(self.im),
  }
}

///|
pub fn Complex::cosh(self : Complex) -> Complex {
  {
    re: @math.cosh(self.re) * @math.cos(self.im),
    im: @math.sinh(self.re) * @math.sin(self.im),
  }
}

///|
pub fn Complex::tanh(self : Complex) -> Complex {
  self.sinh() / self.cosh()
}

///|
pub impl Add for Complex with fn add(self, other) {
  { re: self.re + other.re, im: self.im + other.im, }
}

///|
pub impl Sub for Complex with fn sub(self, other) {
  { re: self.re - other.re, im: self.im - other.im, }
}

///|
pub impl Mul for Complex with fn mul(self, other) {
  {
    re: self.re * other.re - self.im * other.im,
    im: self.re * other.im + self.im * other.re,
  }
}

///|
/// Division by the scaled denominator, which keeps its magnitude in range.
pub impl Div for Complex with fn div(self, other) {
  if !other.is_nonzero() {
    // Dividing by zero: the parts follow IEEE 754 on their own, so a non-zero numerator part becomes
    // an infinity and a zero part becomes a NaN, which is what C99 does.
    return { re: self.re / 0.0, im: self.im / 0.0, }
  }
  if other.re.abs() >= other.im.abs() {
    let r = other.im / other.re
    let d = other.re + other.im * r
    { re: (self.re + self.im * r) / d, im: (self.im - self.re * r) / d, }
  } else {
    let r = other.re / other.im
    let d = other.re * r + other.im
    { re: (self.re * r + self.im) / d, im: (self.im * r - self.re) / d, }
  }
}

///|
pub impl Neg for Complex with fn neg(self) {
  { re: -self.re, im: -self.im, }
}

///|
pub impl Eq for Complex with fn equal(self, other) {
  self.re == other.re && self.im == other.im
}

///|
pub impl Show for Complex with fn to_string(self) {
  self.to_repr_string()
}

///|
/// The form NumPy prints: `(1+2j)`, `(1-2j)`, `(1.5+0j)`.
fn Complex::to_repr_string(self : Complex) -> String {
  let sign = if is_negative(self.im) { "-" } else { "+" }
  "(" + format_real(self.re) + sign + format_real(self.im.abs()) + "j)"
}

///|
pub impl @debug.Debug for Complex with fn to_repr(self) {
  @debug.Repr::literal(
    "Complex(" + format_real(self.re) + ", " + format_real(self.im) + ")",
  )
}

///|
/// True when the sign bit is set, which `-0.0` and `-inf` also are; `x < 0.0` misses both.
fn is_negative(value : Double) -> Bool {
  value < 0.0 || (value == 0.0 && 1.0 / value < 0.0)
}

///|
/// True when either part is non-zero.
pub fn Complex::is_nonzero(self : Complex) -> Bool {
  self.re != 0.0 || self.im != 0.0
}
