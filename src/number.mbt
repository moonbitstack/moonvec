// Copyright 2026 Leo Cheng
// SPDX-License-Identifier: Apache-2.0

///|
/// One element of an array, widened to the type that holds its whole range.
///
/// The variants name the class the value belongs to rather than one variant per dtype: reading an
/// `int16` gives `Signed` too, and writing it back checks the width. This keeps a kernel written
/// once per operation instead of once per dtype.
pub(all) enum Number {
  Boolean(Bool)
  Signed(Int64)
  Unsigned(UInt64)
  Real(Double)
  Complex(Complex)
}

///|
/// A complex value as a `Number`, spelling out both types so the variant and the struct do not
/// shadow each other.
pub fn complex_number(re : Double, im : Double) -> Number {
  Complex(Complex::new(re, im))
}

///|
pub fn Number::to_double(self : Number) -> Double {
  match self {
    Boolean(b) => if b { 1.0 } else { 0.0 }
    Signed(v) => v.to_double()
    Unsigned(v) => v.to_double()
    Real(v) => v
    Complex(v) => v.re
  }
}

///|
/// The value truncated toward zero, the rule C uses when a float is stored into an integer.
pub fn Number::to_int64(self : Number) -> Int64 {
  match self {
    Boolean(b) => if b { 1L } else { 0L }
    Signed(v) => v
    Unsigned(v) => v.reinterpret_as_int64()
    Real(v) => real_to_int64(v)
    Complex(v) => real_to_int64(v.re)
  }
}

///|
pub fn Number::to_bool(self : Number) -> Bool {
  match self {
    Boolean(b) => b
    Signed(v) => v != 0L
    Unsigned(v) => v != 0UL
    Real(v) => v != 0.0
    Complex(v) => v.re != 0.0 || v.im != 0.0
  }
}

///|
pub fn Number::to_complex(self : Number) -> Complex {
  match self {
    Boolean(b) => { re: if b { 1.0 } else { 0.0 }, im: 0.0, }
    Signed(v) => { re: v.to_double(), im: 0.0, }
    Unsigned(v) => { re: v.to_double(), im: 0.0, }
    Real(v) => { re: v, im: 0.0, }
    Complex(v) => v
  }
}

///|
fn real_to_int64(value : Double) -> Int64 {
  if value.is_nan() {
    0L
  } else if value >= 9.223372036854775e18 {
    9223372036854775807L
  } else if value <= -9.223372036854775e18 {
    -9223372036854775808L
  } else {
    value.to_int64()
  }
}

///|
/// The same value, stored as `dtype` would store it: integers wrap to their width, floats round.
pub fn Number::cast(self : Number, dtype : DType) -> Number {
  match dtype {
    Bool => Boolean(self.to_bool())
    I8 => Signed(wrap_int64(self.to_int64(), 8, true))
    I16 => Signed(wrap_int64(self.to_int64(), 16, true))
    I32 => Signed(wrap_int64(self.to_int64(), 32, true))
    I64 => Signed(self.to_int64())
    U8 => Unsigned(wrap_uint64(self.to_int64(), 8))
    U16 => Unsigned(wrap_uint64(self.to_int64(), 16))
    U32 => Unsigned(wrap_uint64(self.to_int64(), 32))
    U64 => Unsigned(self.to_int64().reinterpret_as_uint64())
    F16 => Real(round_to_half(self.to_double()).to_double())
    F32 => Real(Float::from_double(self.to_double()).to_double())
    F64 => Real(self.to_double())
    C64 => {
      let c = self.to_complex()
      Complex({
        re: Float::from_double(c.re).to_double(),
        im: Float::from_double(c.im).to_double(),
      })
    }
    C128 => Complex(self.to_complex())
  }
}

///|
/// The double nearest to the value a half precision float holds, so that a value read back equals
/// what was written.
fn round_to_half(value : Double) -> Float {
  @moonmath.Half::from_float(Float::from_double(value)).to_float()
}

///|
/// Keeps the low `bits` bits of `value` and sign-extends them when `signed`.
fn wrap_int64(value : Int64, bits : Int32, signed : Bool) -> Int64 {
  let mask = if bits >= 64 { 0xFFFFFFFFFFFFFFFFUL } else { (1UL << bits) - 1UL }
  let low = value.reinterpret_as_uint64() & mask
  let sign_bit_set = bits < 64 && ((low >> (bits - 1)) & 1UL) == 1UL
  if signed && sign_bit_set {
    (low | (mask ^ 0xFFFFFFFFFFFFFFFFUL)).reinterpret_as_int64()
  } else {
    low.reinterpret_as_int64()
  }
}

///|
fn wrap_uint64(value : Int64, bits : Int32) -> UInt64 {
  if bits >= 64 {
    value.reinterpret_as_uint64()
  } else {
    value.reinterpret_as_uint64() & ((1UL << bits) - 1UL)
  }
}

///|
/// Compares two values by magnitude only; complex pairs compare by real part then imaginary part,
/// which is the order NumPy sorts them in.
pub fn Number::compare_value(self : Number, other : Number) -> Int32 {
  match (self, other) {
    (Boolean(a), Boolean(b)) => if a == b { 0 } else if a { 1 } else { -1 }
    (Signed(a), Signed(b)) => a.compare(b)
    (Unsigned(a), Unsigned(b)) => a.compare(b)
    (Signed(_), Unsigned(_)) | (Unsigned(_), Signed(_)) => {
      let x = promote(
        if self is Signed(_) {
          I64
        } else {
          U64
        },
        if other is Signed(_) {
          I64
        } else {
          U64
        },
      )
      if x.is_float() {
        self.to_double().compare(other.to_double())
      } else {
        self.to_int64().compare(other.to_int64())
      }
    }
    (Real(a), Real(b)) => compare_real(a, b)
    (Complex(a), Complex(b)) => {
      let by_real = compare_real(a.re, b.re)
      if by_real != 0 {
        by_real
      } else {
        compare_real(a.im, b.im)
      }
    }
    (Complex(a), _) => {
      let other = other.to_complex()
      let by_real = compare_real(a.re, other.re)
      if by_real != 0 {
        by_real
      } else {
        compare_real(a.im, other.im)
      }
    }
    (_, Complex(_)) => -other.compare_value(self)
    _ => compare_real(self.to_double(), other.to_double())
  }
}

///|
/// Orders two doubles with NaN last, so a sort puts the values that cannot be compared at the end
/// instead of scattering them.
pub fn compare_real(left : Double, right : Double) -> Int32 {
  if left < right {
    -1
  } else if left > right {
    1
  } else if left == right {
    0
  } else if left.is_nan() && right.is_nan() {
    0
  } else if left.is_nan() {
    1
  } else {
    -1
  }
}

///|
pub impl Eq for Number with fn equal(self, other) {
  self.compare_value(other) == 0
}

///|
pub impl @debug.Debug for Number with fn to_repr(self) {
  @debug.Repr::literal(
    match self {
      Boolean(b) => if b { "Boolean(true)" } else { "Boolean(false)" }
      Signed(v) => "Signed(" + v.to_string() + ")"
      Unsigned(v) => "Unsigned(" + v.to_string() + ")"
      Real(v) => "Real(" + format_real(v) + ")"
      Complex(v) =>
        "Complex(" + format_real(v.re) + ", " + format_real(v.im) + ")"
    },
  )
}

///|
pub extend Number with Show::{to_string}

///|
pub impl Show for Number with fn to_string(self) {
  match self {
    Boolean(b) => if b { "true" } else { "false" }
    Signed(v) => v.to_string()
    Unsigned(v) => v.to_string()
    Real(v) => format_real(v)
    Complex(v) => v.to_repr_string()
  }
}

///|
/// A double as NumPy writes it: `1.0` rather than `1`, with `nan` and `inf` spelled out.
pub fn format_real(value : Double) -> String {
  if value.is_nan() {
    "nan"
  } else if value.is_inf() {
    if value > 0.0 {
      "inf"
    } else {
      "-inf"
    }
  } else {
    let text = value.to_string()
    if text.contains(".") || text.contains("e") || text.contains("E") {
      text.replace_all(old="e+", new="e")
    } else {
      text + ".0"
    }
  }
}
