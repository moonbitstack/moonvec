// Copyright 2026 Leo Cheng
// SPDX-License-Identifier: Apache-2.0

///|
/// The binary operations a backend has to provide. They are named here rather than passed as
/// closures so that a backend that cannot call back into MoonBit — a shader, say — can still run
/// them.
pub(all) enum BinaryOp {
  Add
  Subtract
  Multiply
  Divide
  FloorDivide
  Remainder
  Power
  Maximum
  Minimum
  ArcTangent2
  Hypotenuse
  BitAnd
  BitOr
  BitXor
  LeftShift
  RightShift
  Equal
  NotEqual
  Less
  LessEqual
  Greater
  GreaterEqual
  LogicalAnd
  LogicalOr
  LogicalXor
}

///|
/// The unary operations a backend has to provide.
pub(all) enum UnaryOp {
  Negate
  Absolute
  Square
  Reciprocal
  Sign
  Sqrt
  Cbrt
  Exp
  Exp2
  Expm1
  Log
  Log2
  Log10
  Log1p
  Sin
  Cos
  Tan
  ArcSin
  ArcCos
  ArcTan
  Sinh
  Cosh
  Tanh
  ArcSinh
  ArcCosh
  ArcTanh
  Floor
  Ceil
  Trunc
  Rint
  Degrees
  Radians
  LogicalNot
  Conjugate
}

///|
/// The reductions a backend has to provide, as an associative pair of values.
pub(all) enum ReduceOp {
  Sum
  Product
  Minimum
  Maximum
  Any
  All
}

///|
/// Applies a binary operation to two values already in `dtype`.
///
/// The result may have a wider class than `dtype` — a comparison gives a boolean, absolute value of
/// a complex gives a real — and the caller's output buffer decides what it is stored as.
pub fn apply_binary(
  op : BinaryOp,
  dtype : DType,
  left : Number,
  right : Number,
) -> Number raise VecError {
  let left = left.cast(dtype)
  let right = right.cast(dtype)
  match (left, right) {
    (Signed(a), Signed(b)) => apply_binary_signed(op, a, b)
    (Unsigned(a), Unsigned(b)) => apply_binary_unsigned(op, a, b)
    (Real(a), Real(b)) => apply_binary_real(op, a, b)
    (Complex(a), Complex(b)) => apply_binary_complex(op, a, b)
    (Boolean(a), Boolean(b)) => apply_binary_bool(op, a, b)
    _ => apply_binary_real(op, left.to_double(), right.to_double())
  }
}

///|
fn apply_binary_signed(
  op : BinaryOp,
  a : Int64,
  b : Int64,
) -> Number raise VecError {
  match op {
    Add => Signed(a + b)
    Subtract => Signed(a - b)
    Multiply => Signed(a * b)
    Divide => Real(a.to_double() / b.to_double())
    FloorDivide => {
      if b == 0L {
        raise Domain(message="integer division by zero")
      }
      Signed(floor_div_int(a, b))
    }
    Remainder => {
      if b == 0L {
        raise Domain(message="integer remainder by zero")
      }
      Signed(floor_mod_int(a, b))
    }
    Power => Signed(int_power(a, b))
    Maximum => Signed(if a > b { a } else { b })
    Minimum => Signed(if a < b { a } else { b })
    ArcTangent2 => Real(@math.atan2(a.to_double(), b.to_double()))
    Hypotenuse => Real(@math.hypot(a.to_double(), b.to_double()))
    BitAnd => Signed(a & b)
    BitOr => Signed(a | b)
    BitXor => Signed(a ^ b)
    LeftShift => Signed(shift_left_int(a, b))
    RightShift => Signed(shift_right_int(a, b))
    Equal => Boolean(a == b)
    NotEqual => Boolean(a != b)
    Less => Boolean(a < b)
    LessEqual => Boolean(a <= b)
    Greater => Boolean(a > b)
    GreaterEqual => Boolean(a >= b)
    LogicalAnd => Boolean(a != 0L && b != 0L)
    LogicalOr => Boolean(a != 0L || b != 0L)
    LogicalXor => Boolean((a != 0L) != (b != 0L))
  }
}

///|
fn apply_binary_unsigned(
  op : BinaryOp,
  a : UInt64,
  b : UInt64,
) -> Number raise VecError {
  match op {
    Add => Unsigned(a + b)
    Subtract => Unsigned(a - b)
    Multiply => Unsigned(a * b)
    Divide => Real(a.to_double() / b.to_double())
    FloorDivide => {
      if b == 0UL {
        raise Domain(message="integer division by zero")
      }
      Unsigned(a / b)
    }
    Remainder => {
      if b == 0UL {
        raise Domain(message="integer remainder by zero")
      }
      Unsigned(a % b)
    }
    Power => Unsigned(uint_power(a, b))
    Maximum => Unsigned(if a > b { a } else { b })
    Minimum => Unsigned(if a < b { a } else { b })
    ArcTangent2 => Real(@math.atan2(a.to_double(), b.to_double()))
    Hypotenuse => Real(@math.hypot(a.to_double(), b.to_double()))
    BitAnd => Unsigned(a & b)
    BitOr => Unsigned(a | b)
    BitXor => Unsigned(a ^ b)
    LeftShift => Unsigned(shift_left_uint(a, b))
    RightShift => Unsigned(shift_right_uint(a, b))
    Equal => Boolean(a == b)
    NotEqual => Boolean(a != b)
    Less => Boolean(a < b)
    LessEqual => Boolean(a <= b)
    Greater => Boolean(a > b)
    GreaterEqual => Boolean(a >= b)
    LogicalAnd => Boolean(a != 0UL && b != 0UL)
    LogicalOr => Boolean(a != 0UL || b != 0UL)
    LogicalXor => Boolean((a != 0UL) != (b != 0UL))
  }
}

///|
fn apply_binary_real(
  op : BinaryOp,
  a : Double,
  b : Double,
) -> Number raise VecError {
  match op {
    Add => Real(a + b)
    Subtract => Real(a - b)
    Multiply => Real(a * b)
    Divide => Real(a / b)
    FloorDivide => Real((a / b).floor())
    Remainder => Real(float_mod(a, b))
    Power => Real(@math.pow(a, b))
    Maximum =>
      Real(
        if a.is_nan() || b.is_nan() {
          @double.not_a_number
        } else if a > b {
          a
        } else {
          b
        },
      )
    Minimum =>
      Real(
        if a.is_nan() || b.is_nan() {
          @double.not_a_number
        } else if a < b {
          a
        } else {
          b
        },
      )
    ArcTangent2 => Real(@math.atan2(a, b))
    Hypotenuse => Real(@math.hypot(a, b))
    BitAnd | BitOr | BitXor | LeftShift | RightShift =>
      raise Domain(message="bitwise operations need integer operands")
    Equal => Boolean(a == b)
    NotEqual => Boolean(a != b)
    Less => Boolean(a < b)
    LessEqual => Boolean(a <= b)
    Greater => Boolean(a > b)
    GreaterEqual => Boolean(a >= b)
    LogicalAnd => Boolean(a != 0.0 && b != 0.0)
    LogicalOr => Boolean(a != 0.0 || b != 0.0)
    LogicalXor => Boolean((a != 0.0) != (b != 0.0))
  }
}

///|
fn apply_binary_complex(
  op : BinaryOp,
  a : Complex,
  b : Complex,
) -> Number raise VecError {
  match op {
    Add => Complex(a + b)
    Subtract => Complex(a - b)
    Multiply => Complex(a * b)
    Divide => Complex(a / b)
    Power => Complex(a.pow(b))
    Equal => Boolean(a == b)
    NotEqual => Boolean(a != b)
    LogicalAnd => Boolean(a.is_nonzero() && b.is_nonzero())
    LogicalOr => Boolean(a.is_nonzero() || b.is_nonzero())
    LogicalXor => Boolean(a.is_nonzero() != b.is_nonzero())
    Maximum => Complex(if compare_complex(a, b) >= 0 { a } else { b })
    Minimum => Complex(if compare_complex(a, b) <= 0 { a } else { b })
    Less => Boolean(compare_complex(a, b) < 0)
    LessEqual => Boolean(compare_complex(a, b) <= 0)
    Greater => Boolean(compare_complex(a, b) > 0)
    GreaterEqual => Boolean(compare_complex(a, b) >= 0)
    _ => raise Domain(message="the operation has no complex form")
  }
}

///|
fn apply_binary_bool(
  op : BinaryOp,
  a : Bool,
  b : Bool,
) -> Number raise VecError {
  match op {
    Add | LogicalOr | Maximum | BitOr => Boolean(a || b)
    Multiply | LogicalAnd | Minimum | BitAnd => Boolean(a && b)
    LogicalXor => Boolean(a != b)
    Subtract =>
      raise Domain(message="subtracting booleans has no boolean answer")
    Divide => Real(if b { 1.0 } else { 0.0 })
    FloorDivide => {
      if !b {
        raise Domain(message="integer division by zero")
      }
      Boolean(a)
    }
    Remainder => {
      if !b {
        raise Domain(message="integer remainder by zero")
      }
      Boolean(false)
    }
    Power => Boolean(a || !b)
    BitXor => Boolean(a != b)
    LeftShift => Boolean(if b { false } else { a })
    RightShift => Boolean(a)
    Equal => Boolean(a == b)
    NotEqual => Boolean(a != b)
    Less => Boolean(!a && b)
    LessEqual => Boolean(!a || b)
    Greater => Boolean(a && !b)
    GreaterEqual => Boolean(a || !b)
    ArcTangent2 | Hypotenuse => Real(0.0)
  }
}

///|
/// Applies a unary operation to a value already cast to `dtype`.
pub fn apply_unary(
  op : UnaryOp,
  dtype : DType,
  value : Number,
) -> Number raise VecError {
  let value = value.cast(dtype)
  match value {
    Signed(a) => apply_unary_signed(op, a)
    Unsigned(a) => apply_unary_unsigned(op, a)
    Real(a) => apply_unary_real(op, a)
    Complex(a) => apply_unary_complex(op, a)
    Boolean(a) => apply_unary_bool(op, a)
  }
}

///|
fn apply_unary_signed(op : UnaryOp, a : Int64) -> Number {
  match op {
    Negate => Signed(-a)
    Absolute => Signed(if a < 0L { -a } else { a })
    Square => Signed(a * a)
    Sign => Signed(if a > 0L { 1L } else if a < 0L { -1L } else { 0L })
    LogicalNot => Boolean(a == 0L)
    Reciprocal => Real(1.0 / a.to_double())
    _ => apply_unary_real(op, a.to_double())
  }
}

///|
fn apply_unary_unsigned(op : UnaryOp, a : UInt64) -> Number {
  match op {
    Absolute => Unsigned(a)
    Square => Unsigned(a * a)
    Sign => Unsigned(if a > 0UL { 1UL } else { 0UL })
    LogicalNot => Boolean(a == 0UL)
    Reciprocal => Real(1.0 / a.to_double())
    _ => apply_unary_real(op, a.to_double())
  }
}

///|
fn apply_unary_real(op : UnaryOp, a : Double) -> Number {
  match op {
    Negate => Real(-a)
    Absolute => Real(a.abs())
    Square => Real(a * a)
    Reciprocal => Real(1.0 / a)
    Sign =>
      Real(
        if a.is_nan() {
          @double.not_a_number
        } else if a > 0.0 {
          1.0
        } else if a < 0.0 {
          -1.0
        } else {
          0.0
        },
      )
    Sqrt => Real(a.sqrt())
    Cbrt => Real(@math.cbrt(a))
    Exp => Real(@math.exp(a))
    Exp2 => Real(@math.pow(2.0, a))
    Expm1 => Real(@math.expm1(a))
    Log => Real(@math.ln(a))
    Log2 => Real(@math.log2(a))
    Log10 => Real(@math.log10(a))
    Log1p => Real(@math.ln_1p(a))
    Sin => Real(@math.sin(a))
    Cos => Real(@math.cos(a))
    Tan => Real(@math.tan(a))
    ArcSin => Real(@math.asin(a))
    ArcCos => Real(@math.acos(a))
    ArcTan => Real(@math.atan(a))
    Sinh => Real(@math.sinh(a))
    Cosh => Real(@math.cosh(a))
    Tanh => Real(@math.tanh(a))
    ArcSinh => Real(@math.asinh(a))
    ArcCosh => Real(@math.acosh(a))
    ArcTanh => Real(@math.atanh(a))
    Floor => Real(a.floor())
    Ceil => Real(a.ceil())
    Trunc => Real(a.trunc())
    Rint => Real(round_half_even(a))
    Degrees => Real(a * 57.29577951308232)
    Radians => Real(a * 0.017453292519943295)
    LogicalNot => Boolean(a == 0.0)
    Conjugate => Real(a)
  }
}

///|
fn apply_unary_complex(op : UnaryOp, a : Complex) -> Number raise VecError {
  match op {
    Negate => Complex(-a)
    Absolute => Real(a.abs())
    Square => Complex(a * a)
    Reciprocal => complex_number(1.0 / a.re, 0.0)
    Sqrt => Complex(a.sqrt())
    Exp => Complex(a.exp())
    Log => Complex(a.log())
    Sin => Complex(a.sin())
    Cos => Complex(a.cos())
    Tan => Complex(a.tan())
    Sinh => Complex(a.sinh())
    Cosh => Complex(a.cosh())
    Tanh => Complex(a.tanh())
    Conjugate => Complex(a.conj())
    LogicalNot => Boolean(!a.is_nonzero())
    Sign => Complex(a / Complex::new(a.abs(), 0.0))
    _ => raise Domain(message="the operation needs a real operand")
  }
}

///|
fn apply_unary_bool(op : UnaryOp, a : Bool) -> Number {
  match op {
    LogicalNot => Boolean(!a)
    Negate => Signed(if a { -1L } else { 0L })
    Absolute | Conjugate => Boolean(a)
    _ => apply_unary_real(op, if a { 1.0 } else { 0.0 })
  }
}

///|
/// Combines two values in a reduction; the pair must be associative and have the identity the
/// caller started from.
pub fn apply_reduce(
  op : ReduceOp,
  dtype : DType,
  left : Number,
  right : Number,
) -> Number raise VecError {
  let left = left.cast(dtype)
  let right = right.cast(dtype)
  match op {
    Sum => apply_binary(Add, dtype, left, right)
    Product => apply_binary(Multiply, dtype, left, right)
    Minimum => apply_binary(Minimum, dtype, left, right)
    Maximum => apply_binary(Maximum, dtype, left, right)
    Any => Boolean(left.to_bool() || right.to_bool())
    All => Boolean(left.to_bool() && right.to_bool())
  }
}

///|
/// The value a reduction starts from: the identity that leaves every element unchanged.
///
/// The bounds are written per width rather than by casting one wide value, because a cast would wrap:
/// the largest `int8` is 127, not `int64`'s maximum truncated to eight bits.
pub fn reduce_identity(op : ReduceOp, dtype : DType) -> Number {
  match op {
    Sum => Number::cast(Signed(0L), dtype)
    Product => Number::cast(Signed(1L), dtype)
    Minimum =>
      match dtype {
        Bool => Boolean(true)
        I8 => Signed(127L)
        I16 => Signed(32767L)
        I32 => Signed(2147483647L)
        I64 => Signed(0x7FFFFFFFFFFFFFFFL)
        U8 => Unsigned(255UL)
        U16 => Unsigned(65535UL)
        U32 => Unsigned(4294967295UL)
        U64 => Unsigned(0xFFFFFFFFFFFFFFFFUL)
        F16 | F32 | F64 => Real(@double.infinity)
        C64 | C128 => complex_number(@double.infinity, @double.infinity)
      }
    Maximum =>
      match dtype {
        Bool => Boolean(false)
        I8 => Signed(-128L)
        I16 => Signed(-32768L)
        I32 => Signed(-2147483648L)
        I64 => Signed(-0x7FFFFFFFFFFFFFFFL - 1L)
        U8 | U16 | U32 | U64 => Unsigned(0UL)
        F16 | F32 | F64 => Real(@double.neg_infinity)
        C64 | C128 => complex_number(@double.neg_infinity, @double.neg_infinity)
      }
    Any => Boolean(false)
    All => Boolean(true)
  }
}

///|
/// Orders two complex values by real part, then by imaginary part.
fn compare_complex(a : Complex, b : Complex) -> Int32 {
  let by_real = compare_real(a.re, b.re)
  if by_real != 0 {
    by_real
  } else {
    compare_real(a.im, b.im)
  }
}

///|
/// Division rounded towards negative infinity, which is what `//` means for integers.
fn floor_div_int(a : Int64, b : Int64) -> Int64 {
  let quotient = a / b
  if a % b != 0L && (a < 0L) != (b < 0L) {
    quotient - 1L
  } else {
    quotient
  }
}

///|
/// The remainder that pairs with `floor_div_int`: it carries the sign of the divisor.
fn floor_mod_int(a : Int64, b : Int64) -> Int64 {
  let rest = a % b
  if rest != 0L && (rest < 0L) != (b < 0L) {
    rest + b
  } else {
    rest
  }
}

///|
/// The remainder with the sign of the divisor.
fn float_mod(a : Double, b : Double) -> Double {
  if b == 0.0 || a.is_nan() || b.is_nan() || a.is_inf() {
    @double.not_a_number
  } else {
    let rest = a - (a / b).floor() * b
    if rest == 0.0 {
      0.0 * b
    } else {
      rest
    }
  }
}

///|
fn int_power(base : Int64, exponent : Int64) -> Int64 raise VecError {
  if exponent < 0L {
    raise Domain(
      message="an integer power with a negative exponent has no integer result",
    )
  }
  let mut result = 1L
  let mut factor = base
  let mut left = exponent
  while left > 0L {
    if (left & 1L) == 1L {
      result = result * factor
    }
    factor = factor * factor
    left = left >> 1
  }
  result
}

///|
fn uint_power(base : UInt64, exponent : UInt64) -> UInt64 {
  let mut result = 1UL
  let mut factor = base
  let mut left = exponent
  while left > 0UL {
    if (left & 1UL) == 1UL {
      result = result * factor
    }
    factor = factor * factor
    left = left >> 1
  }
  result
}

///|
/// Shifts left, refusing a negative count; a count past the width leaves zero.
fn shift_left_int(value : Int64, count : Int64) -> Int64 raise VecError {
  if count < 0L {
    raise Domain(message="a shift count cannot be negative")
  } else if count >= 64L {
    0L
  } else {
    value << count.to_int()
  }
}

///|
fn shift_right_int(value : Int64, count : Int64) -> Int64 raise VecError {
  if count < 0L {
    raise Domain(message="a shift count cannot be negative")
  } else if count >= 64L {
    if value < 0L {
      -1L
    } else {
      0L
    }
  } else {
    value >> count.to_int()
  }
}

///|
fn shift_left_uint(value : UInt64, count : UInt64) -> UInt64 {
  if count >= 64UL {
    0UL
  } else {
    value << count.to_int()
  }
}

///|
fn shift_right_uint(value : UInt64, count : UInt64) -> UInt64 {
  if count >= 64UL {
    0UL
  } else {
    value >> count.to_int()
  }
}

///|
/// Rounds to the nearest integer, with a tie going to the even one.
fn round_half_even(value : Double) -> Double {
  if value.is_nan() || value.is_inf() {
    return value
  }
  let floor = value.floor()
  let rest = value - floor
  if rest < 0.5 {
    floor
  } else if rest > 0.5 {
    floor + 1.0
  } else if (floor / 2.0).floor() * 2.0 == floor {
    floor
  } else {
    floor + 1.0
  }
}
