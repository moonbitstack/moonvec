// Copyright 2026 Leo Cheng
// SPDX-License-Identifier: Apache-2.0

///|
/// The element type of an array.
///
/// The set follows what NumPy offers: one boolean type, the signed and unsigned integers, the
/// half, single and double precision floats, and the two complex widths. There is no platform
/// integer type, because an array's dtype has to mean the same thing on every target.
pub(all) enum DType {
  Bool
  I8
  I16
  I32
  I64
  U8
  U16
  U32
  U64
  F16
  F32
  F64
  C64
  C128
}

///|
/// The number of bytes one element takes in a buffer.
pub fn DType::item_size(self : DType) -> Int32 {
  match self {
    Bool | I8 | U8 => 1
    I16 | U16 | F16 => 2
    I32 | U32 | F32 => 4
    I64 | U64 | F64 | C64 => 8
    C128 => 16
  }
}

///|
/// The name NumPy gives this type, lower case and without spaces.
pub fn DType::name(self : DType) -> String {
  match self {
    Bool => "bool"
    I8 => "int8"
    I16 => "int16"
    I32 => "int32"
    I64 => "int64"
    U8 => "uint8"
    U16 => "uint16"
    U32 => "uint32"
    U64 => "uint64"
    F16 => "float16"
    F32 => "float32"
    F64 => "float64"
    C64 => "complex64"
    C128 => "complex128"
  }
}

///|
/// Reads a dtype back from its name, as written by `to_string` or a vector file.
pub fn DType::parse(text : StringView) -> DType? {
  match text {
    "bool" => Some(Bool)
    "int8" => Some(I8)
    "int16" => Some(I16)
    "int32" => Some(I32)
    "int64" => Some(I64)
    "uint8" => Some(U8)
    "uint16" => Some(U16)
    "uint32" => Some(U32)
    "uint64" => Some(U64)
    "float16" => Some(F16)
    "float32" => Some(F32)
    "float64" => Some(F64)
    "complex64" => Some(C64)
    "complex128" => Some(C128)
    _ => None
  }
}

///|
pub impl Show for DType with fn to_string(self) {
  self.name()
}

///|
pub impl @debug.Debug for DType with fn to_repr(self) {
  @debug.Repr::literal("DType::" + self.constructor_name())
}

///|
/// The spelling of the enum constructor, for debug output and error messages.
fn DType::constructor_name(self : DType) -> String {
  @debug.Repr::literal(
    match self {
      Bool => "Bool"
      I8 => "I8"
      I16 => "I16"
      I32 => "I32"
      I64 => "I64"
      U8 => "U8"
      U16 => "U16"
      U32 => "U32"
      U64 => "U64"
      F16 => "F16"
      F32 => "F32"
      F64 => "F64"
      C64 => "C64"
      C128 => "C128"
    },
  ).to_string()
}

///|
pub impl Eq for DType with fn equal(self, other) {
  self.index() == other.index()
}

///|
/// The row and column of the dtype in the promotion table.
fn DType::index(self : DType) -> Int32 {
  match self {
    Bool => 0
    I8 => 1
    I16 => 2
    I32 => 3
    I64 => 4
    U8 => 5
    U16 => 6
    U32 => 7
    U64 => 8
    F16 => 9
    F32 => 10
    F64 => 11
    C64 => 12
    C128 => 13
  }
}

///|
/// The dtype the table puts at `(row, column)`; `rows[i]` is the row of dtype number `i`.
///
/// Transcribed from `numpy.result_type` by `scripts/gen_promotion.py`; the test that replays the
/// vector file compares every one of the 196 cells against NumPy's own answer.
let promotion_table : Array[Array[DType]] = [
  [Bool, I8, I16, I32, I64, U8, U16, U32, U64, F16, F32, F64, C64, C128],
  [I8, I8, I16, I32, I64, I16, I32, I64, F64, F16, F32, F64, C64, C128],
  [I16, I16, I16, I32, I64, I16, I32, I64, F64, F32, F32, F64, C64, C128],
  [I32, I32, I32, I32, I64, I32, I32, I64, F64, F64, F64, F64, C128, C128],
  [I64, I64, I64, I64, I64, I64, I64, I64, F64, F64, F64, F64, C128, C128],
  [U8, I16, I16, I32, I64, U8, U16, U32, U64, F16, F32, F64, C64, C128],
  [U16, I32, I32, I32, I64, U16, U16, U32, U64, F32, F32, F64, C64, C128],
  [U32, I64, I64, I64, I64, U32, U32, U32, U64, F64, F64, F64, C128, C128],
  [U64, F64, F64, F64, F64, U64, U64, U64, U64, F64, F64, F64, C128, C128],
  [F16, F16, F32, F64, F64, F16, F32, F64, F64, F16, F32, F64, C64, C128],
  [F32, F32, F32, F64, F64, F32, F32, F64, F64, F32, F32, F64, C64, C128],
  [F64, F64, F64, F64, F64, F64, F64, F64, F64, F64, F64, F64, C128, C128],
  [C64, C64, C64, C128, C128, C64, C64, C128, C128, C64, C64, C128, C64, C128],
  [
    C128,
    C128,
    C128,
    C128,
    C128,
    C128,
    C128,
    C128,
    C128,
    C128,
    C128,
    C128,
    C128,
    C128,
  ],
]

///|
/// The dtype two arrays of these dtypes produce when an arithmetic operation meets them.
///
/// This is not a widening hierarchy: `uint64` with `int8` gives `float64`, because no integer type
/// holds both ranges. The table is symmetric, so the order of the arguments does not matter.
pub fn promote(left : DType, right : DType) -> DType {
  promotion_table[left.index()][right.index()]
}

///|
/// The dtype every one of `dtypes` promotes to; `F64` for an empty list, the neutral element.
pub fn promote_all(dtypes : ArrayView[DType]) -> DType {
  let mut result = F64
  let mut first = true
  for dtype in dtypes {
    if first {
      result = dtype
      first = false
    } else {
      result = promote(result, dtype)
    }
  }
  result
}

///|
pub fn DType::is_bool(self : DType) -> Bool {
  self is Bool
}

///|
/// True for the signed and unsigned integer dtypes, false for `bool`.
pub fn DType::is_integer(self : DType) -> Bool {
  match self {
    I8 | I16 | I32 | I64 | U8 | U16 | U32 | U64 => true
    _ => false
  }
}

///|
pub fn DType::is_signed(self : DType) -> Bool {
  match self {
    I8 | I16 | I32 | I64 => true
    _ => false
  }
}

///|
/// True for `F16`, `F32` and `F64`; the complex dtypes are not floats.
pub fn DType::is_float(self : DType) -> Bool {
  match self {
    F16 | F32 | F64 => true
    _ => false
  }
}

///|
pub fn DType::is_complex(self : DType) -> Bool {
  match self {
    C64 | C128 => true
    _ => false
  }
}

///|
/// True for anything whose values are stored as a real number: floats and complex.
pub fn DType::is_real(self : DType) -> Bool {
  self.is_float() || self.is_complex()
}

///|
/// True for the dtypes an operation on integers keeps exact: integers and `bool`.
pub fn DType::is_exact(self : DType) -> Bool {
  self.is_bool() || self.is_integer()
}

///|
/// The float dtype that holds this one at the same width, for operations that leave the integer
/// domain: `I8` gives `F16`, `I32` gives `F64`. Complex dtypes are returned unchanged.
pub fn DType::to_real(self : DType) -> DType {
  match self {
    Bool | I8 | U8 => F16
    I16 | U16 => F32
    I32 | U32 | I64 | U64 => F64
    F16 | F32 | F64 | C64 | C128 => self
  }
}

///|
/// The dtype the true division of these two dtypes produces: integers turn into `float64`.
pub fn division_dtype(left : DType, right : DType) -> DType {
  let result = promote(left, right)
  if result.is_exact() {
    F64
  } else {
    result
  }
}

///|
/// The dtype that holds every value of `self` without loss, the smallest such float being `F64`.
pub fn DType::float_size(self : DType) -> Int32 {
  match self {
    F16 => 2
    F32 | C64 => 4
    _ => 8
  }
}
