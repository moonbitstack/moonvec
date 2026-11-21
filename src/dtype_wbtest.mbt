// Copyright 2026 Leo Cheng
// SPDX-License-Identifier: Apache-2.0

///|
test "every dtype reports its width and its name" {
  let dtypes : Array[DType] = [
    Bool,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F16,
    F32,
    F64,
    C64,
    C128,
  ]
  let widths = dtypes.map(fn(dtype) { dtype.item_size() })
  assert_eq(widths, [1, 1, 2, 4, 8, 1, 2, 4, 8, 2, 4, 8, 8, 16])
  for dtype in dtypes {
    assert_eq(DType::parse(dtype.name()), Some(dtype))
  }
}

///|
test "the dtype classes are mutually exclusive" {
  assert_eq(Bool.is_bool(), true)
  assert_eq(Bool.is_integer(), false)
  assert_eq(I32.is_integer(), true)
  assert_eq(U32.is_signed(), false)
  assert_eq(I32.is_signed(), true)
  assert_eq(F32.is_float(), true)
  assert_eq(C64.is_float(), false)
  assert_eq(C64.is_complex(), true)
  assert_eq(F64.is_exact(), false)
  assert_eq(I8.is_exact(), true)
}

///|
test "promotion is symmetric and covers the pairs that widen" {
  let dtypes : Array[DType] = [
    Bool,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F16,
    F32,
    F64,
    C64,
    C128,
  ]
  for left in dtypes {
    for right in dtypes {
      assert_eq(promote(left, right), promote(right, left))
    }
  }
  assert_eq(promote(I8, U8), I16)
  assert_eq(promote(I32, U32), I64)
  assert_eq(promote(U64, I8), F64)
  assert_eq(promote(F32, I32), F64)
  assert_eq(promote(C64, F64), C128)
  assert_eq(promote(Bool, Bool), Bool)
}

///|
test "true division of two integers answers in float64" {
  assert_eq(division_dtype(I32, I32), F64)
  assert_eq(division_dtype(F32, F32), F32)
  assert_eq(division_dtype(C64, C64), C64)
  assert_eq(division_dtype(Bool, Bool), F64)
}

///|
test "promote_all folds a list of dtypes" {
  assert_eq(promote_all([I8, U8, I32]), I32)
  assert_eq(promote_all([I8]), I8)
  assert_eq(promote_all([]), F64)
}

///|
test "a cast wraps an integer and truncates a float" {
  assert_eq(Number::cast(Signed(300L), U8), Unsigned(44UL))
  assert_eq(Number::cast(Signed(200L), I8), Signed(-56L))
  assert_eq(Number::cast(Real(2.9), I32), Signed(2L))
  assert_eq(Number::cast(Real(-2.9), I32), Signed(-2L))
  assert_eq(Number::cast(Real(1.5), Bool), Boolean(true))
  assert_eq(Number::cast(Signed(0L), Bool), Boolean(false))
}
