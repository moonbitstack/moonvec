// Copyright 2026 Leo Cheng
// SPDX-License-Identifier: Apache-2.0

///|
test "a small array prints every element" {
  let matrix = from_doubles([1.0, 2.0, 3.0, 4.0], [2, 2])
  assert_eq(
    matrix.to_string(),
    "array([[1.0, 2.0],\n [3.0, 4.0]], dtype=float64)",
  )
}

///|
test "a scalar prints as one value without brackets" {
  let value = from_doubles([2.5], [])
  assert_eq(value.to_string(), "array(2.5, dtype=float64)")
}

///|
test "an empty array prints its dtype and nothing else" {
  assert_eq(zeros([0]).to_string(), "array([], dtype=float64)")
  assert_eq(zeros([0, 3]).to_string(), "array([], dtype=float64)")
}

///|
test "integers print without a decimal point and floats with one" {
  assert_eq(
    from_numbers([Signed(3L)], [1]).to_string(),
    "array([3], dtype=int64)",
  )
  assert_eq(from_doubles([3.0], [1]).to_string(), "array([3.0], dtype=float64)")
  assert_eq(
    from_numbers([Unsigned(7UL)], [1], dtype=U8).to_string(),
    "array([7], dtype=uint8)",
  )
  assert_eq(
    from_numbers([Boolean(true)], [1]).to_string(),
    "array([true], dtype=bool)",
  )
}

///|
test "a complex value prints like NumPy prints one" {
  let value = from_numbers([complex_number(1.0, -2.0)], [1], dtype=C128)
  assert_eq(value.to_string(), "array([(1.0-2.0j)], dtype=complex128)")
}

///|
test "a large array elides the middle of each axis" {
  let wide = arange(start=0.0, stop=1500.0)
  let text = wide.to_string()
  assert_eq(text.contains("..."), true)
  assert_eq(text.contains("1499.0"), true)
  let grid = zeros([40, 40])
  let text = grid.to_string()
  assert_eq(text.contains("..."), true)
  assert_eq(text.contains("dtype=float64"), true)
}

///|
test "NaN and infinity are spelled out" {
  let values = from_doubles(
    [@double.not_a_number, @double.infinity, @double.neg_infinity],
    [3],
  )
  assert_eq(values.to_string(), "array([nan, inf, -inf], dtype=float64)")
}

///|
test "the debug form is the printed form, so a failing test reads like an array" {
  let value = from_doubles([1.0], [1])
  assert_eq(@debug.to_string(value), value.to_string())
}
