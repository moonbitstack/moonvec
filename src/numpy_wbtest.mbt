// Copyright 2026 Leo Cheng
// SPDX-License-Identifier: Apache-2.0

// Cases transcribed from NumPy's own test suite, in the areas this library covers. Each test names
// the NumPy test it came from so that a reader can compare the two. NumPy is the reference here, and
// where this library deliberately differs the test says so.

///|
/// From `numpy/_core/tests/test_numeric.py`, `TestBroadcast.test_broadcast_in_args`: four shapes that
/// only agree when aligned from the right.
test "numpy: broadcast_in_args" {
  let first = broadcast_shape([6, 7], [5, 6, 1])
  assert_eq(first, [5, 6, 7])
  let second = broadcast_shape(first, [7])
  assert_eq(second, [5, 6, 7])
  let third = broadcast_shape(second, [5, 1, 7])
  assert_eq(third, [5, 6, 7])
}

///|
/// From `test_numeric.py`, `TestTensordot.test_zero_dimension`: a contraction over an axis of length
/// zero gives a block of zeros with the outer shape.
test "numpy: a product over an empty axis" {
  let left = zeros([3, 0])
  let right = zeros([0, 4])
  let product = matmul(left, right)
  assert_eq(product.shape(), [3, 4])
  assert_eq(product.to_doubles(), [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  ])
}

///|
/// From `test_numeric.py`, `TestAsType.test_astype`: the values and the dtype both follow the cast.
test "numpy: astype to an unsigned type" {
  let data = from_numbers(
    [Signed(1L), Signed(2L), Signed(3L), Signed(4L)],
    [2, 2],
    dtype=I64,
  )
  let converted = astype(data, dtype=U32)
  assert_eq(converted.dtype(), U32)
  assert_eq(converted.to_numbers(), [
    Unsigned(1UL),
    Unsigned(2UL),
    Unsigned(3UL),
    Unsigned(4UL),
  ])
}

///|
/// From `test_umath.py`, `TestPower.test_power_zero`: zero to the zeroth power is one, and zero to a
/// positive power is zero.
test "numpy: power of zero" {
  let zero = from_doubles([0.0], [1])
  let zero_exponent = from_doubles([0.0], [1])
  assert_eq(power(zero, zero_exponent).to_doubles(), [1.0])
  assert_eq(power(zero, from_doubles([2.0], [1])).to_doubles(), [0.0])
}

///|
/// From `test_umath.py`, `TestPower.test_zero_power_nonzero`: a complex zero raised to a power whose
/// real part is positive stays zero.
test "numpy: a complex zero raised to a power" {
  let zero = from_numbers([complex_number(0.0, 0.0)], [1], dtype=C128)
  let positive = from_numbers([complex_number(1.0, 4.0)], [1], dtype=C128)
  let answer = power(zero, positive)
  assert_eq(answer.at([0]), complex_number(0.0, 0.0))
  let other = power(
    zero,
    from_numbers([complex_number(2.0, -3.0)], [1], dtype=C128),
  )
  assert_eq(other.at([0]), complex_number(0.0, 0.0))
}

///|
/// From `test_umath.py`, `TestDivision.test_zero_division_complex`: dividing one by a complex zero has
/// an infinite real part.
test "numpy: one divided by a complex zero" {
  let one = from_numbers([complex_number(1.0, 0.0)], [1], dtype=C128)
  let zero = from_numbers([complex_number(0.0, 0.0)], [1], dtype=C128)
  let answer = divide(one, zero).at([0]).to_complex()
  assert_eq(answer.re.is_inf(), true)
  assert_eq(answer.im.is_nan(), true)
}

///|
/// From `test_umath.py`, `TestDivision.test_divide_by_zero`: NumPy answers an integer division by zero
/// with zero and a warning, and its own comment says "the return value cannot be well defined here".
///
/// This library refuses instead: a number that is not a quotient must not look like one. The test is
/// written as the difference it is, so that a reader can find the decision rather than a silent
/// deviation.
test "numpy: integer division by zero, refused here" {
  let numerator = from_numbers([Signed(1L)], [1], dtype=I32)
  let zero = zeros([1], dtype=I32)
  let refused : Result[Unit, VecError] = try {
    ignore(floor_divide(numerator, zero))
    Ok(())
  } catch {
    error => Err(error)
  }
  assert_eq(refused is Err(Domain(_)), true)
}

///|
/// From `test_umath.py`, `TestHypot`: `hypot` avoids the overflow the naive form has.
test "numpy: hypot does not overflow" {
  let large = from_doubles([1.0e200], [1])
  let same = from_doubles([1.0e200], [1])
  let answer = hypot(large, same).at([0]).to_double()
  assert_eq(answer.is_inf(), false)
  assert_eq(answer > 1.4e200, true)
}

///|
/// From `test_umath.py`, `TestMaximum`: a NaN wins on either side, which is the definition NumPy uses.
test "numpy: maximum propagates a NaN" {
  let missing = from_doubles([@double.not_a_number], [1])
  let value = from_doubles([1.0], [1])
  assert_eq(maximum(missing, value).at([0]).to_double().is_nan(), true)
  assert_eq(maximum(value, missing).at([0]).to_double().is_nan(), true)
  assert_eq(minimum(missing, value).at([0]).to_double().is_nan(), true)
}

///|
/// From `test_umath.py`, `TestSign`: the sign of a zero is zero, of a negative number minus one, and
/// of a NaN a NaN.
test "numpy: sign of zero, of a negative number and of a NaN" {
  let values = from_doubles([0.0, -3.5, @double.not_a_number], [3])
  let signs = sign(values)
  assert_eq(signs.at([0]), Real(0.0))
  assert_eq(signs.at([1]), Real(-1.0))
  assert_eq(signs.at([2]).to_double().is_nan(), true)
}

///|
/// From `test_numeric.py`, `TestNonarrayArgs.test_size`: a reduction over every element gives a
/// scalar, and the same reduction with `keepdims` keeps every axis with length one.
test "numpy: a full reduction is a scalar" {
  let values = from_doubles([1.0, 2.0, 3.0], [3])
  assert_eq(sum(values).shape(), [])
  assert_eq(sum(values, keepdims=true).shape(), [1])
  let grid = from_doubles([1.0, 2.0, 3.0, 4.0], [2, 2])
  assert_eq(sum(grid).shape(), [])
  assert_eq(sum(grid, keepdims=true).shape(), [1, 1])
}

///|
/// From `test_numeric.py`, `TestClip`: a clip above the highest value leaves the array alone and a clip
/// below the lowest changes nothing either.
test "numpy: clip at the edges" {
  let values = from_doubles([1.0, 2.0, 3.0], [3])
  assert_eq(clip(values, min=0.0, max=4.0).to_doubles(), [1.0, 2.0, 3.0])
  assert_eq(clip(values, min=2.0, max=4.0).to_doubles(), [2.0, 2.0, 3.0])
}
