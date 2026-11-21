// Copyright 2026 Leo Cheng
// SPDX-License-Identifier: Apache-2.0

///|
test "a reduction along each axis of a three-dimensional array" {
  let cube = from_doubles([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2, 2, 2])
  assert_eq(sum(cube, axis=0).shape(), [2, 2])
  assert_eq(sum(cube, axis=0).to_doubles(), [6.0, 8.0, 10.0, 12.0])
  assert_eq(sum(cube, axis=1).to_doubles(), [4.0, 6.0, 12.0, 14.0])
  assert_eq(sum(cube, axis=2).to_doubles(), [3.0, 7.0, 11.0, 15.0])
  assert_eq(sum(cube, axis=-1).to_doubles(), [3.0, 7.0, 11.0, 15.0])
}

///|
test "keepdims leaves the reduced axis with length one" {
  let matrix = from_doubles([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
  assert_eq(sum(matrix, axis=1, keepdims=true).shape(), [2, 1])
  assert_eq(sum(matrix, axis=0, keepdims=true).shape(), [1, 3])
  assert_eq(sum(matrix, keepdims=true).shape(), [1, 1])
  assert_eq(sum(matrix).shape(), [])
}

///|
test "an axis outside the array is refused" {
  let matrix = zeros([2, 3])
  let refused : Result[Unit, VecError] = try {
    ignore(sum(matrix, axis=2))
    Ok(())
  } catch {
    error => Err(error)
  }
  assert_eq(refused is Err(AxisOutOfRange(_)), true)
}

///|
test "summing nothing gives the identity, but the smallest of nothing is refused" {
  let empty = zeros([0, 3])
  assert_eq(sum(empty).item(), Real(0.0))
  assert_eq(prod(empty).item(), Real(1.0))
  assert_eq(all(empty).item(), Boolean(true))
  assert_eq(any(empty).item(), Boolean(false))
  assert_eq(sum(empty, axis=0).to_doubles(), [0.0, 0.0, 0.0])
  let refused : Result[Unit, VecError] = try {
    ignore(min(empty, axis=0))
    Ok(())
  } catch {
    error => Err(error)
  }
  assert_eq(refused is Err(EmptyAxis(_)), true)
}

///|
test "a sum of booleans counts them" {
  let flags = from_numbers([Boolean(true), Boolean(true), Boolean(false)], [3])
  assert_eq(sum(flags).item(), Signed(2L))
  assert_eq(sum(flags).dtype(), I64)
  assert_eq(any(flags).item(), Boolean(true))
  assert_eq(all(flags).item(), Boolean(false))
}

///|
test "a mean of integers answers in float64 and keeps a float's width" {
  let integers = from_numbers([Signed(1L), Signed(2L)], [2], dtype=I32)
  assert_eq(mean(integers).dtype(), F64)
  assert_eq(mean(integers).item(), Real(1.5))
  let halves = from_doubles([1.0, 2.0], [2], dtype=F32)
  assert_eq(mean(halves).dtype(), F32)
}

///|
test "a variance removes the degrees of freedom asked for" {
  let values = from_doubles([1.0, 2.0, 3.0, 4.0], [4])
  assert_eq(variance(values).item(), Real(1.25))
  assert_eq(variance(values, ddof=1).item(), Real(5.0 / 3.0))
  assert_eq(std(values).item(), Real(1.25.sqrt()))
  let refused : Result[Unit, VecError] = try {
    ignore(variance(values, ddof=4))
    Ok(())
  } catch {
    error => Err(error)
  }
  assert_eq(refused is Err(Domain(_)), true)
}

///|
test "the first of equal extremes wins the index" {
  let values = from_doubles([1.0, 3.0, 3.0, 0.0], [4])
  assert_eq(argmax(values).item(), Signed(1L))
  assert_eq(argmin(values).item(), Signed(3L))
  let matrix = from_doubles([1.0, 5.0, 5.0, 2.0], [2, 2])
  assert_eq(argmax(matrix, axis=1).to_numbers(), [Signed(1L), Signed(0L)])
}

///|
test "a reduction over a transposed view reads the right elements" {
  let matrix = from_doubles([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
  let flipped = transpose(matrix)
  assert_eq(sum(flipped, axis=0).to_doubles(), [6.0, 15.0])
  assert_eq(sum(flipped, axis=1).to_doubles(), [5.0, 7.0, 9.0])
  assert_eq(max(flipped).item(), Real(6.0))
}
