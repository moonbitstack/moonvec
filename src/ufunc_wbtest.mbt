// Copyright 2026 Leo Cheng
// SPDX-License-Identifier: Apache-2.0

///|
test "broadcasting aligns the shapes from the right" {
  assert_eq(broadcast_shape([2, 3], [3]), [2, 3])
  assert_eq(broadcast_shape([3], [2, 3]), [2, 3])
  assert_eq(broadcast_shape([2, 1], [1, 4]), [2, 4])
  assert_eq(broadcast_shape([2, 1, 3], [4, 3]), [2, 4, 3])
  assert_eq(broadcast_shape([], [2, 2]), [2, 2])
  assert_eq(broadcast_shape([0, 3], [3]), [0, 3])
}

///|
test "broadcasting refuses shapes that cannot meet" {
  let refused : Result[Array[Int32], VecError] = Ok(
    broadcast_shape([2, 3], [4]),
  ) catch {
    error => Err(error)
  }
  assert_eq(refused is Err(ShapeMismatch(_)), true)
}

///|
test "a broadcast axis reads the same element whatever its index" {
  let row = from_doubles([10.0, 20.0, 30.0], [1, 3])
  let wide = broadcast_to(row, [4, 3])
  assert_eq(wide.shape(), [4, 3])
  assert_eq(wide.at([3, 1]), Real(20.0))
  let column = from_doubles([1.0, 2.0], [2, 1])
  let tall = broadcast_to(column, [2, 3])
  assert_eq(tall.at([1, 2]), Real(2.0))
}

///|
test "adding a row to a matrix repeats it down the rows" {
  let matrix = from_doubles([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
  let row = from_doubles([10.0, 20.0, 30.0], [3])
  assert_eq(add(matrix, row).to_doubles(), [11.0, 22.0, 33.0, 14.0, 25.0, 36.0])
}

///|
test "adding a column to a matrix repeats it across the columns" {
  let matrix = from_doubles([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
  let column = from_doubles([10.0, 20.0], [2, 1])
  assert_eq(add(matrix, column).to_doubles(), [
    11.0, 12.0, 13.0, 24.0, 25.0, 26.0,
  ])
}

///|
test "two broadcast axes at once" {
  let left = from_doubles([1.0, 2.0, 3.0], [3, 1, 1])
  let right = from_doubles([10.0, 20.0, 30.0, 40.0], [1, 4, 1])
  let product = multiply(left, right)
  assert_eq(product.shape(), [3, 4, 1])
  assert_eq(product.at([2, 3, 0]), Real(120.0))
}

///|
test "an operand read through a view contributes its offset" {
  // A slice starting at element two: an elementwise pass must not read from the buffer's start.
  let source = from_doubles([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [6])
  let tail = slice(source, [span(2, 6)])
  assert_eq(tail.to_doubles(), [3.0, 4.0, 5.0, 6.0])
  assert_eq(multiply(tail, tail).to_doubles(), [9.0, 16.0, 25.0, 36.0])
  assert_eq(sum(tail).item(), Real(18.0))
}

///|
test "an operand read through a transposed view contributes its strides" {
  let source = from_doubles([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
  let flipped = transpose(source)
  assert_eq(add(flipped, flipped).to_doubles(), [2.0, 8.0, 4.0, 10.0, 6.0, 12.0])
}

///|
test "an empty array takes part in an elementwise operation without reading anything" {
  let empty = zeros([0, 3])
  let row = arange(start=0.0, stop=3.0)
  let sum_of_empty = add(empty, row)
  assert_eq(sum_of_empty.shape(), [0, 3])
  assert_eq(sum_of_empty.size(), 0)
}

///|
test "the result dtype follows promotion before any element is read" {
  let integers = from_numbers([Signed(1L), Signed(2L)], [2], dtype=I8)
  let unsigneds = from_numbers([Unsigned(3UL), Unsigned(4UL)], [2], dtype=U8)
  assert_eq(add(integers, unsigneds).dtype(), I16)
  assert_eq(divide(integers, integers).dtype(), F64)
  assert_eq(less(integers, unsigneds).dtype(), Bool)
}

///|
test "arithmetic on integers wraps at the width of the dtype" {
  let top = from_numbers([Signed(120L)], [1], dtype=I8)
  let sum_of_top = add(top, top)
  assert_eq(sum_of_top.dtype(), I8)
  assert_eq(sum_of_top.at([0]), Signed(-16L))
}

///|
test "an integer division by zero is refused" {
  let ones = from_numbers([Signed(1L)], [1], dtype=I32)
  let zeros_array = zeros([1], dtype=I32)
  let refused : Result[Unit, VecError] = try {
    ignore(floor_divide(ones, zeros_array))
    Ok(())
  } catch {
    error => Err(error)
  }
  assert_eq(refused is Err(Domain(_)), true)
}

///|
test "the operator forms agree with the named functions" {
  let left = from_doubles([1.0, 2.0], [2])
  let right = from_doubles([3.0, 4.0], [2])
  assert_eq(left + right, add(left, right))
  assert_eq(left - right, subtract(left, right))
  assert_eq(left * right, multiply(left, right))
  assert_eq(left / right, divide(left, right))
  assert_eq(-left, negative(left))
}

///|
test "a scalar keeps the array's dtype unless the operation is true division" {
  let counts = from_numbers([Signed(1L), Signed(2L)], [2], dtype=I32)
  assert_eq(add_scalar(counts, 3.0).dtype(), I32)
  assert_eq(add_scalar(counts, 3.0).to_numbers(), [Signed(4L), Signed(5L)])
  assert_eq(divide_scalar(counts, 2.0).dtype(), F64)
  assert_eq(divide_scalar(counts, 2.0).to_numbers(), [Real(0.5), Real(1.0)])
}
