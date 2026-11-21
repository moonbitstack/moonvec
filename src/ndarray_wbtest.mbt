// Copyright 2026 Leo Cheng
// SPDX-License-Identifier: Apache-2.0

///|
test "a zero-filled array has the shape it was asked for" {
  let a = zeros([2, 3])
  assert_eq(a.shape(), [2, 3])
  assert_eq(a.ndim(), 2)
  assert_eq(a.size(), 6)
  assert_eq(a.dtype(), F64)
  assert_eq(a.strides(), [3, 1])
  assert_eq(a.is_contiguous(), true)
  assert_eq(a.to_doubles(), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
}

///|
test "an empty shape is one element, not zero" {
  let a = zeros([])
  assert_eq(a.ndim(), 0)
  assert_eq(a.size(), 1)
  assert_eq(a.item(), Real(0.0))
}

///|
test "a zero-length axis has no elements" {
  let a = zeros([0, 3])
  assert_eq(a.size(), 0)
  assert_eq(a.to_numbers(), [])
}

///|
test "elements are read and written by index" {
  let a = zeros([2, 2], dtype=I32)
  a.set_at([0, 1], Signed(7L))
  assert_eq(a.at([0, 1]), Signed(7L))
  assert_eq(a.at([1, 1]), Signed(0L))
}

///|
/// True when the body raises a `VecError`, which is how a test checks a refusal.
fn refused(body : () -> Unit raise VecError) -> Bool {
  let outcome : Result[Unit, VecError] = try {
    body()
    Ok(())
  } catch {
    error => Err(error)
  }
  outcome is Err(_)
}

///|
test "an index outside the shape is refused" {
  let a = zeros([2, 2])
  assert_eq(refused(fn() raise VecError { ignore(a.at([2, 0])) }), true)
  assert_eq(refused(fn() raise VecError { ignore(a.at([0, 0, 0])) }), true)
}

///|
test "arange counts the steps the way NumPy does" {
  assert_eq(arange(start=0.0, stop=3.0).to_doubles(), [0.0, 1.0, 2.0])
  let stepped = arange(start=0.0, stop=1.0, step=0.3)
  assert_eq(stepped.size(), 4)
  assert_eq(stepped.at([1]), Real(0.3))
  assert_eq(stepped.at([2]), Real(0.6))
  assert_eq(arange(start=3.0, stop=0.0, step=-1.0).to_doubles(), [3.0, 2.0, 1.0])
  assert_eq(arange(start=0.0, stop=1.0, step=-1.0).size(), 0)
}

///|
test "linspace includes the endpoint unless told not to" {
  assert_eq(linspace(start=0.0, stop=1.0, num=5).to_doubles(), [
    0.0, 0.25, 0.5, 0.75, 1.0,
  ])
  assert_eq(linspace(start=0.0, stop=1.0, num=4, endpoint=false).to_doubles(), [
    0.0, 0.25, 0.5, 0.75,
  ])
  assert_eq(linspace(start=2.0, stop=5.0, num=1).to_doubles(), [2.0])
  assert_eq(linspace(start=0.0, stop=1.0, num=0).size(), 0)
}

///|
test "eye puts ones on the chosen diagonal" {
  assert_eq(eye(n=3).to_doubles(), [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
  assert_eq(eye(n=2, k=1).to_doubles(), [0.0, 1.0, 0.0, 0.0])
  assert_eq(eye(n=2, k=-1).to_doubles(), [0.0, 0.0, 1.0, 0.0])
  assert_eq(eye(n=2, m=3).shape(), [2, 3])
}

///|
test "full and its like-shaped form" {
  let a = full([2], Real(7.5), dtype=F32)
  assert_eq(a.dtype(), F32)
  assert_eq(a.to_doubles(), [7.5, 7.5])
  let b = zeros_like(a)
  assert_eq(b.dtype(), F32)
  assert_eq(b.to_doubles(), [0.0, 0.0])
  let c = ones_like(a)
  assert_eq(c.to_doubles(), [1.0, 1.0])
}

///|
test "a copy is independent of its source" {
  let a = from_doubles([1.0, 2.0, 3.0, 4.0], [2, 2])
  let b = a.copy()
  b.set_at([0, 0], Real(9.0))
  assert_eq(a.at([0, 0]), Real(1.0))
  assert_eq(b.at([0, 0]), Real(9.0))
}

///|
test "reshape reads a contiguous array without copying" {
  let a = from_doubles([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
  let b = reshape(a, [3, 2])
  assert_eq(b.shape(), [3, 2])
  assert_eq(b.to_doubles(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  b.set_at([0, 1], Real(20.0))
  assert_eq(a.at([0, 1]), Real(20.0))
}

///|
test "reshape infers one dimension written as -1" {
  let a = zeros([2, 3, 4])
  assert_eq(reshape(a, [-1]).shape(), [24])
  assert_eq(reshape(a, [6, -1]).shape(), [6, 4])
}

///|
test "reshape refuses a size that does not fit" {
  let a = zeros([2, 3])
  assert_eq(refused(fn() raise VecError { ignore(reshape(a, [4, 2])) }), true)
}

///|
test "transpose reverses the axes without copying" {
  let a = from_doubles([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
  let b = transpose(a)
  assert_eq(b.shape(), [3, 2])
  assert_eq(b.to_doubles(), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
  assert_eq(b.is_contiguous(), false)
  b.set_at([0, 1], Real(40.0))
  assert_eq(a.at([1, 0]), Real(40.0))
}

///|
test "transpose takes an explicit axis order" {
  let a = from_doubles([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
  let b = transpose(a, axes=[1, 0])
  assert_eq(b.to_doubles(), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
}

///|
test "swap and move of axes" {
  let a = zeros([2, 3, 4])
  assert_eq(swap_axes(a, 0, 2).shape(), [4, 3, 2])
  assert_eq(move_axis(a, 0, 2).shape(), [3, 4, 2])
  assert_eq(move_axis(a, 2, 0).shape(), [4, 2, 3])
}

///|
test "squeeze and expand_dims" {
  let a = zeros([1, 3, 1])
  assert_eq(squeeze(a).shape(), [3])
  assert_eq(squeeze(a, axis=0).shape(), [3, 1])
  assert_eq(expand_dims(zeros([3]), axis=0).shape(), [1, 3])
  assert_eq(expand_dims(zeros([3]), axis=1).shape(), [3, 1])
  assert_eq(expand_dims(zeros([3]), axis=-1).shape(), [3, 1])
}

///|
test "basic slicing is a view" {
  let a = from_doubles([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
  let row = slice(a, [At(1), All])
  assert_eq(row.shape(), [3])
  assert_eq(row.to_doubles(), [4.0, 5.0, 6.0])
  row.set_at([0], Real(40.0))
  assert_eq(a.at([1, 0]), Real(40.0))
  let column = slice(a, [All, At(2)])
  assert_eq(column.shape(), [2])
  assert_eq(column.to_doubles(), [3.0, 6.0])
}

///|
test "a span clips its bounds and takes a step" {
  let a = arange(start=0.0, stop=10.0)
  assert_eq(slice(a, [span(2, 5)]).to_doubles(), [2.0, 3.0, 4.0])
  assert_eq(slice(a, [span(0, 10, step=3)]).to_doubles(), [0.0, 3.0, 6.0, 9.0])
  assert_eq(slice(a, [span(0, 100)]).size(), 10)
  assert_eq(slice(a, [span(8, 2, step=-2)]).to_doubles(), [8.0, 6.0, 4.0])
  assert_eq(slice(a, [span(-3, -1)]).to_doubles(), [7.0, 8.0])
}

///|
test "NewAxis adds an axis to a view" {
  let a = arange(start=0.0, stop=3.0)
  let b = slice(a, [NewAxis, All])
  assert_eq(b.shape(), [1, 3])
  let c = slice(a, [All, NewAxis])
  assert_eq(c.shape(), [3, 1])
}

///|
test "slicing past the axes takes the rest whole" {
  let a = zeros([2, 3, 4])
  assert_eq(slice(a, [At(0)]).shape(), [3, 4])
  assert_eq(slice(a, []).shape(), [2, 3, 4])
}

///|
test "integer array indexing picks positions and copies" {
  let a = arange(start=0.0, stop=6.0)
  let picked = take(a, [4, 0, 2])
  assert_eq(picked.to_doubles(), [4.0, 0.0, 2.0])
  picked.set_at([0], Real(99.0))
  assert_eq(a.at([4]), Real(4.0))
}

///|
test "integer array indexing along an axis keeps the rank" {
  let a = from_doubles([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
  let picked = take(a, [1, 1, 0], axis=0)
  assert_eq(picked.shape(), [3, 3])
  assert_eq(picked.to_doubles(), [4.0, 5.0, 6.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0])
  let columns = take(a, [2, 0], axis=1)
  assert_eq(columns.shape(), [2, 2])
  assert_eq(columns.to_doubles(), [3.0, 1.0, 6.0, 4.0])
}

///|
test "a boolean mask keeps the positions it marks" {
  let a = arange(start=0.0, stop=6.0)
  let picked = mask(a, [true, false, true, false, false, true])
  assert_eq(picked.to_doubles(), [0.0, 2.0, 5.0])
}

///|
test "the printed form matches NumPy's layout" {
  let a = from_doubles([1.0, 2.0, 3.0, 4.0], [2, 2])
  assert_eq(a.to_string(), "array([[1.0, 2.0],\n [3.0, 4.0]], dtype=float64)")
  let b = from_numbers([Signed(1L), Signed(2L)], [2], dtype=I32)
  assert_eq(b.to_string(), "array([1, 2], dtype=int32)")
}
