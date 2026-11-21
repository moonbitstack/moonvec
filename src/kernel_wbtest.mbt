// Copyright 2026 Leo Cheng
// SPDX-License-Identifier: Apache-2.0

///|
/// A second backend that does the work the slow way and counts the runs it was handed. It exists to
/// show that the array layer really does go through the contract: nothing above `Kernel` names the
/// CPU backend, so a kernel that is wrong shows up as wrong answers rather than as silence.
struct Counting {
  mut runs : Int32
  mut elements : Int32
}

///|
fn Counting::new() -> Counting {
  { runs: 0, elements: 0, }
}

///|
impl Kernel for Counting with fn name(_self) {
  "counting"
}

///|
impl Kernel for Counting with fn binary(
  self,
  op,
  dtype,
  a,
  a_at,
  a_step,
  b,
  b_at,
  b_step,
  out,
  out_at,
  out_step,
  len,
) {
  self.runs = self.runs + 1
  self.elements = self.elements + len
  for i in 0..<len {
    let value = apply_binary(
      op,
      dtype,
      a.read(a_at + i * a_step),
      b.read(b_at + i * b_step),
    )
    out.write(out_at + i * out_step, value)
  }
}

///|
impl Kernel for Counting with fn unary(
  self,
  op,
  dtype,
  a,
  a_at,
  a_step,
  out,
  out_at,
  out_step,
  len,
) {
  self.runs = self.runs + 1
  self.elements = self.elements + len
  for i in 0..<len {
    out.write(
      out_at + i * out_step,
      apply_unary(op, dtype, a.read(a_at + i * a_step)),
    )
  }
}

///|
impl Kernel for Counting with fn matmul(
  self,
  dtype,
  a,
  a_at,
  b,
  b_at,
  out,
  out_at,
  m,
  k,
  n,
) {
  self.runs = self.runs + 1
  self.elements = self.elements + m * n
  for i in 0..<m {
    for j in 0..<n {
      let mut total = reduce_identity(Sum, dtype)
      for inner in 0..<k {
        let product = apply_binary(
          Multiply,
          dtype,
          a.read(a_at + i * k + inner),
          b.read(b_at + inner * n + j),
        )
        total = apply_binary(Add, dtype, total, product)
      }
      out.write(out_at + i * n + j, total)
    }
  }
}

///|
impl Kernel for Counting with fn reduce(
  self,
  op,
  dtype,
  a,
  a_at,
  a_step,
  len,
  into,
) {
  self.runs = self.runs + 1
  self.elements = self.elements + len
  let mut total = into
  for i in 0..<len {
    total = apply_reduce(op, dtype, total, a.read(a_at + i * a_step))
  }
  total
}

///|
test "an operation goes through the backend it is given" {
  let backend = Counting::new()
  let left = from_doubles([1.0, 2.0, 3.0], [3])
  let right = from_doubles([4.0, 5.0, 6.0], [3])
  let total = add(left, right, kernel=backend)
  assert_eq(total.to_doubles(), [5.0, 7.0, 9.0])
  // A contiguous run of three is one call with three elements.
  assert_eq(backend.runs > 0, true)
  assert_eq(backend.elements, 3)
}

///|
test "a second backend gives the same answer as the default one" {
  let left = from_doubles([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
  let right = from_doubles([0.5, 1.5, 2.5], [3])
  let backend = Counting::new()
  assert_eq(
    add(left, right, kernel=backend).to_doubles(),
    add(left, right).to_doubles(),
  )
  assert_eq(
    sum(left, axis=1, kernel=backend).to_doubles(),
    sum(left, axis=1).to_doubles(),
  )
  assert_eq(sqrt(left, kernel=backend).to_doubles(), sqrt(left).to_doubles())
}

///|
test "the vector path and the scalar tail agree" {
  // Lengths on either side of a vector register and through a tail of every size.
  for length in 1..<17 {
    let values = arange(start=0.0, stop=length.to_double())
    let doubled = multiply(values, values)
    for i in 0..<length {
      let expected = i.to_double() * i.to_double()
      assert_eq(doubled.at([i]).to_double(), expected)
    }
    assert_eq(
      sum(values).item().to_double(),
      length.to_double() * (length - 1).to_double() / 2.0,
    )
  }
}

///|
test "a float32 array goes through the vector path too" {
  let values = from_doubles([1.0, 2.0, 3.0, 4.0, 5.0], [5], dtype=F32)
  let squares = multiply(values, values)
  assert_eq(squares.dtype(), F32)
  assert_eq(squares.to_doubles(), [1.0, 4.0, 9.0, 16.0, 25.0])
  assert_eq(negative(values).to_doubles(), [-1.0, -2.0, -3.0, -4.0, -5.0])
  assert_eq(sum(values, axis=0, kernel=Counting::new()).to_doubles(), [15.0])
}

///|
test "a matrix product goes through the backend" {
  let backend = Counting::new()
  let left = from_doubles([1.0, 2.0, 3.0, 4.0], [2, 2])
  let right = from_doubles([0.0, 1.0, 1.0, 0.0], [2, 2])
  let product = matmul(left, right, kernel=backend)
  assert_eq(product.to_doubles(), [2.0, 1.0, 4.0, 3.0])
  assert_eq(backend.runs, 1)
}
