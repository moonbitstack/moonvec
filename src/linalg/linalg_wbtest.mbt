// Copyright 2026 Leo Cheng
// SPDX-License-Identifier: Apache-2.0

///|
/// True when every element of `actual` is within `tolerance` of the matching element of `expected`.
fn close_to(
  actual : @v.NDArray,
  expected : @v.NDArray,
  tolerance : Double,
) -> Bool raise @v.VecError {
  if actual.shape() != expected.shape() {
    return false
  }
  let left = actual.to_doubles()
  let right = expected.to_doubles()
  for i in 0..<left.length() {
    let scale = left[i].abs().max(right[i].abs()).max(1.0)
    if (left[i] - right[i]).abs() > tolerance * scale {
      return false
    }
  }
  true
}

///|
/// The largest absolute difference between two matrices, for reporting.
fn largest_difference(
  actual : @v.NDArray,
  expected : @v.NDArray,
) -> Double raise @v.VecError {
  let left = actual.to_doubles()
  let right = expected.to_doubles()
  let mut worst = 0.0
  for i in 0..<left.length() {
    worst = worst.max((left[i] - right[i]).abs())
  }
  worst
}

///|
test "solving a two by two system" {
  let a = @v.matrix([[3.0, 1.0], [1.0, 2.0]])
  let b = @v.from_doubles([9.0, 8.0], [2])
  let x = solve(a, b)
  assert_eq(close_to(x, @v.from_doubles([2.0, 3.0], [2]), 1.0e-12), true)
}

///|
test "solving a system with several right-hand sides" {
  let a = @v.matrix([[2.0, 0.0], [0.0, 4.0]])
  let b = @v.matrix([[2.0, 4.0], [8.0, 4.0]])
  let x = solve(a, b)
  assert_eq(x.shape(), [2, 2])
  assert_eq(close_to(x, @v.matrix([[1.0, 2.0], [2.0, 1.0]]), 1.0e-12), true)
}

///|
test "a singular system is refused" {
  let a = @v.matrix([[1.0, 2.0], [2.0, 4.0]])
  let b = @v.from_doubles([1.0, 1.0], [2])
  let refused : Result[Unit, @v.VecError] = try {
    let _ = solve(a, b)
    Ok(())
  } catch {
    error => Err(error)
  }
  assert_eq(refused is Err(NotSolvable(_)), true)
}

///|
test "the determinant is the product of the pivots" {
  assert_eq(det(@v.matrix([[1.0, 2.0], [3.0, 4.0]])), -2.0)
  assert_eq(
    det(@v.matrix([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]])),
    24.0,
  )
  assert_eq(det(@v.matrix([[1.0, 2.0], [2.0, 4.0]])), 0.0)
}

///|
test "an inverse multiplies back to the identity" {
  let a = @v.matrix([[4.0, 7.0], [2.0, 6.0]])
  let inverse = inv(a)
  let product = @v.matmul(a, inverse)
  assert_eq(close_to(product, @v.eye(n=2), 1.0e-12), true)
}

///|
test "the Cholesky factor of a known matrix" {
  let a = @v.matrix([
    [4.0, 12.0, -16.0],
    [12.0, 37.0, -43.0],
    [-16.0, -43.0, 98.0],
  ])
  let l = cholesky(a)
  let expected = @v.matrix([[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]])
  assert_eq(close_to(l, expected, 1.0e-12), true)
}

///|
test "Cholesky refuses a matrix that is not positive definite" {
  let a = @v.matrix([[1.0, 2.0], [2.0, 1.0]])
  let refused : Result[Unit, @v.VecError] = try {
    let _ = cholesky(a)
    Ok(())
  } catch {
    error => Err(error)
  }
  assert_eq(refused is Err(NotSolvable(_)), true)
}

///|
test "a QR factorisation rebuilds the matrix and has orthonormal columns" {
  let a = @v.matrix([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
  let (q, r) = qr(a)
  assert_eq(q.shape(), [3, 2])
  assert_eq(r.shape(), [2, 2])
  let rebuilt = @v.matmul(q, r)
  assert_eq(largest_difference(rebuilt, a) < 1.0e-12, true)
  let gram = @v.matmul(@v.transpose(q), q)
  assert_eq(largest_difference(gram, @v.eye(n=2)) < 1.0e-12, true)
  // R is upper triangular: the entry below the diagonal is zero.
  assert_eq(r.at([1, 0]).to_double(), 0.0)
}

///|
test "least squares fits a line through points off it" {
  // Three points, none of them on a line: the fit has slope 2.5 and intercept minus two thirds.
  let a = @v.matrix([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
  let b = @v.from_doubles([2.0, 4.0, 7.0], [3])
  let x = lstsq(a, b)
  let expected = @v.from_doubles([-2.0 / 3.0, 2.5], [2])
  assert_eq(close_to(x, expected, 1.0e-12), true)
}

///|
test "a singular value decomposition rebuilds the matrix" {
  let a = @v.matrix([[3.0, 0.0], [0.0, -2.0]])
  let (u, s, vt) = svd(a)
  assert_eq(close_to(s, @v.from_doubles([3.0, 2.0], [2]), 1.0e-12), true)
  let diagonal = @v.from_doubles([3.0, 0.0, 0.0, 2.0], [2, 2])
  let rebuilt = @v.matmul(@v.matmul(u, diagonal), vt)
  assert_eq(largest_difference(rebuilt, a) < 1.0e-12, true)
}

///|
test "a wide matrix has as many singular values as its shorter side" {
  let a = @v.matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  let (u, s, vt) = svd(a)
  assert_eq(s.shape(), [2])
  assert_eq(u.shape(), [2, 2])
  assert_eq(vt.shape(), [2, 3])
  let diagonal = @v.from_doubles(
    [s.at([0]).to_double(), 0.0, 0.0, s.at([1]).to_double()],
    [2, 2],
  )
  let rebuilt = @v.matmul(@v.matmul(u, diagonal), vt)
  assert_eq(largest_difference(rebuilt, a) < 1.0e-12, true)
}

///|
test "eigenvalues of a symmetric matrix arrive in order" {
  let a = @v.matrix([[2.0, 1.0], [1.0, 2.0]])
  let (values, vectors) = eigh(a)
  assert_eq(close_to(values, @v.from_doubles([1.0, 3.0], [2]), 1.0e-12), true)
  // Each column of the vectors is an eigenvector: a v = lambda v.
  assert_eq(vectors.shape(), [2, 2])
  for column in 0..<2 {
    let vector = @v.from_doubles(
      [vectors.at([0, column]).to_double(), vectors.at([1, column]).to_double()],
      [2, 1],
    )
    let applied = @v.matmul(a, vector)
    let scaled = lam(values.at([column]).to_double(), vector)
    assert_eq(largest_difference(applied, scaled) < 1.0e-12, true)
  }
}

///|
/// Multiplies every element of `vector` by `factor`.
fn lam(factor : Double, vector : @v.NDArray) -> @v.NDArray raise @v.VecError {
  @v.multiply_scalar(vector, factor)
}

///|
test "matrix powers multiply out" {
  let a = @v.matrix([[2.0, 1.0], [0.0, 1.0]])
  let cubed = matrix_power(a, exponent=3)
  let expected = @v.matrix([[8.0, 7.0], [0.0, 1.0]])
  assert_eq(close_to(cubed, expected, 1.0e-12), true)
  let inverted = matrix_power(a, exponent=-1)
  assert_eq(close_to(inverted, inv(a), 1.0e-12), true)
  assert_eq(close_to(matrix_power(a, exponent=0), @v.eye(n=2), 1.0e-12), true)
}

///|
test "the Kronecker product tiles the second matrix" {
  let a = @v.matrix([[1.0, 2.0], [3.0, 4.0]])
  let b = @v.matrix([[0.0, 5.0], [6.0, 7.0]])
  let product = kron(a, b)
  assert_eq(product.shape(), [4, 4])
  let expected = @v.matrix([
    [0.0, 5.0, 0.0, 10.0],
    [6.0, 7.0, 12.0, 14.0],
    [0.0, 15.0, 0.0, 20.0],
    [18.0, 21.0, 24.0, 28.0],
  ])
  assert_eq(close_to(product, expected, 1.0e-12), true)
}

///|
test "norms of a vector" {
  let v = @v.from_doubles([3.0, -4.0], [2])
  assert_eq(norm(v).item().to_double(), 5.0)
  assert_eq(norm(v, order=One).item().to_double(), 7.0)
  assert_eq(norm(v, order=Infinity).item().to_double(), 4.0)
  assert_eq(norm(v, order=NegInfinity).item().to_double(), 3.0)
  assert_eq(norm(v, order=P(3.0)).item().to_double() > 4.497, true)
}

///|
test "norms of a matrix" {
  let a = @v.matrix([[1.0, -2.0], [3.0, 4.0]])
  assert_eq(matrix_norm(a), 30.0.sqrt())
  assert_eq(matrix_norm(a, order=One), 6.0)
  assert_eq(matrix_norm(a, order=Infinity), 7.0)
  assert_eq(matrix_norm(a, order=Two) > 5.1, true)
}

///|
test "a norm along an axis keeps the other axes" {
  let a = @v.matrix([[3.0, 0.0], [0.0, 4.0]])
  let down = norm(a, axis=0)
  assert_eq(down.shape(), [2])
  assert_eq(down.at([0]).to_double(), 3.0)
  let across = norm(a, axis=1)
  assert_eq(across.at([1]).to_double(), 4.0)
}
