// Copyright 2026 Leo Cheng
// SPDX-License-Identifier: Apache-2.0

///|
/// Reads an array as a row-major matrix of doubles, together with its row and column counts.
fn as_matrix(a : @v.NDArray) -> (Array[Double], Int32, Int32) raise @v.VecError {
  if a.ndim() != 2 {
    raise @v.ShapeMismatch(want=[2], got=a.shape())
  }
  let shape = a.shape()
  (a.to_doubles(), shape[0], shape[1])
}

///|
/// The dtype a linear algebra answer is stored as: double precision for exact input, the input's own
/// width for a float one.
fn real_dtype(a : @v.NDArray) -> @v.DType {
  match a.dtype() {
    @v.F32 | @v.C64 => @v.F32
    _ => @v.F64
  }
}

///|
/// A matrix from row-major doubles.
fn matrix(
  values : Array[Double],
  rows : Int32,
  columns : Int32,
  dtype : @v.DType,
) -> @v.NDArray raise @v.VecError {
  @v.from_doubles(values, [rows, columns], dtype~)
}

///|
/// A vector from doubles.
fn vector(
  values : Array[Double],
  dtype : @v.DType,
) -> @v.NDArray raise @v.VecError {
  @v.from_doubles(values, [values.length()], dtype~)
}

///|
/// A single number as a 0-dimensional array.
fn scalar(value : Double, dtype : @v.DType) -> @v.NDArray raise @v.VecError {
  @v.from_doubles([value], [], dtype~)
}

///|
/// The LU factorisation with partial pivoting: the multipliers and the pivots in one matrix, and the
/// row swap each step made.
///
/// The leading entry of each column is swapped to the diagonal before the column is eliminated, which
/// keeps the growth of the entries bounded for the matrices that appear in practice. A pivot of zero
/// means the matrix is singular and is reported rather than divided by.
fn lu_factor(
  a : Array[Double],
  n : Int32,
) -> (Array[Double], Array[Int32], Bool) {
  let lu = a.copy()
  let swaps = Array::make(n, 0)
  let mut singular = false
  for column in 0..<n {
    let mut pivot = column
    let mut largest = lu[column * n + column].abs()
    for row in (column + 1)..<n {
      let candidate = lu[row * n + column].abs()
      if candidate > largest {
        largest = candidate
        pivot = row
      }
    }
    swaps[column] = pivot
    if pivot != column {
      for k in 0..<n {
        let left = lu[column * n + k]
        lu[column * n + k] = lu[pivot * n + k]
        lu[pivot * n + k] = left
      }
    }
    let leading = lu[column * n + column]
    if leading == 0.0 {
      singular = true
      continue
    }
    for row in (column + 1)..<n {
      let factor = lu[row * n + column] / leading
      lu[row * n + column] = factor
      for k in (column + 1)..<n {
        lu[row * n + k] = lu[row * n + k] - factor * lu[column * n + k]
      }
    }
  }
  (lu, swaps, singular)
}

///|
/// Applies the row swaps of an LU factorisation to a right-hand side.
fn apply_swaps(
  values : Array[Double],
  swaps : Array[Int32],
  n : Int32,
  width : Int32,
) -> Unit {
  for column in 0..<n {
    let pivot = swaps[column]
    if pivot != column {
      for k in 0..<width {
        let left = values[column * width + k]
        values[column * width + k] = values[pivot * width + k]
        values[pivot * width + k] = left
      }
    }
  }
}

///|
/// Solves `a x = b` by LU factorisation with partial pivoting.
///
/// `b` may be a vector or a matrix, and its first axis must match `a`'s. A singular matrix is
/// refused rather than answered with an arbitrary solution.
pub fn solve(a : @v.NDArray, b : @v.NDArray) -> @v.NDArray raise @v.VecError {
  let (left, n, columns) = as_matrix(a)
  if n != columns {
    raise @v.NotSolvable(message="solve needs a square matrix")
  }
  // A right-hand side that is a vector is solved as a matrix of one column, then squeezed back.
  let vector_side = b.ndim() == 1
  let shaped = if vector_side { @v.reshape(b, [b.size(), 1]) } else { b }
  let (right, rows, width) = as_matrix(shaped)
  if rows != n {
    raise @v.ShapeMismatch(want=a.shape(), got=b.shape())
  }
  let (lu, swaps, singular) = lu_factor(left, n)
  if singular {
    raise @v.NotSolvable(message="the matrix is singular")
  }
  let work = right.copy()
  apply_swaps(work, swaps, n, width)
  // Forward substitution on the unit lower triangle, then back substitution on the upper.
  for row in 1..<n {
    for k in 0..<width {
      let mut total = work[row * width + k]
      for column in 0..<row {
        total = total - lu[row * n + column] * work[column * width + k]
      }
      work[row * width + k] = total
    }
  }
  let mut row = n - 1
  while row >= 0 {
    for k in 0..<width {
      let mut total = work[row * width + k]
      for column in (row + 1)..<n {
        total = total - lu[row * n + column] * work[column * width + k]
      }
      work[row * width + k] = total / lu[row * n + row]
    }
    row = row - 1
  }
  if vector_side {
    vector(work, real_dtype(a))
  } else {
    matrix(work, n, width, real_dtype(a))
  }
}

///|
/// The inverse, by solving for the identity.
pub fn inv(a : @v.NDArray) -> @v.NDArray raise @v.VecError {
  let (_, n, columns) = as_matrix(a)
  if n != columns {
    raise @v.NotSolvable(message="only a square matrix has an inverse")
  }
  let identity = @v.eye(n~, dtype=@v.F64)
  solve(a, identity)
}

///|
/// The determinant, from the product of the pivots and the sign of the row swaps.
pub fn det(a : @v.NDArray) -> Double raise @v.VecError {
  let (values, n, columns) = as_matrix(a)
  if n != columns {
    raise @v.NotSolvable(message="only a square matrix has a determinant")
  }
  let (lu, swaps, singular) = lu_factor(values, n)
  if singular {
    return 0.0
  }
  let mut sign = 1.0
  for column in 0..<n {
    if swaps[column] != column {
      sign = -sign
    }
  }
  let mut product = sign
  for i in 0..<n {
    product = product * lu[i * n + i]
  }
  product
}

///|
/// The Cholesky factorisation `a = l lᵀ` of a symmetric positive definite matrix, returning the lower
/// triangle with zeros above it.
pub fn cholesky(a : @v.NDArray) -> @v.NDArray raise @v.VecError {
  let (values, n, columns) = as_matrix(a)
  if n != columns {
    raise @v.NotSolvable(message="cholesky needs a square matrix")
  }
  let out = Array::make(n * n, 0.0)
  for row in 0..<n {
    for column in 0..<=row {
      let mut total = values[row * n + column]
      for k in 0..<column {
        total = total - out[row * n + k] * out[column * n + k]
      }
      if row == column {
        if total <= 0.0 {
          raise @v.NotSolvable(
            message="the matrix is not positive definite at row " +
              row.to_string(),
          )
        }
        out[row * n + column] = total.sqrt()
      } else {
        out[row * n + column] = total / out[column * n + column]
      }
    }
  }
  matrix(out, n, n, real_dtype(a))
}
