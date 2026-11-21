// Copyright 2026 Leo Cheng
// SPDX-License-Identifier: Apache-2.0

///|
/// The thin QR factorisation by Householder reflections: `q` is `m` by `k` with orthonormal columns
/// and `r` is `k` by `n` upper triangular, where `k` is the smaller of the two sides.
pub fn qr(a : @v.NDArray) -> (@v.NDArray, @v.NDArray) raise @v.VecError {
  let (values, m, n) = as_matrix(a)
  let k = if m < n { m } else { n }
  // The reflections are applied to a working copy; each is stored as its own column of `v`.
  let working = values.copy()
  let reflectors = Array::make(k * m, 0.0)
  let taus = Array::make(k, 0.0)
  for step in 0..<k {
    let mut norm = 0.0
    for row in step..<m {
      let value = working[row * n + step]
      norm = norm + value * value
    }
    norm = norm.sqrt()
    if norm == 0.0 {
      taus[step] = 0.0
      continue
    }
    let leading = working[step * n + step]
    // The opposite sign avoids subtracting two nearly equal numbers.
    let alpha = if leading >= 0.0 { -norm } else { norm }
    let denominator = leading - alpha
    reflectors[step * m + step] = 1.0
    for row in (step + 1)..<m {
      reflectors[step * m + row] = working[row * n + step] / denominator
    }
    let mut squared = 0.0
    for row in step..<m {
      squared = squared +
        reflectors[step * m + row] * reflectors[step * m + row]
    }
    taus[step] = 2.0 / squared
    for column in step..<n {
      let mut dot = 0.0
      for row in step..<m {
        dot = dot + reflectors[step * m + row] * working[row * n + column]
      }
      let scale = taus[step] * dot
      for row in step..<m {
        working[row * n + column] = working[row * n + column] -
          scale * reflectors[step * m + row]
      }
    }
  }
  let r = Array::make(k * n, 0.0)
  for row in 0..<k {
    for column in row..<n {
      r[row * n + column] = working[row * n + column]
    }
  }
  // Q is the identity with the reflections applied in reverse order.
  let q = Array::make(m * k, 0.0)
  for column in 0..<k {
    q[column * k + column] = 1.0
  }
  let mut step = k - 1
  while step >= 0 {
    if taus[step] != 0.0 {
      for column in 0..<k {
        let mut dot = 0.0
        for row in step..<m {
          dot = dot + reflectors[step * m + row] * q[row * k + column]
        }
        let scale = taus[step] * dot
        for row in step..<m {
          q[row * k + column] = q[row * k + column] -
            scale * reflectors[step * m + row]
        }
      }
    }
    step = step - 1
  }
  (matrix(q, m, k, real_dtype(a)), matrix(r, k, n, real_dtype(a)))
}

///|
/// The least squares solution of `a x = b` for an overdetermined system, by QR factorisation.
///
/// A system with fewer rows than columns has infinitely many solutions and is refused, as is a matrix
/// whose columns are linearly dependent, because the answer would not be unique.
pub fn lstsq(a : @v.NDArray, b : @v.NDArray) -> @v.NDArray raise @v.VecError {
  let (_, m, n) = as_matrix(a)
  if m < n {
    // Fewer rows than columns: infinitely many solutions, and the shortest one is wanted. It is
    // Aᵀ y with (A Aᵀ) y = b, which is the least squares problem of Aᵀ.
    let transposed = @v.transpose(a)
    let gram = @v.matmul(a, transposed)
    let multiplier = solve(gram, b)
    return @v.matmul(transposed, multiplier)
  }
  let (q, r) = qr(a)
  let vector_side = b.ndim() == 1
  let shaped = if vector_side { @v.reshape(b, [b.size(), 1]) } else { b }
  let (right, rows, width) = as_matrix(shaped)
  if rows != m {
    raise @v.ShapeMismatch(want=a.shape(), got=b.shape())
  }
  // qᵀ b, then back substitution through r.
  let projected = Array::make(n * width, 0.0)
  let q_values = q.to_doubles()
  for column in 0..<width {
    for row in 0..<n {
      let mut total = 0.0
      for i in 0..<m {
        total = total + q_values[i * n + row] * right[i * width + column]
      }
      projected[row * width + column] = total
    }
  }
  let r_values = r.to_doubles()
  let mut row = n - 1
  while row >= 0 {
    let pivot = r_values[row * n + row]
    if pivot == 0.0 {
      raise @v.NotSolvable(
        message="the columns of the matrix are linearly dependent",
      )
    }
    for column in 0..<width {
      let mut total = projected[row * width + column]
      for k in (row + 1)..<n {
        total = total - r_values[row * n + k] * projected[k * width + column]
      }
      projected[row * width + column] = total / pivot
    }
    row = row - 1
  }
  if vector_side {
    vector(projected, real_dtype(a))
  } else {
    matrix(projected, n, width, real_dtype(a))
  }
}

///|
/// The singular value decomposition by one-sided Jacobi rotations: `a = u s vᵀ`, with the singular
/// values in descending order.
///
/// Each rotation orthogonalises a pair of columns, which converges quickly and keeps the accuracy of
/// small singular values that a normal-equation method would lose.
pub fn svd(
  a : @v.NDArray,
) -> (@v.NDArray, @v.NDArray, @v.NDArray) raise @v.VecError {
  let (values, m, n) = as_matrix(a)
  let transpose_first = m < n
  let rows = if transpose_first { n } else { m }
  let columns = if transpose_first { m } else { n }
  // Work on the tall orientation; if the input was wide, the factors swap at the end.
  let work = Array::make(rows * columns, 0.0)
  for i in 0..<rows {
    for j in 0..<columns {
      work[i * columns + j] = if transpose_first {
        values[j * n + i]
      } else {
        values[i * columns + j]
      }
    }
  }
  let v = Array::make(columns * columns, 0.0)
  for i in 0..<columns {
    v[i * columns + i] = 1.0
  }
  let sweep_limit = 60
  let mut sweep = 0
  while sweep < sweep_limit {
    let mut rotated = false
    for left in 0..<columns {
      for right in (left + 1)..<columns {
        let mut alpha = 0.0
        let mut beta = 0.0
        let mut gamma = 0.0
        for row in 0..<rows {
          let a_row = work[row * columns + left]
          let b_row = work[row * columns + right]
          alpha = alpha + a_row * a_row
          beta = beta + b_row * b_row
          gamma = gamma + a_row * b_row
        }
        if gamma == 0.0 {
          continue
        }
        let tolerance = 1.0e-17 * alpha.sqrt() * beta.sqrt()
        if gamma.abs() <= tolerance {
          continue
        }
        rotated = true
        let zeta = (beta - alpha) / (2.0 * gamma)
        let t = sign_of(zeta) / (zeta.abs() + (1.0 + zeta * zeta).sqrt())
        let cosine = 1.0 / (1.0 + t * t).sqrt()
        let sine = cosine * t
        for row in 0..<rows {
          let a_row = work[row * columns + left]
          let b_row = work[row * columns + right]
          work[row * columns + left] = cosine * a_row - sine * b_row
          work[row * columns + right] = sine * a_row + cosine * b_row
        }
        for row in 0..<columns {
          let a_row = v[row * columns + left]
          let b_row = v[row * columns + right]
          v[row * columns + left] = cosine * a_row - sine * b_row
          v[row * columns + right] = sine * a_row + cosine * b_row
        }
      }
    }
    if !rotated {
      break
    }
    sweep = sweep + 1
  }
  // The singular values are the norms of the columns of the rotated matrix.
  let singular = Array::make(columns, 0.0)
  for column in 0..<columns {
    let mut total = 0.0
    for row in 0..<rows {
      let value = work[row * columns + column]
      total = total + value * value
    }
    singular[column] = total.sqrt()
  }
  // Sort from largest to smallest, carrying both factors along.
  let order = Array::makei(columns, fn(i) { i })
  order.sort_by(fn(left, right) {
    let a_value = singular[left]
    let b_value = singular[right]
    if b_value < a_value {
      -1
    } else if b_value > a_value {
      1
    } else {
      left.compare(right)
    }
  })
  let sorted = Array::make(columns, 0.0)
  let u = Array::make(rows * columns, 0.0)
  let vt = Array::make(columns * columns, 0.0)
  for i in 0..<columns {
    let from = order[i]
    sorted[i] = singular[from]
    for row in 0..<rows {
      u[row * columns + i] = if singular[from] == 0.0 {
        0.0
      } else {
        work[row * columns + from] / singular[from]
      }
    }
    // vᵀ row i is v column `from`.
    for column in 0..<columns {
      vt[i * columns + column] = v[column * columns + from]
    }
  }
  let dtype = real_dtype(a)
  if transpose_first {
    // aᵀ = u s vᵀ means a = v s uᵀ, so the two orthogonal factors swap and the taller one transposes.
    let transposed_u = Array::make(rows * columns, 0.0)
    for i in 0..<rows {
      for j in 0..<columns {
        transposed_u[j * rows + i] = u[i * columns + j]
      }
    }
    (
      matrix(vt, columns, columns, dtype),
      vector(sorted, dtype),
      matrix(transposed_u, columns, rows, dtype),
    )
  } else {
    (
      matrix(u, rows, columns, dtype),
      vector(sorted, dtype),
      matrix(vt, columns, columns, dtype),
    )
  }
}

///|
/// `1` or `-1` by the sign, with a zero taking the positive side so that the rotation never divides
/// by zero.
fn sign_of(value : Double) -> Double {
  if value < 0.0 {
    -1.0
  } else {
    1.0
  }
}

///|
/// The eigenvalues and eigenvectors of a symmetric matrix, by cyclic Jacobi rotations; the columns of
/// the second result are the eigenvectors, in the order of the values.
pub fn eigh(a : @v.NDArray) -> (@v.NDArray, @v.NDArray) raise @v.VecError {
  let (values, n, columns) = as_matrix(a)
  if n != columns {
    raise @v.NotSolvable(message="eigh needs a square matrix")
  }
  let work = values.copy()
  let vectors = Array::make(n * n, 0.0)
  for i in 0..<n {
    vectors[i * n + i] = 1.0
  }
  for _sweep in 0..<60 {
    let mut off = 0.0
    for i in 0..<n {
      for j in (i + 1)..<n {
        off = off + work[i * n + j] * work[i * n + j]
      }
    }
    if off.sqrt() < 1.0e-15 {
      break
    }
    for left in 0..<n {
      for right in (left + 1)..<n {
        let gamma = work[left * n + right]
        if gamma.abs() < 1.0e-300 {
          continue
        }
        let theta = (work[right * n + right] - work[left * n + left]) /
          (2.0 * gamma)
        let t = sign_of(theta) / (theta.abs() + (1.0 + theta * theta).sqrt())
        let cosine = 1.0 / (1.0 + t * t).sqrt()
        let sine = cosine * t
        for k in 0..<n {
          let a_k = work[k * n + left]
          let b_k = work[k * n + right]
          work[k * n + left] = cosine * a_k - sine * b_k
          work[k * n + right] = sine * a_k + cosine * b_k
        }
        for k in 0..<n {
          let a_k = work[left * n + k]
          let b_k = work[right * n + k]
          work[left * n + k] = cosine * a_k - sine * b_k
          work[right * n + k] = sine * a_k + cosine * b_k
        }
        for k in 0..<n {
          let a_k = vectors[k * n + left]
          let b_k = vectors[k * n + right]
          vectors[k * n + left] = cosine * a_k - sine * b_k
          vectors[k * n + right] = sine * a_k + cosine * b_k
        }
      }
    }
  }
  let eigenvalues = Array::make(n, 0.0)
  for i in 0..<n {
    eigenvalues[i] = work[i * n + i]
  }
  let order = Array::makei(n, fn(i) { i })
  order.sort_by(fn(left, right) {
    let a_value = eigenvalues[left]
    let b_value = eigenvalues[right]
    if a_value < b_value {
      -1
    } else if a_value > b_value {
      1
    } else {
      left.compare(right)
    }
  })
  let sorted_values = Array::make(n, 0.0)
  let sorted_vectors = Array::make(n * n, 0.0)
  for i in 0..<n {
    let from = order[i]
    sorted_values[i] = eigenvalues[from]
    for row in 0..<n {
      sorted_vectors[row * n + i] = vectors[row * n + from]
    }
  }
  let dtype = real_dtype(a)
  (vector(sorted_values, dtype), matrix(sorted_vectors, n, n, dtype))
}

///|
/// A square matrix raised to an integer power; a negative power inverts first.
pub fn matrix_power(
  a : @v.NDArray,
  exponent~ : Int32,
) -> @v.NDArray raise @v.VecError {
  let (_, n, columns) = as_matrix(a)
  if n != columns {
    raise @v.NotSolvable(message="only a square matrix has a power")
  }
  if exponent == 0 {
    return @v.eye(n~, dtype=real_dtype(a))
  }
  let mut base = if exponent < 0 { inv(a) } else { a.copy() }
  let mut remaining = if exponent < 0 { -exponent } else { exponent }
  let mut result = @v.eye(n~, dtype=real_dtype(a))
  while remaining > 0 {
    if remaining % 2 == 1 {
      result = @v.matmul(result, base)
    }
    remaining = remaining / 2
    if remaining > 0 {
      base = @v.matmul(base, base)
    }
  }
  result
}

///|
/// The Kronecker product: every element of `a` multiplied by the whole of `b`, tiled to match.
pub fn kron(a : @v.NDArray, b : @v.NDArray) -> @v.NDArray raise @v.VecError {
  let (left, left_rows, left_columns) = as_matrix(a)
  let (right, right_rows, right_columns) = as_matrix(b)
  let rows = left_rows * right_rows
  let columns = left_columns * right_columns
  let out = Array::make(rows * columns, 0.0)
  for i in 0..<left_rows {
    for j in 0..<left_columns {
      let factor = left[i * left_columns + j]
      for k in 0..<right_rows {
        for l in 0..<right_columns {
          out[(i * right_rows + k) * columns + j * right_columns + l] = factor *
            right[k * right_columns + l]
        }
      }
    }
  }
  matrix(out, rows, columns, @v.promote(a.dtype(), b.dtype()))
}
