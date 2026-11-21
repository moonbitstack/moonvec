// Copyright 2026 Leo Cheng
// SPDX-License-Identifier: Apache-2.0

///|
/// Linear algebra: multiply, solve, factor, and measure.
fn main raise {
  let a = @vec.matrix([[3.0, 1.0], [1.0, 2.0]])
  let b = @vec.from_doubles([9.0, 8.0], [2])
  println("solution " + @linalg.solve(a, b).to_string())
  println("determinant " + @vec.format_real(@linalg.det(a)))
  println("inverse " + @linalg.inv(a).to_string())

  let design = @vec.matrix([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]])
  let observed = @vec.from_doubles([1.1, 2.2, 2.9, 4.2], [4])
  println("fitted line " + @linalg.lstsq(design, observed).to_string())

  let (u, s, vt) = @linalg.svd(
    @vec.matrix([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
  )
  println("singular values " + s.to_string())
  let rebuilt = @vec.matmul(
    @vec.matmul(
      u,
      @vec.from_doubles(
        [s.at([0]).to_double(), 0.0, 0.0, s.at([1]).to_double()],
        [2, 2],
      ),
    ),
    vt,
  )
  println("rebuilt " + rebuilt.to_string())
  println(
    "largest singular value " +
    @vec.format_real(@linalg.matrix_norm(rebuilt, order=@linalg.Two)),
  )
}
