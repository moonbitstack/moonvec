// Copyright 2026 Leo Cheng
// SPDX-License-Identifier: Apache-2.0

///|
/// Broadcasting, dtype promotion and what happens when two shapes cannot meet.
fn main raise {
  // A column and a row meet to fill a table, without either being copied first.
  let column = @vec.from_doubles([1.0, 2.0, 3.0], [3, 1])
  let row = @vec.from_doubles([10.0, 20.0, 30.0, 40.0], [1, 4])
  let table = @vec.add(column, row)
  println(table.to_string())

  // The dtype follows the wider of the two, and an integer division answers in floats.
  let counts = @vec.from_numbers(
    [@vec.Signed(1L), @vec.Signed(2L), @vec.Signed(3L)],
    [3],
    dtype=@vec.I32,
  )
  let weights = @vec.from_doubles([0.5, 0.25, 0.125], [3], dtype=@vec.F32)
  println("product " + @vec.multiply(counts, weights).to_string())
  println(
    "quotient " +
    @vec.divide(counts, @vec.full([3], @vec.Real(2.0), dtype=@vec.I32)).to_string(),
  )

  // Choosing between two arrays by a condition, which is how a piecewise function is written.
  let values = @vec.from_doubles([-2.0, -0.5, 0.5, 2.0], [4])
  let clipped = @vec.select(
    @vec.greater(values, @vec.zeros([4])),
    values,
    @vec.zeros([4]),
  )
  println("positive parts " + clipped.to_string())

  // A shape that cannot broadcast is refused with the two shapes in the message.
  let mismatch : Result[@vec.NDArray, @vec.VecError] = Ok(
    @vec.add(@vec.zeros([2, 3]), @vec.zeros([4])),
  ) catch {
    error => Err(error)
  }
  match mismatch {
    Err(error) => println("refused: " + error.message())
    Ok(_) => println("unexpected")
  }
}
