// Copyright 2026 Leo Cheng
// SPDX-License-Identifier: Apache-2.0

///|
/// The first things an array library has to do: make an array, look at it, slice it, and reduce it.
fn main raise {
  let a = @vec.from_doubles([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
  println(a.to_string())
  let row = @vec.slice(a, [@vec.At(0)])
  println(row.to_string())
  let column = @vec.slice(a, [@vec.All, @vec.At(2)])
  println(column.to_string())
  println(@vec.transpose(a).to_string())
  let across = @vec.sum(a, axis=0)
  println("column sums " + across.to_string())
  println("largest " + @vec.max(a).to_string())
  println("mean " + @vec.mean(a).to_string())
}
