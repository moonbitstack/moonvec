// Copyright 2026 Leo Cheng
// SPDX-License-Identifier: Apache-2.0

///|
/// A layout describes where elements sit; an array view is that description applied to a buffer. The
/// two share the same idea of shape and stride, so a tile computed by the layout algebra becomes a
/// view without copying anything.
fn main raise {
  let buffer = @vec.arange(start=0.0, stop=12.0)
  println(buffer.to_string())

  // A tile of four elements per thread, three threads.
  let layout = @cute.compact([12])
  let (tile, rest) = layout.divide(tile=4)
  println("tile " + tile.to_string() + ", tiles " + rest.to_string())

  // The middle tile is a view of the same buffer, four elements long.
  let middle = @cute.Tuple([@cute.Mode(4, 1), @cute.Mode(1, 4)])
  let start = middle.offset([0, 1])
  let block = @vec.NDArray::new(
    buffer=buffer.buffer(),
    shape=[4],
    strides=[1],
    offset=start,
  )
  println("second block " + block.to_string())

  // Writing through the view writes into the buffer.
  block.set_at([0], @vec.Real(-1.0))
  println("after writing " + buffer.to_string())

  // A transposed layout is the same elements read column by column.
  let transposed = @cute.Tuple([@cute.Mode(3, 1), @cute.Mode(4, 3)])
  println("transposed " + transposed.to_string())
  println(
    "its strides " +
    transposed.strides().map(fn(s) { s.to_string() }).join(", "),
  )
}
