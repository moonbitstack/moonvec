// Copyright 2026 Leo Cheng
// SPDX-License-Identifier: Apache-2.0

///|
test "a mode addresses its size with its stride" {
  let layout = mode(size=4, stride=3)
  assert_eq(layout.size(), 4)
  assert_eq(layout.cosize(), 10)
  assert_eq(layout.offset([0]), 0)
  assert_eq(layout.offset([3]), 9)
}

///|
test "a tuple adds the offsets of its modes" {
  let layout = Tuple([Mode(2, 4), Mode(3, 1)])
  assert_eq(layout.size(), 6)
  assert_eq(layout.shape(), [2, 3])
  assert_eq(layout.strides(), [4, 1])
  assert_eq(layout.offset([1, 2]), 6)
  assert_eq(layout.cosize(), 7)
}

///|
test "a compact layout is row-major" {
  let layout = compact([2, 3])
  assert_eq(layout.shape(), [2, 3])
  assert_eq(layout.strides(), [3, 1])
  assert_eq(layout.offset_of_index(4), 4)
  assert_eq(layout.cosize(), 6)
}

///|
test "the offset of an index walks the layout's own order" {
  let transposed = Tuple([Mode(3, 1), Mode(2, 3)])
  assert_eq(transposed.size(), 6)
  assert_eq(transposed.offset_of_index(0), 0)
  assert_eq(transposed.offset_of_index(1), 3)
  assert_eq(transposed.offset_of_index(2), 1)
}

///|
test "a stride of zero repeats the same element" {
  let broadcast = Tuple([Mode(2, 0), Mode(3, 1)])
  assert_eq(broadcast.offset([0, 2]), 2)
  assert_eq(broadcast.offset([1, 2]), 2)
  assert_eq(broadcast.cosize(), 3)
}

///|
test "a negative stride walks backwards" {
  let reversed = Mode(4, -1)
  assert_eq(reversed.offset([0]), 0)
  assert_eq(reversed.offset([3]), -3)
}

///|
test "coalesce merges modes that walk without a gap" {
  let rows = Tuple([Mode(2, 3), Mode(3, 1)])
  let merged = rows.coalesce()
  assert_eq(merged.shape(), [6])
  assert_eq(merged.strides(), [1])
  let gapped = Tuple([Mode(2, 6), Mode(3, 1)])
  assert_eq(gapped.coalesce().shape(), [2, 3])
}

///|
test "coalesce drops a size-one mode" {
  let padded = Tuple([Mode(1, 8), Mode(3, 1)])
  assert_eq(padded.coalesce().shape(), [3])
}

///|
test "divide gives the tile and the tile index" {
  let layout = Mode(12, 1)
  let (tile, rest) = layout.divide(tile=4)
  assert_eq(tile.size(), 4)
  assert_eq(rest.size(), 3)
  assert_eq(tile.offset([2]), 2)
  assert_eq(rest.offset([2]), 8)
  assert_eq(tile.size() * rest.size(), layout.size())
}

///|
test "divide refuses a size that does not split evenly" {
  let layout = Mode(10, 1)
  let refused : Result[Unit, LayoutError] = try {
    let _ = layout.divide(tile=4)
    Ok(())
  } catch {
    error => Err(error)
  }
  match refused {
    Err(NotDivisible(tile~, size~)) => {
      assert_eq(tile, 4)
      assert_eq(size, 10)
    }
    _ => fail("a tile of four does not divide ten")
  }
}

///|
test "a thread and value product addresses a tile per thread" {
  let values = compact([2, 3])
  let threads = Mode(4, 6)
  let combined = product(threads, values)
  assert_eq(combined.size(), 24)
  assert_eq(combined.offset([1, 0, 0]), 6)
  assert_eq(combined.offset([1, 1, 2]), 11)
}

///|
test "a coordinate of the wrong rank is refused" {
  let layout = compact([2, 3])
  let refused : Result[Int32, LayoutError] = Ok(layout.offset([1])) catch {
    error => Err(error)
  }
  match refused {
    Err(RankMismatch(want~, got~)) => {
      assert_eq(want, 2)
      assert_eq(got, 1)
    }
    _ => fail("one entry for two modes is not a coordinate")
  }
}

///|
test "layouts print as shape and stride pairs" {
  assert_eq(compact([2, 3]).to_string(), "(_2:_3, _3:_1)")
  assert_eq(mode(size=4, stride=2).to_string(), "_4:_2")
}
