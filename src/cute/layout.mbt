// Copyright 2026 Leo Cheng
// SPDX-License-Identifier: Apache-2.0

///|
/// A tensor layout: the mapping from a coordinate to an offset in a flat buffer.
///
/// A layout is either a single mode — a length and the distance between neighbours — or a tuple of
/// layouts, which lets a tile of a tile of a matrix stay one object instead of being flattened into a
/// shape nobody can read back. Nothing here holds data: a layout is the description of where the
/// elements of a tensor sit, which is what a kernel needs and what a view is built from.
///
/// A stride may be 0, which repeats an element along that axis (how a broadcast is expressed without
/// copying), or negative, which walks an axis backwards.
pub(all) enum Layout {
  ///
  /// One mode: `size` elements, `stride` apart.
  Mode(Int32, Int32)
  ///
  /// A tuple of layouts, laid out one after another: the offsets add.
  Tuple(Array[Layout])
}

///|
/// A single mode, the building block of every other layout.
pub fn mode(size~ : Int32, stride~ : Int32) -> Layout {
  Mode(size, stride)
}

///|
/// The number of elements the layout addresses.
pub fn Layout::size(self : Layout) -> Int32 {
  match self {
    Mode(size, _) => if size < 0 { 0 } else { size }
    Tuple(parts) => {
      let mut total = 1
      for part in parts {
        total = total * part.size()
      }
      total
    }
  }
}

///|
/// One past the largest offset the layout can produce, which is how much space a buffer of this
/// layout needs.
pub fn Layout::cosize(self : Layout) -> Int32 {
  match self {
    Mode(size, stride) =>
      if size <= 0 {
        0
      } else {
        1 + (size - 1) * abs_stride(stride)
      }
    Tuple(parts) => {
      let mut span = 0
      let mut any = false
      for part in parts {
        if part.size() > 0 {
          span = span + (part.cosize() - 1)
          any = true
        }
      }
      if any {
        span + 1
      } else {
        0
      }
    }
  }
}

///|
/// The absolute value of a stride, without the overflow `abs` of the smallest integer.
fn abs_stride(stride : Int32) -> Int32 {
  if stride < 0 {
    -stride
  } else {
    stride
  }
}

///|
/// The shape of every mode, flattened from the outside in.
pub fn Layout::shape(self : Layout) -> Array[Int32] {
  let out : Array[Int32] = []
  self.collect_shape(out)
  out
}

///|
fn Layout::collect_shape(self : Layout, out : Array[Int32]) -> Unit {
  match self {
    Mode(size, _) => out.push(size)
    Tuple(parts) =>
      for part in parts {
        part.collect_shape(out)
      }
  }
}

///|
/// The stride of every mode, flattened the same way as `shape`, so the two arrays line up.
pub fn Layout::strides(self : Layout) -> Array[Int32] {
  let out : Array[Int32] = []
  self.collect_strides(out)
  out
}

///|
fn Layout::collect_strides(self : Layout, out : Array[Int32]) -> Unit {
  match self {
    Mode(_, stride) => out.push(stride)
    Tuple(parts) =>
      for part in parts {
        part.collect_strides(out)
      }
  }
}

///|
/// The offset of a coordinate, one entry per mode in flattened order.
pub fn Layout::offset(
  self : Layout,
  coordinate : Array[Int32],
) -> Int32 raise LayoutError {
  let needed = self.shape().length()
  if coordinate.length() != needed {
    raise RankMismatch(want=needed, got=coordinate.length())
  }
  self.offset_at(coordinate, 0)
}

///|
fn Layout::offset_at(
  self : Layout,
  coordinate : Array[Int32],
  cursor : Int32,
) -> Int32 {
  match self {
    Mode(_, stride) => coordinate[cursor] * stride
    Tuple(parts) => {
      let mut total = 0
      let mut position = cursor
      for part in parts {
        total = total + part.offset_at(coordinate, position)
        position = position + part.shape().length()
      }
      total
    }
  }
}

///|
/// The offset of the element at `index`, counting the elements in row-major order as the layout
/// itself orders them.
pub fn Layout::offset_of_index(
  self : Layout,
  index : Int32,
) -> Int32 raise LayoutError {
  let total = self.size()
  if index < 0 || index >= total {
    raise IndexOutOfRange(index~, size=total)
  }
  let shape = self.shape()
  let strides = self.strides()
  let mut remaining = index
  let mut offset = 0
  let mut axis = shape.length() - 1
  while axis >= 0 {
    let coordinate = remaining % shape[axis]
    remaining = remaining / shape[axis]
    offset = offset + coordinate * strides[axis]
    axis = axis - 1
  }
  offset
}

///|
/// A layout of `size` elements with no gaps, the usual row-major arrangement.
pub fn compact(shape : Array[Int32]) -> Layout {
  let modes : Array[Layout] = []
  let mut stride = 1
  let mut axis = shape.length() - 1
  while axis >= 0 {
    modes.push(Mode(shape[axis], stride))
    stride = stride * shape[axis]
    axis = axis - 1
  }
  modes.rev_in_place()
  Tuple(modes)
}

///|
/// Two layouts side by side, which is how a thread index and a value index combine into one offset.
pub fn product(left : Layout, right : Layout) -> Layout {
  Tuple([left, right])
}

///|
/// Merges adjacent modes that walk the buffer at a constant step, so that a layout with one row per
/// thread becomes one mode instead of many.
///
/// Two modes merge when the outer one's stride equals the inner one's stride times the inner one's
/// size: then the pair is a single run. A size-one mode disappears, and a stride-zero mode is kept
/// apart because it repeats rather than advances.
pub fn Layout::coalesce(self : Layout) -> Layout {
  let shape = self.shape()
  let strides = self.strides()
  let modes : Array[Layout] = []
  let mut axis = 0
  while axis < shape.length() {
    if shape[axis] == 1 {
      axis = axis + 1
      continue
    }
    let mut size = shape[axis]
    let mut stride = strides[axis]
    while axis + 1 < shape.length() &&
          stride == strides[axis + 1] * shape[axis + 1] {
      axis = axis + 1
      size = size * shape[axis]
      stride = strides[axis]
    }
    modes.push(Mode(size, stride))
    axis = axis + 1
  }
  if modes.length() == 1 {
    modes[0]
  } else {
    Tuple(modes)
  }
}

///|
/// Splits the leading mode into a tile of `tile` elements and the rest.
///
/// This is the operation a kernel uses to give each thread a block: the tile part addresses the
/// elements within one block, the rest picks the block. The two multiply back to the original size.
pub fn Layout::divide(
  self : Layout,
  tile~ : Int32,
) -> (Layout, Layout) raise LayoutError {
  let shape = self.shape()
  let strides = self.strides()
  if shape.length() == 0 {
    raise EmptyLayout
  }
  if tile <= 0 || shape[0] % tile != 0 {
    raise NotDivisible(tile~, size=shape[0])
  }
  let outer : Array[Layout] = [Mode(shape[0] / tile, strides[0] * tile)]
  let inner : Array[Layout] = [Mode(tile, strides[0])]
  for axis in 1..<shape.length() {
    outer.push(Mode(shape[axis], strides[axis]))
    inner.push(Mode(1, 0))
  }
  (Tuple(inner), Tuple(outer))
}

///|
/// What a layout operation refuses.
pub(all) suberror LayoutError {
  ///
  /// A coordinate does not have one entry per mode.
  RankMismatch(want~ : Int32, got~ : Int32)
  ///
  /// An index is outside the layout.
  IndexOutOfRange(index~ : Int32, size~ : Int32)
  ///
  /// A size that cannot be split by the tile asked for.
  NotDivisible(tile~ : Int32, size~ : Int32)
  ///
  /// The layout has no modes to split.
  EmptyLayout
}

///|
/// One line naming what the layout algebra refused.
pub fn LayoutError::message(self : LayoutError) -> String {
  match self {
    RankMismatch(want~, got~) =>
      "a coordinate of " +
      got.to_string() +
      " entries for " +
      want.to_string() +
      " modes"
    IndexOutOfRange(index~, size~) =>
      "index " +
      index.to_string() +
      " is outside a layout of " +
      size.to_string()
    NotDivisible(tile~, size~) =>
      "a size of " +
      size.to_string() +
      " does not split into tiles of " +
      tile.to_string()
    EmptyLayout => "the layout has no modes"
  }
}

///|
pub impl @debug.Debug for Layout with fn to_repr(self) {
  @debug.Repr::literal(describe(self))
}

///|
pub extend Layout with Show::{to_string}

///|
/// The layout in `size:stride` form, with modes nested in parentheses.
pub impl Show for Layout with fn to_string(self) {
  describe(self)
}

///|
/// `_2:_3` for the shape `[2]` and stride `[3]`, and `(_2:_1, _4:_2)` for a tuple.
fn describe(layout : Layout) -> String {
  match layout {
    Mode(size, stride) => "_" + size.to_string() + ":_" + stride.to_string()
    Tuple(parts) =>
      if parts.length() == 0 {
        "()"
      } else {
        "(" + parts.map(fn(part) { describe(part) }).join(", ") + ")"
      }
  }
}

///|
pub impl Eq for Layout with fn equal(self, other) {
  self.shape() == other.shape() && self.strides() == other.strides()
}
