// Copyright 2026 Leo Cheng
// SPDX-License-Identifier: Apache-2.0

///|
/// How many elements an array may have before the printed form elides the middle of each axis.
let print_limit : Int32 = 1000

///|
/// How many elements are shown at each end of an elided axis.
let print_edge : Int32 = 3

///|
/// The array laid out one row per line, with the dtype after the values.
pub fn NDArray::to_string(self : NDArray) -> String raise VecError {
  let mut text = "array("
  text = text + render(self)
  text = text + ", dtype=" + self.dtype().name() + ")"
  text
}

///|
fn render(array : NDArray) -> String raise VecError {
  let elide = array.size() > print_limit
  render_at(array, 0, array.offset(), "", elide)
}

///|
/// Renders the sub-array starting at `offset` whose remaining axes start at `axis`.
fn render_at(
  array : NDArray,
  axis : Int32,
  offset : Int32,
  indent : String,
  elide : Bool,
) -> String raise VecError {
  let ndim = array.ndim()
  if ndim == 0 {
    return array.buffer.read(offset).to_string()
  }
  let shape = array.shape()
  let strides = array.strides()
  if axis == ndim - 1 {
    let length = shape[axis]
    let shown = shown_positions(length, elide)
    let parts : Array[String] = []
    for position in shown {
      match position {
        Some(position) =>
          parts.push(
            array.buffer.read(offset + position * strides[axis]).to_string(),
          )
        None => parts.push("...")
      }
    }
    "[" + parts.join(", ") + "]"
  } else {
    let length = shape[axis]
    let shown = shown_positions(length, elide)
    let parts : Array[String] = []
    for position in shown {
      match position {
        Some(position) =>
          parts.push(
            render_at(
              array,
              axis + 1,
              offset + position * strides[axis],
              indent + " ",
              elide,
            ),
          )
        None => parts.push("...")
      }
    }
    "[" + parts.join(",\n" + indent + " ") + "]"
  }
}

///|
/// The positions an axis shows: all of them, or the first and last few with a gap in between.
fn shown_positions(length : Int32, elide : Bool) -> Array[Int32?] {
  let out : Array[Int32?] = []
  if !elide || length <= 2 * print_edge {
    for i in 0..<length {
      out.push(Some(i))
    }
  } else {
    for i in 0..<print_edge {
      out.push(Some(i))
    }
    out.push(None)
    for i in (length - print_edge)..<length {
      out.push(Some(i))
    }
  }
  out
}

///|
pub impl Show for NDArray with fn to_string(self) {
  match (Ok(self.to_string()) catch { _ => Err(()) }) {
    Ok(text) => text
    Err(_) => "array(<unreadable>)"
  }
}

///|
pub impl @debug.Debug for NDArray with fn to_repr(self) {
  @debug.Repr::literal(
    match (Ok(self.to_string()) catch { _ => Err(()) }) {
      Ok(text) => text
      Err(_) => "array(<unreadable>)"
    },
  )
}

///|
/// Two arrays are equal when their shapes, dtypes and elements agree, with NaN equal to itself so
/// that a test can compare two arrays that both hold a NaN in the same place.
pub impl Eq for NDArray with fn equal(self, other) {
  if self.shape() != other.shape() || self.dtype() != other.dtype() {
    return false
  }
  let left = match (Ok(self.to_numbers()) catch { _ => Err(()) }) {
    Ok(values) => values
    Err(_) => return false
  }
  let right = match (Ok(other.to_numbers()) catch { _ => Err(()) }) {
    Ok(values) => values
    Err(_) => return false
  }
  for i in 0..<left.length() {
    if !numbers_identical(left[i], right[i]) {
      return false
    }
  }
  true
}

///|
/// Equality that treats a NaN as equal to a NaN and `-0.0` as equal to `0.0`, which is what a test
/// comparing a computed array with an expected one wants.
fn numbers_identical(left : Number, right : Number) -> Bool {
  let a = left.to_double()
  let b = right.to_double()
  if a.is_nan() && b.is_nan() {
    return true
  }
  match (left, right) {
    (Complex(x), Complex(y)) => x.re == y.re && x.im == y.im
    _ => a == b
  }
}
