// Copyright 2026 Leo Cheng
// SPDX-License-Identifier: Apache-2.0

///|
/// The page drives the library through these two entry points; everything it draws is computed here
/// and nowhere else, so a wrong answer on the page is a wrong answer in the library.
extern "js" fn publish_broadcast(f : (String) -> String) -> Unit =
  #| (f) => { (globalThis.moonvecDemo ??= {}).broadcast = f }

///|
extern "js" fn publish_fit(f : (String) -> String) -> Unit =
  #| (f) => { (globalThis.moonvecDemo ??= {}).fit = f }

///|
/// `2,3|1 2 3 4 5 6|3|10 20 30` broadcasts the two operands and answers `ok|2,3|11 12 ...`; a shape
/// that cannot broadcast answers `err|` and the library's own message.
pub fn broadcast(input : String) -> String {
  let outcome : Result[String, @vec.VecError] = Ok(run_broadcast(input)) catch {
    error => Err(error)
  }
  match outcome {
    Ok(text) => "ok|" + text
    Err(error) => "err|" + error.message()
  }
}

///|
fn run_broadcast(input : String) -> String raise @vec.VecError {
  let parts = input.split("|").collect()
  if parts.length() != 4 {
    raise @vec.Domain(message="expected shape, values, shape, values")
  }
  let left = @vec.from_doubles(parse_values(parts[1]), parse_shape(parts[0]))
  let right = @vec.from_doubles(parse_values(parts[3]), parse_shape(parts[2]))
  let total = @vec.add(left, right)
  let shape = total.shape()
  let mut header = ""
  for i in 0..<shape.length() {
    if i > 0 {
      header = header + ","
    }
    header = header + shape[i].to_string()
  }
  let values = total.to_doubles()
  let mut body = ""
  for i in 0..<values.length() {
    if i > 0 {
      body = body + " "
    }
    body = body + @vec.format_real(values[i])
  }
  header + "|" + body
}

///|
/// `1,2;2,3.4;3,5.9` fits a line by least squares and answers `ok|slope|intercept|residual`.
pub fn fit(input : String) -> String {
  let outcome : Result[String, @vec.VecError] = Ok(run_fit(input)) catch {
    error => Err(error)
  }
  match outcome {
    Ok(text) => "ok|" + text
    Err(error) => "err|" + error.message()
  }
}

///|
fn run_fit(input : String) -> String raise @vec.VecError {
  let points = input.split(";").collect()
  if points.length() < 2 {
    raise @vec.Domain(message="a line needs at least two points")
  }
  let xs : Array[Double] = []
  let ys : Array[Double] = []
  for point in points {
    let pair = point.split(",").collect()
    if pair.length() != 2 {
      raise @vec.Domain(message="a point is an x and a y")
    }
    xs.push(parse_double(pair[0]))
    ys.push(parse_double(pair[1]))
  }
  let count = xs.length()
  let design = @vec.from_doubles(intercalate(xs.map(fn(_x) { 1.0 }), xs), [
    count, 2,
  ])
  let target = @vec.from_doubles(ys, [count])
  let model = @linalg.lstsq(design, target)
  let intercept = model.at([0]).to_double()
  let slope = model.at([1]).to_double()
  let predicted = @vec.add(
    @vec.multiply(
      @vec.from_doubles(xs, [count]),
      @vec.full([count], @vec.Real(slope)),
    ),
    @vec.full([count], @vec.Real(intercept)),
  )
  let residual = @vec.subtract(target, predicted)
  let spread = @vec.sum(@vec.square(residual))
  @vec.format_real(slope) +
  "|" +
  @vec.format_real(intercept) +
  "|" +
  @vec.format_real(spread.item().to_double())
}

///|
/// The rows of the design matrix alternate between the constant column and the x values.
fn intercalate(ones : Array[Double], xs : Array[Double]) -> Array[Double] {
  let out : Array[Double] = []
  for i in 0..<ones.length() {
    out.push(ones[i])
    out.push(xs[i])
  }
  out
}

///|
fn parse_shape(text : StringView) -> Array[Int32] {
  let dims : Array[Int32] = []
  for part in text.split(",") {
    dims.push(parse_int(part))
  }
  dims
}

///|
fn parse_values(text : StringView) -> Array[Double] {
  let values : Array[Double] = []
  for part in text.split(" ") {
    if !part.is_empty() {
      values.push(parse_double(part))
    }
  }
  values
}

///|
fn parse_int(text : StringView) -> Int32 {
  let mut value : Int32 = 0
  for i in 0..<text.length() {
    let code = text.code_unit_at(i).to_int()
    if code >= 48 && code <= 57 {
      value = value * 10 + (code - 48)
    }
  }
  value
}

///|
fn parse_double(text : StringView) -> Double {
  let mut value = 0.0
  let mut scale = 0.1
  let mut after_point = false
  let mut negative = false
  for i in 0..<text.length() {
    let code = text.code_unit_at(i).to_int()
    if code == 45 {
      negative = true
    } else if code == 46 {
      after_point = true
    } else if code >= 48 && code <= 57 {
      let digit = (code - 48).to_double()
      if after_point {
        value = value + digit * scale
        scale = scale * 0.1
      } else {
        value = value * 10.0 + digit
      }
    }
  }
  if negative {
    -value
  } else {
    value
  }
}

///|
fn main {
  publish_broadcast(broadcast)
  publish_fit(fit)
}
