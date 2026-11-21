// Copyright 2026 Leo Cheng
// SPDX-License-Identifier: Apache-2.0

///|
/// Runs one generated case: the operation name, with its parameters after colons, chooses the call.
fn run_vector_case(case : VectorCase) -> Array[NDArray] raise VecError {
  let pieces = case.op.split(":").collect().map(fn(piece) { piece.to_owned() })
  let operation = case.op.split(":").next().unwrap().to_owned()
  let parameters = pieces[1:].to_owned()
  let provided = case.inputs
  let axis = optional_axis(parameters)
  let keepdims = keeps_dims(parameters)
  let results : Array[NDArray] = []
  match operation {
    // Arithmetic, with both operand shapes broadcasting.
    "add" => results.push(add(operand(provided, 0), operand(provided, 1)))
    "subtract" =>
      results.push(subtract(operand(provided, 0), operand(provided, 1)))
    "multiply" =>
      results.push(multiply(operand(provided, 0), operand(provided, 1)))
    "divide" => results.push(divide(operand(provided, 0), operand(provided, 1)))
    "floor_divide" =>
      results.push(floor_divide(operand(provided, 0), operand(provided, 1)))
    "remainder" =>
      results.push(remainder(operand(provided, 0), operand(provided, 1)))
    "power" => results.push(power(operand(provided, 0), operand(provided, 1)))
    "maximum" =>
      results.push(maximum(operand(provided, 0), operand(provided, 1)))
    "minimum" =>
      results.push(minimum(operand(provided, 0), operand(provided, 1)))
    "arctan2" =>
      results.push(arctan2(operand(provided, 0), operand(provided, 1)))
    "hypot" => results.push(hypot(operand(provided, 0), operand(provided, 1)))
    // Comparisons.
    "equal" => results.push(equal(operand(provided, 0), operand(provided, 1)))
    "not_equal" =>
      results.push(not_equal(operand(provided, 0), operand(provided, 1)))
    "less" => results.push(less(operand(provided, 0), operand(provided, 1)))
    "less_equal" =>
      results.push(less_equal(operand(provided, 0), operand(provided, 1)))
    "greater" =>
      results.push(greater(operand(provided, 0), operand(provided, 1)))
    "greater_equal" =>
      results.push(greater_equal(operand(provided, 0), operand(provided, 1)))
    "logical_and" =>
      results.push(logical_and(operand(provided, 0), operand(provided, 1)))
    "logical_or" =>
      results.push(logical_or(operand(provided, 0), operand(provided, 1)))
    "logical_xor" =>
      results.push(logical_xor(operand(provided, 0), operand(provided, 1)))
    // Unary functions.
    "sqrt" => results.push(sqrt(operand(provided, 0)))
    "cbrt" => results.push(cbrt(operand(provided, 0)))
    "exp" => results.push(exp(operand(provided, 0)))
    "exp2" => results.push(exp2(operand(provided, 0)))
    "expm1" => results.push(expm1(operand(provided, 0)))
    "log" => results.push(log(operand(provided, 0)))
    "log2" => results.push(log2(operand(provided, 0)))
    "log10" => results.push(log10(operand(provided, 0)))
    "log1p" => results.push(log1p(operand(provided, 0)))
    "sin" => results.push(sin(operand(provided, 0)))
    "cos" => results.push(cos(operand(provided, 0)))
    "tan" => results.push(tan(operand(provided, 0)))
    "arcsin" => results.push(arcsin(operand(provided, 0)))
    "arccos" => results.push(arccos(operand(provided, 0)))
    "arctan" => results.push(arctan(operand(provided, 0)))
    "sinh" => results.push(sinh(operand(provided, 0)))
    "cosh" => results.push(cosh(operand(provided, 0)))
    "tanh" => results.push(tanh(operand(provided, 0)))
    "arcsinh" => results.push(arcsinh(operand(provided, 0)))
    "arccosh" => results.push(arccosh(operand(provided, 0)))
    "arctanh" => results.push(arctanh(operand(provided, 0)))
    "floor" => results.push(floor(operand(provided, 0)))
    "ceil" => results.push(ceil(operand(provided, 0)))
    "trunc" => results.push(trunc(operand(provided, 0)))
    "rint" => results.push(rint(operand(provided, 0)))
    "negative" => results.push(negative(operand(provided, 0)))
    "absolute" => results.push(absolute(operand(provided, 0)))
    "square" => results.push(square(operand(provided, 0)))
    "sign" => results.push(sign(operand(provided, 0)))
    "reciprocal" => results.push(reciprocal(operand(provided, 0)))
    "degrees" => results.push(degrees(operand(provided, 0)))
    "radians" => results.push(radians(operand(provided, 0)))
    "logical_not" => results.push(logical_not(operand(provided, 0)))
    "conjugate" => results.push(conj(operand(provided, 0)))
    "isnan" => results.push(isnan(operand(provided, 0)))
    "isinf" => results.push(isinf(operand(provided, 0)))
    "isfinite" => results.push(isfinite(operand(provided, 0)))
    "gamma" => results.push(gamma(operand(provided, 0)))
    "lgamma" => results.push(lgamma(operand(provided, 0)))
    "erf" => results.push(erf(operand(provided, 0)))
    "erfc" => results.push(erfc(operand(provided, 0)))
    // Reductions, with the axis and keepdims the generator chose.
    "sum" => results.push(sum(operand(provided, 0), axis?, keepdims~))
    "prod" => results.push(prod(operand(provided, 0), axis?, keepdims~))
    "min" => results.push(min(operand(provided, 0), axis?, keepdims~))
    "max" => results.push(max(operand(provided, 0), axis?, keepdims~))
    "mean" => results.push(mean(operand(provided, 0), axis?, keepdims~))
    "variance" => results.push(variance(operand(provided, 0), axis?, keepdims~))
    "std" => results.push(std(operand(provided, 0), axis?, keepdims~))
    "any" => results.push(any(operand(provided, 0), axis?, keepdims~))
    "all" => results.push(all(operand(provided, 0), axis?, keepdims~))
    "argmin" => results.push(argmin(operand(provided, 0), axis?, keepdims~))
    "argmax" => results.push(argmax(operand(provided, 0), axis?, keepdims~))
    // Linear algebra.
    "matmul" => results.push(matmul(operand(provided, 0), operand(provided, 1)))
    "dot" => results.push(dot(operand(provided, 0), operand(provided, 1)))
    "outer" => results.push(outer(operand(provided, 0), operand(provided, 1)))
    "inner" => results.push(inner(operand(provided, 0), operand(provided, 1)))
    "trace" => {
      let offset = int_setting(parameters, "offset", 0)
      results.push(trace(operand(provided, 0), offset~))
    }
    // Shape.
    "reshape" =>
      results.push(reshape(operand(provided, 0), axis_list(parameters[0])))
    "transpose" =>
      match string_setting(parameters, "axes") {
        Some(axes) =>
          results.push(transpose(operand(provided, 0), axes=axis_list(axes)))
        None => results.push(transpose(operand(provided, 0)))
      }
    "swap_axes" =>
      results.push(
        swap_axes(
          operand(provided, 0),
          parse_int(parameters[0]),
          parse_int(parameters[1]),
        ),
      )
    "move_axis" =>
      results.push(
        move_axis(
          operand(provided, 0),
          parse_int(parameters[0]),
          parse_int(parameters[1]),
        ),
      )
    "ravel" => results.push(ravel(operand(provided, 0)))
    "squeeze" =>
      match optional_axis(parameters) {
        Some(axis) => results.push(squeeze(operand(provided, 0), axis~))
        None => results.push(squeeze(operand(provided, 0)))
      }
    "expand_dims" =>
      results.push(
        expand_dims(operand(provided, 0), axis=parse_int(parameters[0])),
      )
    "concatenate" =>
      results.push(concatenate(provided, axis=parse_int(parameters[0])))
    "stack" => results.push(stack(provided, axis=parse_int(parameters[0])))
    "hstack" => results.push(hstack(provided))
    "vstack" => results.push(vstack(provided))
    "split_at" => {
      let cuts = parameters[0]
        .split("x")
        .collect()
        .map(fn(part) { parse_int(part) })
      for part in split_at(operand(provided, 0), cuts) {
        results.push(part)
      }
    }
    "slice" => results.push(slice(operand(provided, 0), index_list(parameters)))
    "take" => {
      let values = match
        (Ok(operand(provided, 1).to_numbers()) catch { _ => Err(()) }) {
        Ok(values) => values
        Err(_) => []
      }
      let picks = values.map(fn(value) { value.to_int64().to_int() })
      results.push(take(operand(provided, 0), picks, axis?))
    }
    "mask" =>
      results.push(
        mask(operand(provided, 0), boolean_list(operand(provided, 1))),
      )
    "sort" =>
      match optional_axis(parameters) {
        Some(axis) => results.push(sort(operand(provided, 0), axis~))
        None => results.push(sort(operand(provided, 0)))
      }
    "argsort" =>
      match optional_axis(parameters) {
        Some(axis) => results.push(argsort(operand(provided, 0), axis~))
        None => results.push(argsort(operand(provided, 0)))
      }
    "select" =>
      results.push(
        select(operand(provided, 0), operand(provided, 1), operand(provided, 2)),
      )
    "clip" => {
      let low = parse_double(parameters[0])
      let high = parse_double(parameters[1])
      results.push(clip(operand(provided, 0), min=low, max=high))
    }
    "cast" =>
      match DType::parse(parameters[0]) {
        Some(dtype) => results.push(astype(operand(provided, 0), dtype~))
        None => results.push(zeros([0]))
      }
    _ => results.push(zeros([0]))
  }
  results
}

///|
/// The named operand, or an empty array when the case does not have it, so that a malformed file is
/// reported as a difference rather than stopping the test.
fn operand(inputs : Array[NDArray], index : Int32) -> NDArray {
  if index < inputs.length() {
    inputs[index]
  } else {
    alloc([0], F64)
  }
}

///|
/// `axis=0` gives `Some(0)`, `axis=-` gives `None`, and no setting at all gives `None`.
fn optional_axis(parameters : Array[String]) -> Int32? {
  match string_setting(parameters, "axis") {
    Some(text) => if text == "-" { None } else { Some(parse_int(text)) }
    None => None
  }
}

///|
fn keeps_dims(parameters : Array[String]) -> Bool {
  int_setting(parameters, "keepdims", 0) == 1
}

///|
fn string_setting(parameters : Array[String], key : String) -> String? {
  let prefix = key + "="
  for parameter in parameters {
    if parameter.has_prefix(prefix) {
      return Some(parameter[prefix.length():].to_owned())
    }
  }
  None
}

///|
fn int_setting(
  parameters : Array[String],
  key : String,
  fallback : Int32,
) -> Int32 {
  match string_setting(parameters, key) {
    Some(text) => parse_int(text)
    None => fallback
  }
}

///|
/// `2x3` into `[2, 3]`, and `-` into the empty shape.
fn axis_list(text : String) -> Array[Int32] {
  if text == "-" {
    []
  } else {
    text.split("x").collect().map(fn(part) { parse_int(part) })
  }
}

///|
fn boolean_list(array : NDArray) -> Array[Bool] {
  let values = match (Ok(array.to_numbers()) catch { _ => Err(()) }) {
    Ok(values) => values
    Err(_) => []
  }
  values.map(fn(value) { value.to_bool() })
}

///|
/// The slicing parameters, read as a sequence of selectors.
fn index_list(parameters : Array[String]) -> Array[Index] {
  let indices : Array[Index] = []
  let mut i = 0
  while i < parameters.length() {
    let token = parameters[i]
    if token == "all" {
      indices.push(All)
      i = i + 1
    } else if token == "newaxis" {
      indices.push(NewAxis)
      i = i + 1
    } else if token == "at" {
      indices.push(At(parse_int(parameters[i + 1])))
      i = i + 2
    } else if token == "span" {
      indices.push(
        Span(
          parse_int(parameters[i + 1]),
          parse_int(parameters[i + 2]),
          parse_int(parameters[i + 3]),
        ),
      )
      i = i + 4
    } else {
      i = i + 1
    }
  }
  indices
}
