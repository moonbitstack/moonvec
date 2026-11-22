<div align="center">

# Numerical MoonBit Library

**moonvec — N-dimensional arrays for MoonBit: dtypes, broadcasting, universal functions and linear algebra.**

[Overview](https://moonbitstack.github.io/moonvec/) &nbsp;·&nbsp;
[Guide](https://moonbitstack.github.io/moonvec/guide.html) &nbsp;·&nbsp;
[API](https://moonbitstack.github.io/moonvec/api.html) &nbsp;·&nbsp;
[mooncakes](https://mooncakes.io/docs/moonbitstack/moonvec)

</div>

`moonvec` is a numerical array library. An array is a view over a flat buffer: a dtype, a shape, a
stride per axis and an offset. Slicing and transposing move those numbers instead of copying
elements, and every elementwise operation runs over broadcast shapes with the type promotion NumPy
uses.

- One array type over fourteen dtypes, including half precision and complex.
- Broadcasting for every elementwise function, with no temporary arrays.
- The reduction, shape, joining and sorting operations an array library is expected to have.
- Dense linear algebra: solving, determinants, Cholesky, QR, least squares, SVD and symmetric
  eigenvalues.
- A layout algebra for tiling, the description a kernel needs to know where its elements sit.
- A backend contract, so an elementwise loop can be replaced by a faster one without touching
  anything above it.

```bash
moon add moonbitstack/moonvec
```

```moonbit
let a = @moonvec.from_doubles([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
let b = @moonvec.transpose(a)
println(@moonvec.matmul(a, b).to_string())
```

```
array([[14.0, 32.0],
 [32.0, 77.0]], dtype=float64)
```

## Contents

- [Making an array](#making-an-array)
- [Reading and writing](#reading-and-writing)
- [Slicing](#slicing)
- [Universal functions](#universal-functions)
- [Reductions](#reductions)
- [Reshaping and joining](#reshaping-and-joining)
- [Sorting and choosing](#sorting-and-choosing)
- [Linear algebra](#linear-algebra)
- [Layouts](#layouts)
- [The interface table](#the-interface-table)
- [Rules that surprise people](#rules-that-surprise-people)
- [Consistency](#consistency)

## Making an array

Every constructor takes the dtype as a labelled argument; `float64` is the default, as it is in
NumPy.

```moonbit
let zeros = @moonvec.zeros([2, 3])
let ones = @moonvec.ones([2, 3], dtype=@moonvec.I32)
let filled = @moonvec.full([2, 2], @moonvec.Real(0.5), dtype=@moonvec.F32)
let stepped = @moonvec.arange(start=0.0, stop=1.0, step=0.25)
let spaced = @moonvec.linspace(start=0.0, stop=1.0, num=5)
let identity = @moonvec.eye(n=3, dtype=@moonvec.F64)
println(stepped.to_string())
println(spaced.to_string())
```

```
array([0.0, 0.25, 0.5, 0.75], dtype=float64)
array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float64)
```

`arange` counts the steps from the span, so `arange(start=0.0, stop=1.0, step=0.3)` has four elements
and stops just below 0.9. A step that points away from the stop gives an empty array rather than an
endless loop.

There is also `from_doubles` for a flat list with a shape, `from_numbers` when the values are already
typed, `from_bytes` for data read from a file, `matrix` for a list of rows, and the `*_like`
constructors that take their shape and dtype from an array you already have.

## Reading and writing

```moonbit
let a = @moonvec.from_doubles([1.0, 2.0, 3.0, 4.0], [2, 2])
a.set_at([1, 0], @moonvec.Real(9.0))
println(a.to_string())
println(a.at([1, 0]).to_string())
```

```
array([[1.0, 2.0],
 [9.0, 4.0]], dtype=float64)
9.0
```

`item()` reads the single value of a 0-dimensional array, which is what a full reduction returns.
`to_numbers()` and `to_doubles()` copy the elements out in row-major order, whatever strides the view
has.

## Slicing

A slice is a view: nothing is copied, and a write through the slice is a write into the original.

```moonbit
let a = @moonvec.from_doubles([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
let row = @moonvec.slice(a, [@moonvec.At(1)])
let column = @moonvec.slice(a, [@moonvec.All, @moonvec.At(2)])
let window = @moonvec.slice(a, [@moonvec.All, @moonvec.span(1, 3)])
println(row.to_string())
println(column.to_string())
println(window.is_contiguous())
```

```
array([4.0, 5.0, 6.0], dtype=float64)
array([3.0, 6.0], dtype=float64)
false
```

`At` removes an axis, `NewAxis` adds one, and `span(start, stop, step)` walks an axis forwards or
backwards with the bounds clamped the way NumPy clamps them. Integer array indexing and boolean masks
cannot be expressed by strides, so `take` and `mask` copy:

```moonbit
let a = @moonvec.arange(start=0.0, stop=6.0)
let picked = @moonvec.take(a, [4, 0, 2])
let kept = @moonvec.mask(a, [true, false, true, false, false, true])
println(picked.to_string())
println(kept.to_string())
```

```
array([4.0, 0.0, 2.0], dtype=float64)
array([0.0, 2.0, 5.0], dtype=float64)
```

## Universal functions

An elementwise function broadcasts its operands, promotes their dtypes and writes into one new array.
The binary ones have operator forms as well.

```moonbit
let column = @moonvec.from_doubles([1.0, 2.0, 3.0], [3, 1])
let row = @moonvec.from_doubles([10.0, 20.0, 30.0], [1, 3])
println((column + row).to_string())
let counts = @moonvec.full([3], @moonvec.Signed(7L), dtype=@moonvec.I32)
println(@moonvec.divide(counts, counts).to_string())
```

```
array([[11.0, 21.0, 31.0],
 [12.0, 22.0, 32.0],
 [13.0, 23.0, 33.0]], dtype=float64)
array([1.0, 1.0, 1.0], dtype=float64)
```

The operator forms (`+`, `-`, `*`, `/`, `%` and unary `-`) abort on a shape mismatch, because MoonBit
does not let an operator carry an error. The named functions raise `VecError`, which carries both
shapes in the message.

Arithmetic keeps integers exact and wraps on overflow. The division family and `power` promote
booleans to `int8`, as NumPy does. Everything else — trigonometric, exponential, rounding, the special
functions — answers in floating point.

```moonbit
let angles = @moonvec.linspace(start=0.0, stop=3.141592653589793, num=3)
println(@moonvec.sin(angles).to_string())
println(@moonvec.erf(@moonvec.from_doubles([0.0, 1.0, 2.0], [3])).to_string())
```

```
array([0.0, 1.0, 1.2246467991473532e-16], dtype=float64)
array([0.0, 0.8427007929497147, 0.9953222650189527], dtype=float64)
```

## Reductions

Each reduction takes an optional `axis` and a `keepdims` switch. Without an axis the whole array is
reduced and the answer is 0-dimensional.

```moonbit
let a = @moonvec.from_doubles([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
println(@moonvec.sum(a, axis=0).to_string())
println(@moonvec.max(a, axis=1, keepdims=true).to_string())
println(@moonvec.argmax(a).to_string())
println(@moonvec.mean(a).to_string())
```

```
array([5.0, 7.0, 9.0], dtype=float64)
array([[3.0],
 [6.0]], dtype=float64)
array(5, dtype=int64)
array(3.5, dtype=float64)
```

`sum`, `prod`, `min`, `max`, `mean`, `variance`, `std`, `any`, `all`, `argmin` and `argmax` are the
whole set. A sum of booleans or of a narrow integer counts in `int64`, which is what NumPy does; a mean
is `float64` for exact input and keeps the width of a float one. `variance` and `std` take `ddof`.

## Reshaping and joining

```moonbit
let a = @moonvec.arange(start=0.0, stop=6.0)
println(@moonvec.reshape(a, [2, 3]).to_string())
println(@moonvec.reshape(a, [-1, 2]).shape().map(fn(n) { n.to_string() }).join(" by "))
let rows = @moonvec.stack([a, a], axis=0)
println(rows.shape().map(fn(n) { n.to_string() }).join(" by "))
for part in @moonvec.split(rows, parts=2, axis=1) {
  println(part.shape().map(fn(n) { n.to_string() }).join(" by "))
}
```

```
array([[0.0, 1.0, 2.0],
 [3.0, 4.0, 5.0]], dtype=float64)
3 by 2
2 by 6
2 by 3
2 by 3
```

`reshape` gives a view when the array is contiguous and copies when it is not; `ravel` and `flatten`
differ in the same way. `transpose`, `swap_axes`, `move_axis`, `squeeze` and `expand_dims` are views,
and `concatenate`, `stack`, `hstack`, `vstack`, `split` and `split_at` allocate.

## Sorting and choosing

```moonbit
let a = @moonvec.from_doubles([3.0, 1.0, 2.0, 5.0, 4.0], [5])
println(@moonvec.sort(a).to_string())
println(@moonvec.argsort(a).to_string())
let values = @moonvec.from_doubles([-2.0, 0.5, 3.0], [3])
let zeros = @moonvec.zeros([3])
println(@moonvec.select(@moonvec.greater(values, zeros), values, zeros).to_string())
println(@moonvec.clip(values, min=0.0, max=1.0).to_string())
```

```
array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float64)
array([1, 2, 0, 4, 3], dtype=int64)
array([0.0, 0.5, 3.0], dtype=float64)
array([0.0, 0.5, 1.0], dtype=float64)
```

Both sorts are stable and put NaNs last. `where` is a keyword in MoonBit, so the choice function is
called `select`.

## Linear algebra

`matmul` contracts the last axis of one array with the second-to-last of the other, broadcasts the
leading axes, and drops the axes a 1-D operand borrowed. `dot`, `inner`, `outer` and `trace` are the
companions.

```moonbit
let a = @moonvec.matrix([[3.0, 1.0], [1.0, 2.0]])
let b = @moonvec.from_doubles([9.0, 8.0], [2])
println(@linalg.solve(a, b).to_string())
println(@moonvec.format_real(@linalg.det(a)))
```

```
array([2.0, 3.0], dtype=float64)
5.0
```

The `linalg` package holds `solve`, `inv`, `det`, `cholesky`, `qr`, `lstsq`, `svd`, `eigh`,
`matrix_power`, `kron`, `norm` and `matrix_norm`. Factorisations run in double precision and store
their answer in the caller's dtype. `lstsq` returns the shortest solution when there are fewer rows
than columns, and refuses a matrix whose columns are linearly dependent.

## Layouts

The `cute` package is the layout algebra: shapes and strides that nest, with the tiling operations a
kernel needs. It holds no data, so an array view is built from one.

```moonbit
let layout = @cute.compact([2, 3])
println(layout.to_string())
println(layout.offset([1, 2]))
let (tile, tiles) = @cute.compact([6]).divide(tile=3)
println(tile.to_string())
println(tiles.to_string())
```

```
(_2:_3, _3:_1)
5
(_3:_1)
(_2:_3)
```

`coalesce` merges modes that walk without a gap, `product` combines a thread layout with a value
layout, and `strides()` hands the result to `NDArray::new`.

## The interface table

The root package `moonbitstack/moonvec`:

| Area | Interface |
|:--|:--|
| dtype | `DType` with `Bool`, `I8`, `I16`, `I32`, `I64`, `U8`, `U16`, `U32`, `U64`, `F16`, `F32`, `F64`, `C64`, `C128`; `item_size`, `name`, `parse`, `is_bool`, `is_integer`, `is_signed`, `is_float`, `is_complex`, `is_real`, `is_exact`, `to_real`, `float_size` |
| promotion | `promote`, `promote_all`, `division_dtype` |
| values | `Number` (`Boolean`, `Signed`, `Unsigned`, `Real`, `Complex`), `Complex`, `format_real` |
| buffer | `Buffer` (`new`, `from_bytes`, `dtype`, `len`, `read`, `write`, `fill`, `to_bytes`) |
| array | `NDArray` (`new`, `shape`, `strides`, `offset`, `ndim`, `size`, `dtype`, `buffer`, `is_contiguous`, `is_scalar`, `at`, `set_at`, `item`, `to_numbers`, `to_doubles`, `copy`, `to_string`) |
| creation | `zeros`, `ones`, `full`, `zeros_like`, `ones_like`, `full_like`, `arange`, `linspace`, `eye`, `from_doubles`, `from_numbers`, `from_bytes`, `array`, `matrix` |
| casting | `astype` |
| indexing | `Index` (`All`, `At`, `Span`, `NewAxis`, `Take`, `Mask`), `span`, `slice`, `take`, `mask` |
| broadcasting | `broadcast`, `broadcast_to`, `broadcast_shape`, `broadcast_strides` |
| arithmetic | `add`, `subtract`, `multiply`, `divide`, `floor_divide`, `remainder`, `mod`, `power`, `maximum`, `minimum`, `arctan2`, `hypot`, `bitwise_and`, `bitwise_or`, `bitwise_xor`, `left_shift`, `right_shift`, `invert`, `negative`, `positive`, `absolute`, `fabs`, `sign`, `square`, `reciprocal`, `clip`, `select`, and the scalar forms `add_scalar`, `subtract_scalar`, `multiply_scalar`, `divide_scalar`, `power_scalar`, `maximum_scalar`, `minimum_scalar` |
| comparisons | `equal`, `not_equal`, `less`, `less_equal`, `greater`, `greater_equal`, `equal_nan`, `logical_and`, `logical_or`, `logical_xor`, `logical_not`, `isnan`, `isinf`, `isfinite`, `signbit` |
| mathematics | `sqrt`, `cbrt`, `exp`, `exp2`, `expm1`, `log`, `log2`, `log10`, `log1p`, `sin`, `cos`, `tan`, `arcsin`, `arccos`, `arctan`, `sinh`, `cosh`, `tanh`, `arcsinh`, `arccosh`, `arctanh`, `floor`, `ceil`, `trunc`, `rint`, `round`, `degrees`, `radians`, `real`, `imag`, `angle`, `conj` |
| special functions | `gamma`, `lgamma`, `digamma`, `erf`, `erfc`, `erfinv`, `erfcinv`, `beta`, `zeta`, `bessel_j`, `bessel_y`, `bessel_i`, `bessel_k` |
| reductions | `sum`, `prod`, `min`, `max`, `mean`, `variance`, `std`, `any`, `all`, `argmin`, `argmax` |
| shape | `reshape`, `transpose`, `swap_axes`, `move_axis`, `ravel`, `flatten`, `squeeze`, `expand_dims` |
| joining | `concatenate`, `stack`, `hstack`, `vstack`, `split`, `split_at` |
| ordering | `sort`, `argsort` |
| products | `matmul`, `dot`, `inner`, `outer`, `trace` |
| backend | `Kernel` (`name`, `binary`, `unary`, `reduce`, `matmul`), `Cpu`, `cpu`, `BinaryOp`, `UnaryOp`, `ReduceOp` |
| errors | `VecError` (`ByteCountMismatch`, `IndexOutOfBounds`, `ShapeMismatch`, `AxisOutOfRange`, `CannotReshape`, `EmptyAxis`, `Domain`, `NotSolvable`), `message` |

`moonbitstack/moonvec/linalg`:

| Area | Interface |
|:--|:--|
| solving | `solve`, `inv`, `det`, `cholesky` |
| factoring | `qr`, `lstsq`, `svd`, `eigh`, `matrix_power`, `kron` |
| measuring | `Norm` (`Frobenius`, `One`, `Two`, `Infinity`, `NegInfinity`, `P`), `norm`, `matrix_norm` |

`moonbitstack/moonvec/cute`:

| Area | Interface |
|:--|:--|
| layout | `Layout` (`Mode`, `Tuple`), `mode`, `compact`, `product` |
| measuring | `size`, `cosize`, `shape`, `strides`, `offset`, `offset_of_index` |
| algebra | `coalesce`, `divide` |
| errors | `LayoutError` (`RankMismatch`, `IndexOutOfRange`, `NotDivisible`, `EmptyLayout`), `message` |

## Rules that surprise people

**A view shares its buffer.** `slice`, `transpose`, `reshape` on a contiguous array, `squeeze`,
`expand_dims` and `broadcast_to` return views. Writing through one is visible through the other.
`copy`, `flatten`, `astype`, `take`, `mask`, `sort` and every arithmetic function allocate.

**The dtype of an operation is fixed before any element is read.** Promotion follows a table that is
transcribed from NumPy and checked cell by cell against it. `uint64` with `int8` gives `float64`,
because no integer holds both ranges.

**Arithmetic wraps, division does not.** Adding two `int8` arrays wraps at 128, as C does. Dividing an
integer by zero is refused with `Domain` rather than answered with zero, because there is no honest
integer answer.

**NaNs propagate and sort last.** `maximum` and `minimum` propagate a NaN. A comparison with a NaN is
false, and `equal_nan` is there for when you want the other answer. A sort puts NaN at the end.

**Complex numbers are ordered lexicographically.** `less`, `greater` and their neighbours compare the
real part first and the imaginary part when the real parts agree, which is what NumPy does.

## Consistency

The behaviour of this library is compared against NumPy 2.5.4 on fixed inputs.
`scripts/gen_vectors.py` writes the inputs and NumPy's answers as raw little-endian bytes; the test
suite replays them and compares element by element, including the dtype of the answer.

| Group | Cases | Passing |
|:--|--:|--:|
| dtype promotion, every pair of the fourteen dtypes | 196 | 196 |
| arithmetic, comparison and logical operations over 24 dtype pairs and 10 shape pairs | 1 267 | 1 267 |
| mathematics over float and complex values, including the edges | 106 | 106 |
| reductions with every axis and `keepdims`, over five dtypes and two ranks | 880 | 880 |
| products, shape, joining, slicing, sorting, choice and casting | 110 | 110 |
| linear algebra: solving, determinants, Cholesky, least squares, norms, powers, products | 29 | 29 |
| **total** | **2 588** | **2 588** |

Where the answers may differ in the last bits — a variance is a sum of squares, a complex power goes
through a logarithm — the vector file records a relative tolerance and the test uses it. Every integer
answer, and every elementwise `+`, `-`, `*` and `/`, must match exactly.

One deliberate difference: integer division by zero is refused rather than answered with zero.

## Licence

Apache-2.0.
