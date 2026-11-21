---
name: moonvec
description: N-dimensional arrays for MoonBit with dtypes, broadcasting, universal functions, reductions and dense linear algebra. Use when you need to hold a numeric grid, slice or reshape it, apply a function elementwise, reduce along an axis, or solve a linear system.
---

# moonvec

```bash
moon add moonbitstack/moonvec
```

The array type and the elementwise functions come from `moonbitstack/moonvec`. Linear algebra lives
in `moonbitstack/moonvec/linalg`, and the layout algebra for tiling in
`moonbitstack/moonvec/cute`.

```moonbit
import {
  "moonbitstack/moonvec" @moonvec,
  "moonbitstack/moonvec/linalg" @linalg,
}
```

## Hold a grid and read it

```moonbit
let a = @moonvec.from_doubles([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
println(a.shape())
a.set_at([1, 2], @moonvec.Real(60.0))
println(a.at([1, 2]).to_string())
```

`zeros`, `ones`, `full`, `arange`, `linspace`, `eye`, `from_doubles`, `from_numbers`, `from_bytes` and
`matrix` make one. Every constructor that allocates takes `dtype~`; the default is `float64`.

## Apply a function and reduce it

```moonbit
let a = @moonvec.from_doubles([1.0, 2.0, 3.0, 4.0], [2, 2])
let scaled = @moonvec.multiply(a, a)
println(@moonvec.sum(scaled, axis=0).to_string())
println(@moonvec.sqrt(a).to_string())
```

The binary functions broadcast: a `[3, 1]` operand and a `[1, 4]` operand meet as `[3, 4]`. The
dtype of the answer is fixed by promotion before any element is read. `axis` is optional on every
reduction, and `keepdims=true` keeps the reduced axis with length 1.

## Solve a system

```moonbit
let a = @moonvec.matrix([[3.0, 1.0], [1.0, 2.0]])
let b = @moonvec.from_doubles([9.0, 8.0], [2])
let x = @linalg.solve(a, b)
println(x.to_string())
```

## Rules that catch people out

- A slice, a transpose and a reshape of a contiguous array are views over the same buffer. Writing
  through one changes the other. `copy`, `flatten`, `astype`, `take` and `mask` copy.
- Every function that can fail raises `VecError`; match on it or let it propagate. The operator forms
  (`+`, `-`, `*`, `/`, `%`) abort on a shape mismatch instead, because an operator cannot carry an
  error.
- Integer arithmetic wraps; dividing an integer by zero raises `VecError.Domain`.
- `where` is a keyword in MoonBit. The choice function is `select(condition, x, y)`.
- A reduction without an axis returns a 0-dimensional array; read it with `item()`.
- `matmul` and `dot` differ above two dimensions: `dot` contracts the last axis of the left operand
  with the second-to-last of the right one.
