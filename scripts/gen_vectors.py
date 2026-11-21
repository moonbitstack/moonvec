"""Generate the vectors the test suite replays against NumPy.

The library's own tests say what the author believed; this file says what NumPy actually does. Every
case here is a fixed input and the answer NumPy gives, written as raw little-endian bytes so that a
value round-trips exactly, including the sign of a zero and the payload of a NaN.

Run with `python scripts/gen_vectors.py`; the output is checked in and CI only replays it, so the
build never needs NumPy. Regeneration is deliberate rather than a CI step: NumPy picks its elementwise
loops by CPU feature at run time, so the last bit of a transcendental answer is not the same on every
machine, and a byte comparison across platforms would fail for reasons that are not about this
library.

Format, one record per case:

    case <name>
    op <operation>            # the operation, with parameters after colons
    tol <relative tolerance>  # optional, 0 when absent, means the answer must match exactly
    in <dtype> <dims> <hex>   # one line per operand; <dims> is `2x3` or `-` for 0-dimensional
    out <dtype> <dims> <hex>
    end

Usage: python scripts/gen_vectors.py [--out src/vectors.txt]
"""

import argparse
import math

import numpy as np

DTYPES = [
    "bool",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "float32",
    "float64",
    "complex64",
    "complex128",
]

# Pairs that between them cross every boundary in the promotion table: same widths, signed with
# unsigned, an integer that grows into a wider one, and each family meeting the floats.
PAIRS = [
    ("int8", "int8"),
    ("int8", "int16"),
    ("int8", "uint8"),
    ("int16", "uint16"),
    ("int32", "uint32"),
    ("int32", "int64"),
    ("int64", "uint64"),
    ("uint8", "uint16"),
    ("uint8", "int64"),
    ("uint64", "int8"),
    ("float16", "float16"),
    ("float16", "float32"),
    ("float32", "float32"),
    ("float32", "int32"),
    ("float64", "float64"),
    ("float64", "int8"),
    ("float64", "uint64"),
    ("complex64", "complex64"),
    ("complex64", "float32"),
    ("complex128", "complex128"),
    ("complex128", "float64"),
    ("bool", "bool"),
    ("bool", "int8"),
    ("bool", "float32"),
]

# Shapes that meet as one operand and a second: the second broadcasts up to the first, or the two
# meet in the middle. Sizes stay small so the file stays readable.
SHAPE_PAIRS = [
    ((2, 3), (2, 3)),
    ((2, 3), (3,)),
    ((2, 3), (1, 3)),
    ((2, 3), (2, 1)),
    ((2, 1, 3), (4, 3)),
    ((3,), (1,)),
    ((), (2, 2)),
    ((4, 1, 2), (1, 3, 1)),
    ((0, 3), (3,)),
    ((2, 0, 3), (1, 1, 3)),
]

CASES = []
SEED = 20261121


def encode(array):
    """The array's bytes as hex, little endian and in a fixed order."""
    array = np.ascontiguousarray(array)
    return array.astype(array.dtype.newbyteorder("<"), copy=False).tobytes().hex() or "-"


def dims(array):
    return "x".join(str(n) for n in array.shape) or "-"


def values_for(dtype, size, rng):
    """A deterministic set of values with the awkward ones included by hand."""
    kind = np.dtype(dtype).kind
    if dtype == "bool":
        return (rng.integers(0, 2, size=size) == 1)
    if kind == "i":
        low, high = -9, 9
        if np.dtype(dtype).itemsize == 1:
            low, high = -6, 6
        return rng.integers(low, high + 1, size=size).astype(dtype)
    if kind == "u":
        return rng.integers(0, 12, size=size).astype(dtype)
    if dtype == "float16":
        return (rng.integers(-40, 41, size=size) / 8.0).astype(dtype)
    if kind == "f":
        raw = rng.integers(-600, 601, size=size) / 64.0
        return raw.astype(dtype)
    real = rng.integers(-300, 301, size=size) / 32.0
    imaginary = rng.integers(-300, 301, size=size) / 32.0
    return (real + 1j * imaginary).astype(dtype)


def record(name, op, operands, out, tol=0.0):
    CASES.append(
        {
            "name": name,
            "op": op,
            "tol": tol,
            "in": operands,
            "out": out,
        }
    )


def binary_op(name, op, a, b):
    with np.errstate(all="ignore"):
        return op(a, b)


def generate_promotions():
    rows = []
    for left in DTYPES:
        for right in DTYPES:
            result = np.result_type(np.dtype(left), np.dtype(right)).name
            rows.append((left, right, result))
    return rows


def generate_arithmetic(rng):
    """The binary operations, over every dtype pair and every shape pair."""
    operations = {
        "add": np.add,
        "subtract": np.subtract,
        "multiply": np.multiply,
        "divide": np.true_divide,
        "floor_divide": np.floor_divide,
        "remainder": np.remainder,
        "power": np.power,
        "maximum": np.maximum,
        "minimum": np.minimum,
        "equal": np.equal,
        "not_equal": np.not_equal,
        "less": np.less,
        "less_equal": np.less_equal,
        "greater": np.greater,
        "greater_equal": np.greater_equal,
        "logical_and": np.logical_and,
        "logical_or": np.logical_or,
        "logical_xor": np.logical_xor,
    }
    for op_index, (name, operation) in enumerate(operations.items()):
        for pair_index, (left, right) in enumerate(PAIRS):
            # Three of the ten shape pairs per operation and dtype pair, taken in a rotation so that
            # every shape is covered several times over the whole file without repeating each one
            # eighteen times.
            chosen = [
                SHAPE_PAIRS[(op_index + pair_index + step) % len(SHAPE_PAIRS)] for step in (0, 3, 7)
            ]
            for shape_a, shape_b in chosen:
                a = values_for(left, int(np.prod(shape_a)) if shape_a else 1, rng).reshape(shape_a)
                b = values_for(right, int(np.prod(shape_b)) if shape_b else 1, rng).reshape(shape_b)
                if name == "power" and np.dtype(left).kind in "iu":
                    b = np.abs(b).astype(right)
                if name in ("floor_divide", "remainder") and np.dtype(right).kind in "iub":
                    b = np.where(b == 0, 1, b).astype(right)
                complex_involved = np.dtype(left).kind == "c" or np.dtype(right).kind == "c"
                try:
                    result = binary_op(name, operation, a, b)
                except (TypeError, ValueError):
                    # NumPy refuses a few pairs, such as subtracting two booleans; the library
                    # refuses them too, and the test for that lives with the operation.
                    continue
                # A power is exp(y log x) and rounds differently in each library; a complex product
                # is computed here in double and rounded once, which is one bit away from NumPy's
                # single-precision path.
                wide = np.dtype(result.dtype).itemsize
                tolerance = 0.0
                if name == "power":
                    tolerance = 1e-5 if wide <= 8 else 1e-12
                elif complex_involved:
                    tolerance = 1e-5 if wide <= 8 else 1e-12
                record(
                    f"{name}/{left}/{right}/{dims(a)}/{dims(b)}",
                    name,
                    [a, b],
                    result,
                    tolerance,
                )


def generate_math(rng):
    """The unary functions, over values that reach their edges."""
    points = np.array(
        [
            -3.5,
            -2.0,
            -1.0,
            -0.5,
            -0.0,
            0.0,
            0.5,
            1.0,
            2.0,
            3.5,
            9.0,
            np.nan,
            np.inf,
            -np.inf,
        ]
    )
    functions = {
        "sqrt": np.sqrt,
        "cbrt": np.cbrt,
        "exp": np.exp,
        "exp2": np.exp2,
        "expm1": np.expm1,
        "log": np.log,
        "log2": np.log2,
        "log10": np.log10,
        "log1p": np.log1p,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "arcsin": np.arcsin,
        "arccos": np.arccos,
        "arctan": np.arctan,
        "sinh": np.sinh,
        "cosh": np.cosh,
        "tanh": np.tanh,
        "arcsinh": np.arcsinh,
        "arccosh": np.arccosh,
        "arctanh": np.arctanh,
        "floor": np.floor,
        "ceil": np.ceil,
        "trunc": np.trunc,
        "rint": np.rint,
        "negative": np.negative,
        "absolute": np.absolute,
        "square": np.square,
        "sign": np.sign,
        "reciprocal": np.reciprocal,
        "degrees": np.degrees,
        "radians": np.radians,
        "logical_not": np.logical_not,
        "conjugate": np.conjugate,
        "isnan": np.isnan,
        "isinf": np.isinf,
        "isfinite": np.isfinite,
        "gamma": None,
        "lgamma": None,
        "erf": None,
        "erfc": None,
    }
    special_points = np.array([0.25, 0.5, 1.0, 1.5, 2.0, 3.5, 9.0, np.nan, np.inf])

    def apply_special(function):
        def wrapped(values):
            out = []
            for value in values:
                try:
                    out.append(function(float(value)))
                except (ValueError, OverflowError):
                    out.append(np.nan)
            return np.array(out, dtype="float64")
        return wrapped

    special = {
        "gamma": apply_special(math.gamma),
        "lgamma": apply_special(math.lgamma),
        "erf": apply_special(math.erf),
        "erfc": apply_special(math.erfc),
    }
    for name, operation in functions.items():
        for dtype in ("float32", "float64"):
            data = (special_points if operation is None else points).astype(dtype)
            with np.errstate(all="ignore"):
                if operation is None:
                    result = special[name](data.astype("float64")).astype(dtype)
                else:
                    result = operation(data)
            exact = name in ("floor", "ceil", "trunc", "rint", "negative", "absolute", "square",
                             "sign", "logical_not", "conjugate", "isnan", "isinf", "isfinite")
            record(
                f"{name}/{dtype}",
                name,
                [data],
                np.asarray(result),
                tol=0.0 if exact else 1e-6,
            )
    # Two complex arguments, where the branch cut is what differs between implementations.
    complex_points = np.array([1 + 2j, -1 + 0.5j, 0 - 1j, 3 + 0j, -2 - 2j])
    for name, operation in {"sqrt": np.sqrt, "exp": np.exp, "log": np.log, "sin": np.sin,
                            "cos": np.cos, "tan": np.tan, "sinh": np.sinh, "cosh": np.cosh,
                            "tanh": np.tanh, "conjugate": np.conjugate}.items():
        for dtype in ("complex64", "complex128"):
            data = complex_points.astype(dtype)
            with np.errstate(all="ignore"):
                result = operation(data)
            record(f"{name}/{dtype}", name, [data], result, tol=1e-5)


def generate_binary_math(rng):
    a = np.array([-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, np.nan, np.inf])
    b = np.array([1.0, 2.0, 0.0, -1.0, -2.0, 0.5, 1.0, 2.0])
    for name, operation in {"arctan2": np.arctan2, "hypot": np.hypot}.items():
        for dtype in ("float32", "float64"):
            record(f"{name}/{dtype}", name, [a.astype(dtype), b.astype(dtype)], operation(a, b).astype(dtype),
                   tol=1e-6)


def generate_reductions(rng):
    """Each reduction over a 2-D and a 3-D array, with and without an axis and keepdims."""
    shapes = [(3, 4), (2, 3, 2)]
    dtypes = ["float64", "int64", "int32", "bool", "float32"]
    names = {
        "sum": np.sum,
        "prod": np.prod,
        "min": np.min,
        "max": np.max,
        "mean": np.mean,
        "variance": np.var,
        "std": np.std,
        "any": np.any,
        "all": np.all,
        "argmin": np.argmin,
        "argmax": np.argmax,
    }
    for shape in shapes:
        size = int(np.prod(shape))
        for dtype in dtypes:
            source = values_for(dtype, size, rng).reshape(shape)
            for name, operation in names.items():
                axes = [None, 0, 1] + ([-1] if len(shape) > 2 else []) + ([2] if len(shape) > 2 else [])
                for axis in axes:
                    for keepdims in (False, True):
                        with np.errstate(all="ignore"):
                            if axis is None:
                                result = operation(source, keepdims=keepdims)
                            else:
                                result = operation(source, axis=axis, keepdims=keepdims)
                        parameter = "axis=-" if axis is None else f"axis={axis}"
                        parameter += f":keepdims={1 if keepdims else 0}"
                        # A variance or a standard deviation is a sum of squares, and NumPy adds
                        # those in pairs while this library adds them two lanes at a time, so the
                        # last bit of a float answer is allowed to differ.
                        tolerance = 0.0
                        if name in ("variance", "std", "mean"):
                            tolerance = 1e-12
                        elif name == "sum" and dtype in ("float32", "float64"):
                            tolerance = 1e-12
                        record(
                            f"{name}/{dtype}/{dims(source)}/{parameter}",
                            f"{name}:{parameter}",
                            [source],
                            np.asarray(result),
                            tolerance,
                        )


def generate_linear_algebra(rng):
    for dtype in ("float64", "float32", "int64"):
        a = values_for(dtype, 6, rng).reshape(2, 3)
        b = values_for(dtype, 12, rng).reshape(3, 4)
        record(f"matmul/{dtype}/2x3/3x4", "matmul", [a, b], np.matmul(a, b))
        record(f"matmul/{dtype}/2x3/3", "matmul", [a, b[:, 0]], np.matmul(a, b[:, 0]))
        record(f"matmul/{dtype}/1d/2d", "matmul", [a[0], a.T], np.matmul(a[0], a.T))
        record(f"matmul/{dtype}/2/2", "matmul", [a[0], a[1]], np.matmul(a[0], a[1]))
        stack_a = values_for(dtype, 12, rng).reshape(2, 2, 3)
        stack_b = values_for(dtype, 12, rng).reshape(2, 3, 2)
        record(f"matmul/{dtype}/batched", "matmul", [stack_a, stack_b], np.matmul(stack_a, stack_b))
        wide = values_for(dtype, 12, rng).reshape(2, 2, 3)
        record(f"matmul/{dtype}/broadcast", "matmul", [wide, b[0:3, 0:2]], np.matmul(wide, b[0:3, 0:2]))
        record(f"dot/{dtype}/3d", "dot", [wide, b], np.dot(wide, b))
        record(f"dot/{dtype}/2d1d", "dot", [a, b[:, 1]], np.dot(a, b[:, 1]))
        record(f"outer/{dtype}", "outer", [a[0], b[0]], np.outer(a[0], b[0]))
        record(f"inner/{dtype}", "inner", [a, b[0:3, 0:3]], np.inner(a, b[0:3, 0:3]))
        square = values_for(dtype, 9, rng).reshape(3, 3)
        record(f"trace/{dtype}", "trace", [square], np.trace(square))
        record(f"trace/{dtype}/skew", "trace:offset=1", [square], np.trace(square, offset=1))


def generate_shape(rng):
    source = values_for("int32", 12, rng).reshape(3, 4)
    record("reshape/3x4->4x3", "reshape:4x3", [source], source.reshape(4, 3))
    record("reshape/3x4->12", "reshape:12", [source], source.reshape(12))
    record("reshape/3x4->2x-1", "reshape:2x-1", [source], source.reshape(2, -1))
    record("transpose/3x4", "transpose", [source], source.T)
    record("transpose/axes", "transpose:axes=1x0", [source], np.transpose(source, (1, 0)))
    cube = values_for("float64", 24, rng).reshape(2, 3, 4)
    record("transpose/3d", "transpose", [cube], cube.transpose())
    record("transpose/3d-axes", "transpose:axes=1x2x0", [cube], np.transpose(cube, (1, 2, 0)))
    record("swap_axes", "swap_axes:0:2", [cube], np.swapaxes(cube, 0, 2))
    record("move_axis", "move_axis:0:2", [cube], np.moveaxis(cube, 0, 2))
    record("ravel", "ravel", [cube], cube.ravel())
    record("ravel/transposed", "ravel", [source.T], np.ravel(source.T))
    record("squeeze", "squeeze", [cube.reshape(1, 2, 1, 3, 4)], np.squeeze(cube.reshape(1, 2, 1, 3, 4)))
    record("squeeze/axis", "squeeze:axis=0", [cube.reshape(1, 2, 3, 4)], np.squeeze(cube.reshape(1, 2, 3, 4), axis=0))
    record("expand_dims", "expand_dims:0", [source], np.expand_dims(source, 0))
    record("expand_dims/end", "expand_dims:-1", [source], np.expand_dims(source, -1))

    other = values_for("int32", 8, rng).reshape(2, 4)
    record("concatenate/0", "concatenate:0", [source, other], np.concatenate([source, other], axis=0))
    record("concatenate/1", "concatenate:1", [source, source], np.concatenate([source, source], axis=1))
    record("stack/0", "stack:0", [source, source], np.stack([source, source], axis=0))
    record("stack/1", "stack:1", [source, source], np.stack([source, source], axis=1))
    record("stack/-1", "stack:-1", [source, source], np.stack([source, source], axis=-1))
    record("hstack", "hstack", [source, source], np.hstack([source, source]))
    record("vstack", "vstack", [source, source], np.vstack([source, source]))

    flat = np.arange(12).astype("int64")
    for at in ([4], [4, 9], [0, 12], [3, 3]):
        record(f"split_at/{'+'.join(map(str, at))}", "split_at:" + "x".join(map(str, at)), [flat],
               np.split(flat, at))

    record("slice/span", "slice:span:1:3:1", [flat], flat[1:3])
    record("slice/step", "slice:span:0:12:5", [flat], flat[0:12:5])
    record("slice/reverse", "slice:span:11:-13:-4", [flat], flat[11::-4])
    record("slice/at", "slice:at:2:all", [source], source[2, :])
    record("slice/newaxis", "slice:newaxis", [flat], flat[np.newaxis, :])
    record("slice/2d", "slice:span:0:2:1:span:1:3:1", [source], source[0:2, 1:3])

    picks_one = np.array([4, 0, 2, -1])
    record("take/1d", "take", [flat, picks_one], np.take(flat, picks_one))
    picks_rows = np.array([2, 0, 0])
    record("take/axis0", "take:axis=0", [source, picks_rows], np.take(source, picks_rows, axis=0))
    picks_columns = np.array([3, 1])
    record("take/axis1", "take:axis=1", [source, picks_columns], np.take(source, picks_columns, axis=1))
    mask_values = np.array(
        [True, False, True, False, False, True, False, True, False, False, True, True]
    )
    record("mask/1d", "mask", [flat, mask_values], flat[mask_values])

    record("sort/1d", "sort", [np.array([3, 1, 2, 5, 4])], np.sort(np.array([3, 1, 2, 5, 4])))
    record("sort/axis0", "sort:axis=0", [source], np.sort(source, axis=0))
    record("sort/axis1", "sort:axis=1", [source], np.sort(source, axis=1))
    record("sort/nan", "sort", [np.array([3.0, np.nan, -1.0, np.inf])],
           np.sort(np.array([3.0, np.nan, -1.0, np.inf])))
    record("argsort/1d", "argsort", [np.array([3, 1, 2, 5, 4])], np.argsort(np.array([3, 1, 2, 5, 4])))
    record("argsort/axis0", "argsort:axis=0", [source], np.argsort(source, axis=0))
    record("argsort/tie", "argsort", [np.array([2, 1, 2, 1])], np.argsort(np.array([2, 1, 2, 1]), kind="stable"))

    condition = np.array([[True, False, True], [False, True, False]])
    x = values_for("float64", 6, rng).reshape(2, 3)
    y = values_for("float64", 3, rng)
    record("select", "select", [condition, x, y], np.where(condition, x, y))
    record("clip", "clip:-1:1", [x], np.clip(x, -1.0, 1.0))
    record("clip/min", "clip:-100:1.5", [x], np.clip(x, -100.0, 1.5))


def generate_casts(rng):
    for dtype in DTYPES:
        source = values_for("float64", 8, rng)
        with np.errstate(all="ignore"):
            result = source.astype(dtype)
        record(f"cast/{dtype}", f"cast:{dtype}", [source], result, tol=0.0)
        values = values_for(dtype, 8, rng)
        record(f"cast/from-{dtype}", "cast:float64", [values], values.astype("float64"))


def generate_linalg(rng):
    """Dense linear algebra, where the answer is unique.

    A QR, SVD or eigen decomposition has factors that are only unique up to signs and order, so those
    are checked by rebuilding the matrix in the unit tests instead of by comparing factors here.
    """
    cases = []

    def add(name, op, operands, out, tol=0.0):
        cases.append({"name": name, "op": op, "tol": tol, "in": operands, "out": out})

    def square(seed, size, dtype="float64"):
        return rng.normal(size=size * size).reshape(size, size).astype(dtype) + np.eye(size) * size

    for dtype in ("float64", "float32"):
        a = square(1, 3, dtype)
        b = rng.normal(size=3).astype(dtype)
        add(f"solve/{dtype}/3", "solve", [a, b], np.linalg.solve(a, b), 1e-9)
        multiple = rng.normal(size=6).reshape(3, 2).astype(dtype)
        add(f"solve/{dtype}/3x2", "solve", [a, multiple], np.linalg.solve(a, multiple), 1e-9)
        add(f"inv/{dtype}/3", "inv", [a], np.linalg.inv(a), 1e-9)
        add(f"det/{dtype}/3", "det", [a], np.asarray(np.linalg.det(a)), 1e-9)
        small = square(2, 2, dtype)
        add(f"det/{dtype}/2", "det", [small], np.asarray(np.linalg.det(small)), 1e-9)

    for dtype in ("float64",):
        spd = np.array([[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]])
        add("cholesky/spd", "cholesky", [spd], np.linalg.cholesky(spd), 1e-12)
        over = rng.normal(size=12).reshape(4, 3)
        target = rng.normal(size=4)
        add("lstsq/4x3", "lstsq", [over, target], np.linalg.lstsq(over, target, rcond=None)[0], 1e-9)
        wide = rng.normal(size=12).reshape(3, 4)
        short_target = rng.normal(size=3)
        add(
            "lstsq/3x4",
            "lstsq",
            [wide, short_target],
            np.linalg.lstsq(wide, short_target, rcond=None)[0],
            1e-9,
        )

    vector = np.array([3.0, -4.0, 12.0])
    for order in ("fro", "1", "2", "inf", "-inf", "3"):
        key = {"fro": None, "inf": np.inf, "-inf": -np.inf, "1": 1, "2": 2, "3": 3}[order]
        add(f"norm/vector/{order}", f"norm:{order}", [vector], np.asarray(np.linalg.norm(vector, ord=key)))
    matrix = np.array([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]])
    for order in ("fro", "1", "inf", "2"):
        key = {"fro": "fro", "1": 1, "inf": np.inf, "2": 2}[order]
        add(f"norm/matrix/{order}", f"norm:{order}", [matrix], np.asarray(np.linalg.norm(matrix, ord=key)), 1e-12)
    add("norm/axis0", "norm:fro:axis=0", [matrix], np.linalg.norm(matrix, axis=0))
    add("norm/axis1", "norm:1:axis=1", [matrix], np.linalg.norm(matrix, ord=1, axis=1))

    power = np.array([[2.0, 1.0], [0.0, 1.0]])
    add("matrix_power/3", "matrix_power:3", [power], np.linalg.matrix_power(power, 3))
    add("matrix_power/-2", "matrix_power:-2", [power], np.linalg.matrix_power(power, -2), 1e-12)
    add("matrix_power/0", "matrix_power:0", [power], np.linalg.matrix_power(power, 0))
    add("kron", "kron", [np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[0.0, 5.0], [6.0, 7.0]])],
        np.kron(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[0.0, 5.0], [6.0, 7.0]])))
    return cases


def write_vectors(cases, promotions, path, np_version, title):
    lines = [
        f"# {title}.",
        f"# numpy {np_version}, generated by scripts/gen_vectors.py, seed {SEED}.",
        "# Each payload is the array's little-endian bytes as hex; `-` stands for none.",
    ]
    if promotions:
        lines.append(f"promotions {len(promotions)}")
        for left, right, result in promotions:
            lines.append(f"promote {left} {right} {result}")
    lines.append(f"cases {len(cases)}")
    for case in cases:
        lines.append(f"case {case['name']}")
        lines.append(f"op {case['op']}")
        if case["tol"]:
            lines.append(f"tol {case['tol']!r}")
        for operand in case["in"]:
            lines.append(f"in {operand.dtype.name} {dims(operand)} {encode(operand)}")
        outputs = case["out"] if isinstance(case["out"], list) else [case["out"]]
        for out in outputs:
            lines.append(f"out {out.dtype.name} {dims(out)} {encode(np.asarray(out))}")
        lines.append("end")
    chunk_size = 250
    chunks = [lines[index : index + chunk_size] for index in range(0, len(lines), chunk_size)]
    parts = [
        "// Copyright 2026 Leo Cheng",
        "// SPDX-License-Identifier: Apache-2.0",
        "",
        f"// Generated by scripts/gen_vectors.py from numpy {np_version}, {len(cases)} cases.",
        "// DON'T EDIT IT.",
    ]
    for number, chunk in enumerate(chunks, start=1):
        parts.append("")
        parts.append("///|")
        parts.append(f"fn vector_lines_{number:02d}() -> String {{")
        parts.append("  let text =")
        for line in chunk:
            parts.append("    #|" + line)
        parts.append("  text")
        parts.append("}")
    parts.append("")
    parts.append("///|")
    parts.append("/// Every generated vector line, in the order the generator wrote them.")
    parts.append("fn vector_text() -> String {")
    parts.append("  let chunks : Array[() -> String] = [")
    for number in range(1, len(chunks) + 1):
        parts.append(f"    vector_lines_{number:02d},")
    parts.append("  ]")
    parts.append("  let parts : Array[String] = []")
    parts.append("  for chunk in chunks {")
    parts.append("    parts.push(chunk())")
    parts.append("  }")
    parts.append('  parts.join("\\n")')
    parts.append("}")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(parts) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="src/vectors_data_wbtest.mbt")
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)

    generate_arithmetic(rng)
    generate_math(rng)
    generate_binary_math(rng)
    generate_reductions(rng)
    generate_linear_algebra(rng)
    generate_shape(rng)
    generate_casts(rng)

    promotions = generate_promotions()
    write_vectors(
        CASES,
        promotions,
        "src/vectors_data_wbtest.mbt",
        np.__version__,
        "moonvec's NumPy differential vectors",
    )
    linalg_cases = generate_linalg(rng)
    write_vectors(
        linalg_cases,
        [],
        "src/linalg/vectors_data_wbtest.mbt",
        np.__version__,
        "linalg's NumPy differential vectors",
    )
    print(
        f"{len(CASES)} array cases with {len(promotions)} promotion cells and "
        f"{len(linalg_cases)} linear algebra cases written"
    )


if __name__ == "__main__":
    main()
