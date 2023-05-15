import numpy as np
import sympy as sp
import sqlite3
import pywt
from scipy.integrate import quad
from scipy.interpolate.interpolate import interp1d
import inspect

global db_cache
db_cache = None

class DB_handler:
    @classmethod
    def get_instance(cls):
        global db_cache
        if db_cache is None:
            db_cache = DB_handler()
        return db_cache

    def __init__(self, db_file='cache.db'):
        self.db_file = db_file
        self.conn = sqlite3.connect(db_file)
        self.cursor = self.conn.cursor()

    def init_table(self, table_name):
        self.cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} (args TEXT PRIMARY KEY, result TEXT)")
        self.conn.commit()

    def insert_row_in_table(self, table_name, args, result):
        try:
            self.cursor.execute(f"INSERT INTO {table_name} VALUES (?, ?)", (args, result))
            self.conn.commit()
        except sqlite3.IntegrityError:
            pass

    def get_result_for_args(self, table_name, args):
        self.cursor.execute(f"SELECT result FROM {table_name} WHERE args=?", (args,))
        return self.cursor.fetchone()

    def get_all_records_for_table(self, table_name):
        self.cursor.execute(f"SELECT args, result FROM {table_name}")
        rows = self.cursor.fetchall()
        result = {args: float(result) for args, result in rows}
        return result

class Interp1dNumeric(interp1d):
    """ Wrapper for interp1 to raise TypeError for object array input
    We need this because sympy will try to evaluate interpolated functions when
    constructing expressions involving floats.  At least sympy 1.0 only accepts
    TypeError or AttributeError as indication that the implemented value cannot
    be sampled with the sympy expression.  Therefore, raise a TypeError
    directly for an input giving an object array (such as a sympy expression),
    rather than letting interp1d raise a ValueError.
    See:
    * https://github.com/nipy/nipy/issues/395
    * https://github.com/sympy/sympy/issues/10810
    """

    def __call__(self, x):
        if np.asarray(x).dtype.type == np.object_:
            raise TypeError('Object arrays not supported')
        return super(Interp1dNumeric, self).__call__(x)


class range_memoize:
    def __init__(self, rangeArgumentPosition):
        self.rangeArgumentPosition = rangeArgumentPosition
        self.memo = {}

    def __call__(self, f):
        def wrapped_f(*args):
            fArgs = args[:self.rangeArgumentPosition] + args[self.rangeArgumentPosition + 1:]
            rangeArg = args[self.rangeArgumentPosition]
            hashableVersionOffArgs = tuple()
            for arg in fArgs:
                x = arg.tobytes() if hasattr(arg, 'tobytes') else arg
                hashableVersionOffArgs += (x,)

            if hashableVersionOffArgs in self.memo:
                length = len(rangeArg)
                if length <= len(self.memo[hashableVersionOffArgs][0]):
                    return self.memo[hashableVersionOffArgs][1][:length]
            result = f(*args)
            self.memo[hashableVersionOffArgs] = (rangeArg, result)
            return result

        return wrapped_f


class method_with_sympy_func_memoize:
    def __init__(self, f):
        self.db_instance = DB_handler.get_instance()
        table_name = f.__name__
        self.db_instance.init_table(table_name)
        preloaded_memo = self.db_instance.get_all_records_for_table(table_name)
        self.memo = preloaded_memo
        self.f = f

    def __call__(self, *args):
        sympy_expr_args = [arg for arg in args if callable(arg) or isinstance(arg, sp.Expr)]
        non_sympy_arg = [arg for arg in args if arg not in sympy_expr_args]
        sympy_expr_args_str = sorted(inspect.getsource(arg) if not isinstance(arg, sp.Expr) else str(arg)
                                     for arg in sympy_expr_args)
        hashable_args = tuple(non_sympy_arg + sympy_expr_args_str)
        hashable_args_str = str(hashable_args)
        if hashable_args_str in self.memo:
            return self.memo[hashable_args_str]
        else:
            result = self.f(*args)
            if isinstance(result, tuple):
                result = result[0]
            self.memo[hashable_args] = result
            self.db_instance.insert_row_in_table(self.f.__name__, hashable_args_str, str(result))
            return result


@method_with_sympy_func_memoize
def quad_with_memoize(*args, **kwargs):
    return quad(*args, **kwargs)


class PolynomialModel:
    MAX_DEGREE = 14
    def __init__(self, x, y, err_threshold=0.01):
        errors_dict = {}

        degree = 0
        while True:
            # Create the system of linear equations
            A = []
            b = []

            for x_i, y_i in zip(x, y):
                row = [x_i ** i for i in range(degree + 1)]
                A.append(row)
                b.append(y_i)

            A = np.array(A)
            b = np.array(b)

            # Solve the system of linear equations
            coefficients, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

            # Construct the polynomial
            t = sp.Symbol('t')
            polynomial = sum([coeff * t ** i for i, coeff in enumerate(coefficients)])
            error = self.get_error(polynomial, x, y)

            errors_dict[polynomial] = error

            if error < err_threshold or degree == self.MAX_DEGREE:
                best_polynomial = min(errors_dict, key=errors_dict.get)
                break
            degree += 1

        self.polynomial = best_polynomial

    @staticmethod
    def get_error(polynomial, x, y):
        error = 0
        for x_i, y_i in zip(x, y):
            error += (y_i - polynomial.subs('t', x_i)) ** 2
        return error.evalf()

    def sympy_implementation(self):
        return self.polynomial


class WaveletReconstructedFunction:
    def __init__(self, coeffs, wavelet_family, levels, T):
        self.coeffs = coeffs
        self.wavelet_family = wavelet_family
        self.levels = levels
        self.T = T

    def vector(self):
        return pywt.waverec(self.coeffs, self.wavelet_family)

    def sympy_implementation(self):
        return sp.interpolating_spline(11, sp.Symbol('t'),
                                       self.vector(),
                                       np.linspace(0, self.T, 2 ** self.levels))

    def __str__(self):
        return f"WaveletReconstructed/{self.wavelet_family}/{self.coeffs}/{self.levels}/{self.T}"
