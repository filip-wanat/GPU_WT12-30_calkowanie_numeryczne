import math
import numpy as np


class Calculator:
    def calc_sin_integral(self, start, end, step=0.01, amplitude=1):
        return sum(
            [amplitude * math.sin(x) * step for x in np.arange(start, end, step)]
        )

    def calc_cos_integral(self, start, end, step=0.01, amplitude=1):
        return sum(
            [amplitude * math.cos(x) * step for x in np.arange(start, end, step)]
        )

    def calc_lin_integral(self, start, end, a, b, step=0.01):
        return sum([(a * x + b) * step for x in np.arange(start, end, step)])


if __name__ == "__main__":
    calc = Calculator()
    print(calc.calc_cos_integral(0, 3.14))
    print(calc.calc_sin_integral(0, 3.14))
    print(calc.calc_lin_integral(0, 1, 1, 0))
