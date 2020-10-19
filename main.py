import random
from utils import maths, stopwatch

NUMBER_OF_TRIES = 10


linear_data = [
    (
        random.randint(1, 10),
        random.randint(11, 20),
        random.randint(1, 20),
        random.randint(1, 20),
    )
    for _ in range(NUMBER_OF_TRIES)
]

trigonometric_data = [
    (random.randint(1, 10), random.randint(11, 20)) for _ in range(NUMBER_OF_TRIES)
]


calculator = maths.Calculator()
stoper = stopwatch.Stopwatch()


for iteration in range(NUMBER_OF_TRIES):

    calculator.calc_lin_integral(
        linear_data[iteration][0],
        linear_data[iteration][1],
        linear_data[iteration][2],
        linear_data[iteration][3],
    )
    calculator.calc_cos_integral(
        trigonometric_data[iteration][0], trigonometric_data[iteration][1]
    )
    calculator.calc_sin_integral(
        trigonometric_data[iteration][0], trigonometric_data[iteration][1]
    )

print(stoper.current_timer())
