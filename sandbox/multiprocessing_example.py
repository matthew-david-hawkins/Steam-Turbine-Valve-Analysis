import time
from multiprocessing import Pool


def sum_square(number):
    s = 0
    for i in range(number):
        s += i * i
    return s


def sum_square_with_mp(numbers):

    start_time = time.time()
    p = Pool()
    for i in numbers:
        result = p.apply(sum_square, args = (i,))


    p.close()
    p.join()

    end_time = time.time() - start_time

    print(f"Processing {len(numbers)} numbers took {end_time} time using multiprocessing.")


def sum_square_no_mp(numbers):

    start_time = time.time()
    result = []

    for i in numbers:
        result.append(sum_square(i))
    end_time = time.time() - start_time

    print(f"Processing {len(numbers)} numbers took {end_time} time using serial processing.")


if __name__ == '__main__':
    numbers = range(40000)
    sum_square_with_mp(numbers)
    sum_square_no_mp(numbers)