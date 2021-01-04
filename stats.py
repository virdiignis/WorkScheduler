import csv
from functools import partial
from hashlib import md5
from multiprocessing import Pool
from typing import Tuple, Any
from itertools import combinations, chain
from numpy import array, ndarray
import numpy as np
import random
import time

import pandas as pd

PEOPLE_PER_SHIFT = 2

EMPLOYEES_NUMBER = 10
EMPLOYEES_TIME_PARTS = np.array([
    1.,
    1.,
    1.,
    1.,
    1.,
    1.,
    1.,
    1,
    0.5,
    1.,
])
assert len(EMPLOYEES_TIME_PARTS) == EMPLOYEES_NUMBER

EMPLOYEES_SHIFTS_UP_TO_NOW = np.array([
    56,
    56,
    56,
    59,
    57,
    55,
    50,
    56,
    28,
    56,
]) / EMPLOYEES_TIME_PARTS
EMPLOYEES_SHIFTS_UP_TO_NOW -= EMPLOYEES_SHIFTS_UP_TO_NOW.min()
EMPLOYEES_SHIFTS_UP_TO_NOW = EMPLOYEES_SHIFTS_UP_TO_NOW.astype('int')

PREF_RED = -500
PREF_YELLOW = -90
PREF_WHITE = 0
PREF_BLUE = 70

SHIFTS_IN_A_ROW_PENALTY = 150
SHIFTS_WITH_8H_BREAK_PENALTY = 80
MORE_THAN_2_SHIFTS_IN_40H_PENALTY = 30
MAX_DIFFERENCE_IN_SHIFTS_PER_PERSON_PENALTY = 10
SHIFTS_COUNT_PER_PERSON_DEVIATION_PENALTY = 20
UNCOVERED_SHIFTS_PENALTY = 290
SHIFT_TIME_INEQUALITY_PENALTY = 10

RANDOM_INSTANCES = 100000
GENERATION_INSTANCES = 100
ALGORITHM_STEPS = 400000
INITIAL_BEST_SCORE = 2800


def log(s: str):
    print(f"{time.strftime('%H:%M:%S')}: {s}")


class Schedule:
    def __init__(self, arr: ndarray, prefs: ndarray):
        assert arr.shape == prefs.shape
        self._arr = arr.astype('bool')
        self._prefs = prefs
        self._score = None

    def get_scored(self):
        return self, self.get_score()

    def get_score(self, verbose=False):
        if self._score is None or verbose:
            self._score = self.calculate_score(verbose)
        return self._score

    def calculate_score(self, verbose=False) -> Any:
        partial_score = 0
        uncovered_shifts = self._arr.shape[0] * 2 - self._arr.sum()
        partial_score -= uncovered_shifts * UNCOVERED_SHIFTS_PENALTY
        shifts_preferences = self._prefs[self._arr]
        partial_score += shifts_preferences.sum()  # preferences conformation
        shifts_in_a_row = 0
        shifts_with_8h_break = 0
        more_than_2_shifts_in_40h = 0

        for i in range(self._arr.shape[0] - 1):
            rest = self._arr[i:]
            shifts_in_a_row += (rest[:2].sum(0) > 1).sum()
            if rest.shape[0] > 2:
                shifts_with_8h_break += (rest[:3].sum(0) > 1).sum()
                if rest.shape[0] > 4:
                    more_than_2_shifts_in_40h += (rest[:5].sum(0) > 2).sum()

        partial_score -= shifts_in_a_row * SHIFTS_IN_A_ROW_PENALTY
        partial_score -= shifts_with_8h_break * SHIFTS_WITH_8H_BREAK_PENALTY
        partial_score -= more_than_2_shifts_in_40h * MORE_THAN_2_SHIFTS_IN_40H_PENALTY

        partial_sums = self._arr.sum(0).astype('float')
        partial_sums /= EMPLOYEES_TIME_PARTS
        partial_sums += EMPLOYEES_SHIFTS_UP_TO_NOW
        shifts_num_difference = partial_sums.ptp().astype('int')
        partial_score -= MAX_DIFFERENCE_IN_SHIFTS_PER_PERSON_PENALTY * shifts_num_difference
        deviations_sum = np.abs(partial_sums - np.mean(partial_sums)).round()
        partial_diffs_info = deviations_sum.sum().astype('int')
        partial_diffs = (deviations_sum ** 2).sum().astype('int')
        partial_score -= partial_diffs * SHIFTS_COUNT_PER_PERSON_DEVIATION_PENALTY

        morngins = self._arr[0::3].sum(0)
        evenings = self._arr[1::3].sum(0)
        nights = self._arr[2::3].sum(0)
        shift_time_inequality = np.vstack([morngins, evenings, nights]).ptp(0).sum()

        partial_score -= shift_time_inequality * SHIFT_TIME_INEQUALITY_PENALTY

        if verbose:
            print(f"Score:\t\t\t\t\t\t\t\t\t\t\t{partial_score}")
            print(
                f"Uncovered shifts:\t\t\t\t\t\t\t\t{uncovered_shifts}\t\tPunishment:\t\t{uncovered_shifts * UNCOVERED_SHIFTS_PENALTY}")
            blue__sum = (shifts_preferences == PREF_BLUE).sum()
            print(f"Shifts on blue:\t\t\t\t\t\t\t\t\t{blue__sum}\t\tBonus:\t\t{blue__sum * PREF_BLUE}")
            print(f"Shifts on white:\t\t\t\t\t\t\t\t{(shifts_preferences == PREF_WHITE).sum()}")
            yellow__sum = (shifts_preferences == PREF_YELLOW).sum()
            print(f"Shifts on yellow:\t\t\t\t\t\t\t\t{yellow__sum}\t\tPunishment:\t\t{-yellow__sum * PREF_YELLOW}")
            red__sum = (shifts_preferences == PREF_RED).sum()
            print(f"Shifts on red:\t\t\t\t\t\t\t\t\t{red__sum}\t\tPunishment:\t\t{-red__sum * PREF_RED}")
            print(
                f"Shifts in a row:\t\t\t\t\t\t\t\t{shifts_in_a_row}\t\tPunishment:\t\t{shifts_in_a_row * SHIFTS_IN_A_ROW_PENALTY}")
            print(
                f"Shifts with 8h break:\t\t\t\t\t\t\t{shifts_with_8h_break}\t\tPunishment:\t\t{shifts_with_8h_break * SHIFTS_WITH_8H_BREAK_PENALTY}")
            print(
                f"More than 2 shifts in 40h:\t\t\t\t\t\t{more_than_2_shifts_in_40h}\t\tPunishment:\t\t{more_than_2_shifts_in_40h * MORE_THAN_2_SHIFTS_IN_40H_PENALTY}")
            print(
                f"Max difference in shifts per person:\t\t\t{shifts_num_difference}\t\tPunishment:\t\t{shifts_num_difference * MAX_DIFFERENCE_IN_SHIFTS_PER_PERSON_PENALTY}")
            print(
                f"Deviations by more than 0.5 from mean:\t\t\t{partial_diffs_info}\t\tPunishment:\t\t{partial_diffs * SHIFTS_COUNT_PER_PERSON_DEVIATION_PENALTY}")
            print(
                f"Shift times inequalites sum:\t\t\t\t\t{shift_time_inequality}\t\tPunishment:\t\t{np.round(shift_time_inequality * SHIFT_TIME_INEQUALITY_PENALTY, 1)}")
            print(flush=True)

        return partial_score

    def __hash__(self):
        return int(md5(self._arr).hexdigest(), 16)

    def __bool__(self):
        return True

    def __eq__(self, other):
        if type(other) is Schedule:
            return self.hash() == other.hash()
        else:
            return self._arr == other.arr

    def __getitem__(self, item):
        return self._arr.__getitem__(item)

    def __setitem__(self, key, value):
        self._score = None
        return self._arr.__setitem__(key, value)

    @property
    def arr(self):
        return self._arr

    @arr.setter
    def arr(self, arr):
        self._score = None
        self._arr = arr

    @property
    def shape(self):
        return self._arr.shape

    @property
    def prefs(self):
        return self._prefs

    def sum(self, *args, **kwargs):
        return self._arr.sum(*args, **kwargs)

    def hash(self):
        return self.__hash__()

    def dump(self):
        with open(f"random/schedule_{time.strftime('%d_%m_%y-%H:%M:%S')}_{self.get_score()}.csv", 'w',
                  newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(self._arr.astype('int'))

    def copy(self):
        return Schedule(self._arr.copy(), self._prefs)

    def __lt__(self, other):
        return self.get_score() < other.get_score()


class ShiftFactory:
    @staticmethod
    def random(shift_prefs: ndarray, prev_shift: ndarray = None, prev2_shift: ndarray = None) -> ndarray:
        assert isinstance(shift_prefs, ndarray)
        assert shift_prefs.shape == (EMPLOYEES_NUMBER,)
        if prev_shift is not None:
            assert isinstance(prev_shift, ndarray)
            assert prev_shift.shape == (EMPLOYEES_NUMBER,)
        if prev2_shift is not None:
            assert isinstance(prev2_shift, ndarray)
            assert prev2_shift.shape == (EMPLOYEES_NUMBER,)
        z = np.zeros(EMPLOYEES_NUMBER, dtype='bool')
        allowed_enum = (shift_prefs != PREF_RED)
        if prev_shift is not None:
            allowed_enum *= np.abs(prev_shift - 1).astype('bool')
        if prev2_shift is not None:
            allowed_enum *= np.abs(prev2_shift - 1).astype('bool')
        allowed_indexes = np.where(allowed_enum)[0]
        max_choice = min(allowed_indexes.shape[0], 2)
        rand_indexes = np.random.choice(allowed_indexes, max_choice, replace=False)
        z[rand_indexes] = 1
        return z


class ScheduleFactory:
    @staticmethod
    def random(map_placeholder, schedule_prefs: ndarray) -> Schedule:
        shifts = []
        for ix, pz in enumerate(schedule_prefs):
            if ix > 1:
                shifts.append(ShiftFactory.random(pz, shifts[ix - 1], shifts[ix - 2]))
            elif ix > 0:
                shifts.append(ShiftFactory.random(pz, shifts[ix - 1]))
            else:
                shifts.append(ShiftFactory.random(pz))

        return Schedule(array(shifts, dtype='bool'), schedule_prefs)

    @staticmethod
    def consecutive(s: Tuple[Schedule, Schedule]) -> Schedule:
        if random.randint(0, 1):
            s1, s2 = s
        else:
            s2, s1 = s
        return Schedule(array(list(s1[i] if i % 2 == 0 else s2[i] for i in range(s1.shape[0])), dtype='bool'), s1.prefs)

    @staticmethod
    def half_by_half(s: Tuple[Schedule, Schedule]) -> Schedule:
        if random.randint(0, 1):
            s1, s2 = s
        else:
            s2, s1 = s
        half = s1.shape[0] // 2
        p1 = list(s1[:half])
        p2 = list(s2[half:])
        p1.extend(p2)
        return Schedule(array(p1, dtype='bool'), s1.prefs)

    @classmethod
    def mutated(cls, schedule: Schedule):
        schedule_copy = schedule.copy()
        for _ in range(random.choice((1, 3))):  # TODO: stestować ten parametr
            variant = random.randint(0, 10)
            if variant <= -1:
                cls.__swap_shifts(schedule_copy)
            elif variant <= 0:
                cls.__swap_shift_between_2_people(schedule_copy)
            elif variant <= 5:
                cls.__remove_person_from_shift(schedule_copy)
            elif variant <= 10:
                cls.__add_person_to_unfilled_shift(schedule_copy)

        return schedule_copy

    @staticmethod
    def __add_person_to_unfilled_shift(schedule_copy):
        indexes = np.where(schedule_copy.sum(1) < PEOPLE_PER_SHIFT)[0]
        if indexes.size:
            shift = np.random.choice(indexes)
            employee = np.random.randint(0, EMPLOYEES_NUMBER)
            while schedule_copy[shift][employee] == 1:
                employee = np.random.randint(0, EMPLOYEES_NUMBER)
            schedule_copy[shift][employee] = 1
        return schedule_copy

    @staticmethod
    def __remove_person_from_shift(schedule_copy):
        shift = np.random.randint(0, schedule_copy.shape[0])
        people_on_shift_indexes = np.where(schedule_copy[shift] == 1)[0]
        if people_on_shift_indexes.size:
            person = np.random.choice(people_on_shift_indexes)
            schedule_copy[shift][person] = 0
        return schedule_copy

    @staticmethod
    def __swap_shift_between_2_people(schedule_copy):
        person_1 = np.random.randint(0, EMPLOYEES_NUMBER)
        person_2 = np.random.randint(0, EMPLOYEES_NUMBER)
        while person_1 == person_2:
            person_2 = np.random.randint(0, EMPLOYEES_NUMBER)
        schedule_copy_arr = np.transpose(schedule_copy.arr)
        person_1_shifts = np.where(schedule_copy_arr[person_1] == 1)[0]
        person_2_shifts = np.where(schedule_copy_arr[person_2] == 1)[0]
        p1_shifts_set = set(person_1_shifts)
        p2_shifts_set = set(person_2_shifts)
        if p1_shifts_set and p2_shifts_set and not p1_shifts_set.issubset(p2_shifts_set) and not p2_shifts_set.issubset(
                p1_shifts_set):
            p1_shift = np.random.choice(person_1_shifts)
            while p1_shift in person_2_shifts:
                p1_shift = np.random.choice(person_1_shifts)
            p2_shift = np.random.choice(person_2_shifts)
            while p2_shift in person_1_shifts:
                p2_shift = np.random.choice(person_2_shifts)
            schedule_copy_arr[person_1][p1_shift] = 0
            schedule_copy_arr[person_2][p1_shift] = 1
            schedule_copy_arr[person_1][p2_shift] = 1
            schedule_copy_arr[person_2][p2_shift] = 0
        schedule_copy.arr = np.transpose(schedule_copy_arr)
        return schedule_copy

    @staticmethod
    def __swap_shifts(schedule_copy):
        p1 = np.random.randint(0, schedule_copy.shape[0])
        p2 = np.random.randint(0, schedule_copy.shape[0])
        while p1 == p2:
            p2 = np.random.randint(0, schedule_copy.shape[0])

        tmp = schedule_copy[p1]
        schedule_copy[p1] = schedule_copy[p2]
        schedule_copy[p2] = tmp
        return schedule_copy

    @staticmethod
    def from_file(path: str, prefs: ndarray) -> Schedule:
        shifts = prefs.shape[0]
        with open(path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            data = [list(map(int, row)) for row in reader]

        if len(data) < shifts:
            diff = shifts - len(data)
            data.extend([[0] * len(data[0])] * diff)
        elif len(data) > shifts:
            data = data[:shifts]

        return Schedule(array(data, dtype='bool'), prefs)


class GeneticAlgorithm:
    def __init__(self, prefs, instances=None):
        self.best = INITIAL_BEST_SCORE
        self.best_hashes = []
        self.pool = Pool()
        self.prefs = prefs
        log("Read preferences.")
        # self._pr = partial(ScheduleFactory.random, schedule_prefs=self.prefs)
        log("Generating instances.")
        # self.instances = list(self.pool.imap_unordered(self._pr, [0] * RANDOM_INSTANCES, 1000))
        # for i in self.instances:
        #     i.dump()
        # return
        self.instances = instances
        # if paths is not None:
        #     self.instances.extend(ScheduleFactory.from_file(path, self.prefs) for path in paths)
        log("Ranking instances.")
        self.ranking(fuzzy=False)
        del self.instances[GENERATION_INSTANCES:]

    @staticmethod
    def read_preferences(path) -> ndarray:
        mapf = lambda x: {
            "-2": PREF_RED,
            "-1": PREF_YELLOW,
            "0": PREF_WHITE,
            "1": PREF_BLUE
        }[x]
        with open(path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            data = [list(map(mapf, row)) for row in reader]
        return array(data, dtype='int')

    def ranking(self, fuzzy=True):
        self.instances = set(self.instances)
        scored = self.pool.imap_unordered(Schedule.get_scored, self.instances, 10)
        # scored = [ps(i) for i in self.instances]
        self.instances = [x[0] for x in
                          sorted(scored, key=lambda x: -x[1] + (np.random.randint(-5, 6) if fuzzy else 0))]

    def run(self, mutation_variant=0, pairing_variant=0, breeding_variant=0, fuzzy_variant=0):
        stats = []
        log(f"Random instances: {RANDOM_INSTANCES}\n"
            f"Generation instances: {GENERATION_INSTANCES}\n"
            f"Mutation variant: {mutation_variant}\n"
            f"Pairing variant: {pairing_variant}\n"
            f"Breeding variant: {breeding_variant}\n"
            f"Fuzzy variant: {fuzzy_variant}\n")
        ta = time.perf_counter()
        ltp = 0
        for i in range(ALGORITHM_STEPS):
            iter_stats = [RANDOM_INSTANCES, GENERATION_INSTANCES, mutation_variant, pairing_variant, breeding_variant,
                          fuzzy_variant, i]
            # log(f"Iteration {i} started.")
            # self.instances.extend(self.pool.imap_unordered(self._pr, [0] * int(GENERATION_INSTANCES / 10)))
            t = time.perf_counter()
            if mutation_variant == 1:
                self.instances.extend(self.pool.imap_unordered(ScheduleFactory.mutated, self.instances[:]))
            elif mutation_variant == 2:
                self.instances.extend(self.pool.imap_unordered(ScheduleFactory.mutated, self.instances[:] * 10))
            elif mutation_variant == 3:
                self.instances.extend(self.pool.imap_unordered(ScheduleFactory.mutated, self.instances[:]))
                self.instances.extend(self.pool.imap_unordered(ScheduleFactory.mutated, self.instances[:]))
                self.instances.extend(self.pool.imap_unordered(ScheduleFactory.mutated, self.instances[:]))
            elif mutation_variant == 4:
                self.instances.extend(self.pool.imap_unordered(ScheduleFactory.mutated, self.instances[:]))
                self.instances.extend(
                    self.pool.imap_unordered(ScheduleFactory.mutated, self.instances[len(self.instances) // 2:]))
                self.instances.extend(
                    self.pool.imap_unordered(ScheduleFactory.mutated, self.instances[len(self.instances) // 2:]))
            mutation_time = time.perf_counter() - t
            # log(f"Instances mutated in {mutation_time :.3f}s.")
            iter_stats.append(mutation_time)
            t = time.perf_counter()
            if mutation_variant:
                self.ranking(fuzzy_variant)
                del self.instances[GENERATION_INSTANCES:]
            ranking_time = time.perf_counter() - t
            iter_stats.append(ranking_time)

            t = time.perf_counter()

            if pairing_variant == 1:
                pairs = [(self.instances[i], self.instances[i + 1]) for i in range(len(self.instances) - 1)]
            elif pairing_variant == 2:
                pairs = [(self.instances[i], self.instances[i + 1]) for i in range(len(self.instances) - 1)]
                pairs.extend((self.instances[i], self.instances[i + 2]) for i in range(len(self.instances) - 2))
            elif pairing_variant == 3:
                pairs = [(self.instances[i], self.instances[i + 1]) for i in range(len(self.instances) - 1)]
                pairs.extend((self.instances[i], self.instances[i + 2]) for i in range(len(self.instances) - 2))
                half_instances = len(self.instances) // 2
                pairs.extend((self.instances[i], self.instances[i + half_instances]) for i in range(half_instances))
            elif pairing_variant == 4:
                half_instances = len(self.instances) // 2
                pairs = [(self.instances[i], self.instances[i + half_instances]) for i in range(half_instances)]
            elif pairing_variant == 5:
                pairs = combinations(self.instances, 2)
            elif pairing_variant == 6:
                pairs = chain(
                    combinations(self.instances[GENERATION_INSTANCES // 2:], 2),
                    combinations(self.instances[:GENERATION_INSTANCES // 2], 2),
                )

            if breeding_variant == 1:
                self.instances.extend(self.pool.imap_unordered(ScheduleFactory.consecutive, pairs))
            elif breeding_variant == 2:
                self.instances.extend(self.pool.imap_unordered(ScheduleFactory.half_by_half, pairs))
            elif breeding_variant == 3:
                self.instances.extend(self.pool.imap_unordered(ScheduleFactory.consecutive, pairs))
                self.instances.extend(self.pool.imap_unordered(ScheduleFactory.half_by_half, pairs))

            crossing_time = time.perf_counter() - t
            # log(f"Instances crossed in {crossing_time :.3f}s.")
            iter_stats.append(crossing_time)

            t = time.perf_counter()
            if breeding_variant:
                self.ranking(fuzzy=False)
                del self.instances[GENERATION_INSTANCES:]
            ranking_time = time.perf_counter() - t
            iter_stats.append(ranking_time)

            best_score = self.instances[0].get_score()
            worst_score = self.instances[-1].get_score()
            time_passed = time.perf_counter() - ta
            iter_stats.append(time_passed)
            iter_stats.append(best_score)
            iter_stats.append(worst_score)
            stats.append(iter_stats)
            if time_passed - ltp > 10:
                ltp = time_passed
                log(f"Time passed: {time_passed}\t\tBest rank: {best_score}\t\tWorst rank: {worst_score}")
            if best_score > self.best:
                # self.instances[0].get_score(verbose=True)
                # self.instances[0].dump()
                self.best = best_score
                self.best_hashes.append(self.instances[0].hash())
            elif best_score == self.best and self.instances[0].hash() not in self.best_hashes:
                # self.instances[0].get_score(verbose=True)
                # self.instances[0].dump()
                self.best_hashes.append(self.instances[0].hash())
            if time_passed > 60:
                break
        # self.ranking(fuzzy=False)
        # self.instances[0].get_score(verbose=True)
        log(f"Score to 60s: {best_score}")
        return stats
        # self.instances[0].dump()


if __name__ == '__main__':
    # with open('schedules/paths.txt') as f:
    #     paths = list(map(str.strip, f.readlines()))
    # GeneticAlgorithm('/home/prance/Studia/COVID/Grafik/prefs_sty.csv')
    # exit()

    with open('random/paths.txt') as f:
        paths = list(map(str.strip, f.readlines()))

    random_instances = (100, 1000, 10000, 30000)
    generation_instances = (10, 25, 50, 100, 200, 300, 400, 500)
    mutation_variants = range(5)
    pairing_variants = range(7)
    breeding_variants = range(4)
    fuzzy_variants = range(2)
    #
    # # test 1
    # g = 100
    # m = 1
    # c = 1
    # e = 2
    # f = 1
    # for r in random_instances:
    #     p_paths = paths[:r]
    #     RANDOM_INSTANCES = r
    #     alg = GeneticAlgorithm('/home/prance/Studia/COVID/Grafik/prefs_gru_bez27.csv', p_paths)
    #     stats = np.array(alg.run(m, c, e, f))
    #     pd.DataFrame(stats).to_csv(f"stats/stats_{r}_{g}_{m}_{c}_{e}_{f}.csv", header=None, index=None)

    prefs = GeneticAlgorithm.read_preferences('/home/prance/Studia/COVID/Grafik/prefs_sty.csv')
    for r in random_instances:
        instances = list(ScheduleFactory.from_file(path, prefs) for path in paths[:r])
        RANDOM_INSTANCES = r
        for g in generation_instances:
            if g > r:
                continue
            GENERATION_INSTANCES = g
            for mutation_variant in mutation_variants:
                for pairing_variant in pairing_variants:
                    if mutation_variant == 0 and pairing_variant == 0:
                        continue
                    if pairing_variant == 0:
                        breeding_variant = 0
                        for fuzzy_variant in fuzzy_variants:
                            alg = GeneticAlgorithm(prefs, instances)
                            stats = np.array(
                                alg.run(mutation_variant, pairing_variant, breeding_variant, fuzzy_variant))
                            pd.DataFrame(stats).to_csv(
                                f"stats/stats_{r}_{g}_{mutation_variant}_{pairing_variant}_{breeding_variant}_{fuzzy_variant}.csv",
                                header=None,
                                index=None)
                        continue
                    else:
                        for breeding_variant in tuple(breeding_variants)[1:]:
                            for fuzzy_variant in fuzzy_variants:
                                alg = GeneticAlgorithm(prefs, instances)
                                stats = np.array(
                                    alg.run(mutation_variant, pairing_variant, breeding_variant, fuzzy_variant))
                                pd.DataFrame(stats).to_csv(
                                    f"stats/stats_{r}_{g}_{mutation_variant}_{pairing_variant}_{breeding_variant}_{fuzzy_variant}.csv",
                                    header=None,
                                    index=None)

# g = GeneticAlgorithm('/home/prance/Studia/COVID/Grafik/prefs_gru_bez27.csv', paths)
# g.run(3, 3, 3, 3)

# prefs = GeneticAlgorithm.read_preferences()
# sc = ScheduleFactory.random(None, prefs)
