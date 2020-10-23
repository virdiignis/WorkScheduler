import csv
from functools import partial
from hashlib import md5
from multiprocessing import Pool
from typing import Tuple, Any

from numpy import array, ndarray
import numpy as np
import random
import time

EMPLOYEES_NUMBER = 10

PREF_RED = -50
PREF_YELLOW = -5
PREF_WHITE = 0
PREF_BLUE = 4

SHIFTS_IN_A_ROW_PENALTY = 10
SHIFTS_WITH_8H_BREAK_PENALTY = 8
MORE_THAN_2_SHIFTS_IN_40H_PENALTY = 3
MAX_DIFFERENCE_IN_SHIFTS_PER_PERSON_PENALTY = 1
SHIFTS_COUNT_PER_PERSON_DEVIATION_PENALTY = 2
UNCOVERED_SHIFTS_PENALTY = 29

RANDOM_INSTANCES = 20000
GENERATION_INSTANCES = 200
ALGORITHM_STEPS = 40000
INITIAL_BEST_SCORE = 0


def log(s: str):
    print(f"{time.strftime('%H:%M:%S')}: {s}", flush=True)


class ShiftPreferences(ndarray):
    pass


class SchedulePreferences(ndarray):
    pass


class Shift(ndarray):
    pass


class ShiftFactory:
    @staticmethod
    def random(shift_prefs: ShiftPreferences) -> Shift:
        z = np.zeros(len(shift_prefs), dtype='int').view(Shift)
        allowed_indexes = np.arange(EMPLOYEES_NUMBER)[shift_prefs != PREF_RED]
        max_choice = min(len(allowed_indexes), 2)
        rand_indexes = random.sample(list(allowed_indexes), max_choice)
        z[rand_indexes] = 1
        return z


class Schedule(ndarray):
    def score(self, schedule_prefs: SchedulePreferences, verbose=False) -> Tuple[Any, int]:
        partial_score = 0
        uncovered_shifts = (self.shape[0] * 2) - int(self.sum())
        partial_score -= uncovered_shifts * UNCOVERED_SHIFTS_PENALTY
        shifts_preferences = schedule_prefs[self.astype('bool')]
        partial_score += int(shifts_preferences.sum())  # preferences conformation
        shifts_in_a_row = 0
        for i in range(self.shape[0] - 1):
            shifts_in_a_row += int((self[i:i + 2].sum(0) > 1).sum())
        partial_score -= shifts_in_a_row * SHIFTS_IN_A_ROW_PENALTY
        shifts_with_8h_break = 0
        for i in range(self.shape[0] - 2):
            shifts_with_8h_break += int((self[i:i + 3].sum(0) > 1).sum())
        partial_score -= shifts_with_8h_break * SHIFTS_WITH_8H_BREAK_PENALTY
        more_than_2_shifts_in_40h = 0
        for i in range(self.shape[0] - 4):
            more_than_2_shifts_in_40h += int((self[i:i + 5].sum(0) > 2).sum())
        partial_score -= more_than_2_shifts_in_40h * MORE_THAN_2_SHIFTS_IN_40H_PENALTY

        partial_sums = self.sum(0)
        partial_sums[8] = int(partial_sums[8]) * 2
        max_shifts_number = int(partial_sums.max())
        min_shifts_number = int(partial_sums.min())
        shifts_num_difference = int(max_shifts_number - min_shifts_number)
        occurrences = int((partial_sums == min_shifts_number).sum())
        partial_score -= MAX_DIFFERENCE_IN_SHIFTS_PER_PERSON_PENALTY * (shifts_num_difference ** 2) * occurrences
        partial_diffs = round(float(np.abs(partial_sums - np.mean(partial_sums)).sum()))
        partial_score -= partial_diffs * SHIFTS_COUNT_PER_PERSON_DEVIATION_PENALTY

        if verbose:
            print(f"Uncovered shifts: {uncovered_shifts}")
            print(f"Shifts on blue: {(shifts_preferences == PREF_BLUE).sum()}")
            print(f"Shifts on white: {(shifts_preferences == PREF_WHITE).sum()}")
            print(f"Shifts on yellow: {(shifts_preferences == PREF_YELLOW).sum()}")
            print(f"Shifts on red: {(shifts_preferences == PREF_RED).sum()}")
            print(f"Shifts in a row: {shifts_in_a_row}")
            print(f"Shifts with 8h break: {shifts_with_8h_break}")
            print(f"More than 2 shifts in 40h: {more_than_2_shifts_in_40h}")
            print(f"Max difference in shifts per person: {shifts_num_difference}")
            print(f"People with minimum shifts number: {occurrences}")
            print(f"Deviations sum: {partial_diffs}")

        return self, partial_score

    def mutation(self):
        schedule_copy = self.copy()
        for _ in range(random.randint(1, 10)):
            variant = random.randint(0, 11)
            if variant <= 3:
                self.__swap_shifts(schedule_copy)
            elif variant <= 9:
                schedule_copy = self.__swap_shift_between_2_people(schedule_copy)
            elif variant == 10:
                self.__remove_person_from_shift(schedule_copy)
            elif variant == 11:
                self.__add_person_to_unfilled_shift(schedule_copy)

        return schedule_copy

    @staticmethod
    def __add_person_to_unfilled_shift(schedule_copy):
        indexes = np.arange(schedule_copy.shape[0])[schedule_copy.sum(1) < 2]
        if int(indexes.size) > 0:
            z = random.choice(indexes)
            p = random.randint(0, EMPLOYEES_NUMBER - 1)
            while schedule_copy[z][p] == 1:
                p = random.randint(0, EMPLOYEES_NUMBER - 1)
            schedule_copy[z][p] = 1

    @staticmethod
    def __remove_person_from_shift(schedule_copy):
        z1 = random.randint(0, schedule_copy.shape[0] - 1)
        p1_indexes = np.arange(schedule_copy.shape[1])[schedule_copy[z1] == 1]
        if int(p1_indexes.size):
            p1 = random.choice(p1_indexes)
            schedule_copy[z1][p1] = 0

    @staticmethod
    def __swap_shift_between_2_people(schedule_copy):
        o1 = random.randint(0, 9)
        o2 = random.randint(0, 9)
        while o1 == o2:
            o2 = random.randint(0, 9)
        schedule_copy = np.transpose(schedule_copy)
        p1_indexes = np.arange(schedule_copy.shape[1])[schedule_copy[o1] == 1]
        p2_indexes = np.arange(schedule_copy.shape[1])[schedule_copy[o2] == 1]
        if int(p1_indexes.size) and int(p2_indexes.size):
            p1_z = random.choice(p1_indexes)
            while p1_z in p2_indexes:
                p1_z = random.choice(p1_indexes)
            p2_z = random.choice(p2_indexes)
            while p2_z in p1_indexes:
                p2_z = random.choice(p2_indexes)
            schedule_copy[o1][p1_z] = 0
            schedule_copy[o2][p1_z] = 1
            schedule_copy[o1][p2_z] = 1
            schedule_copy[o2][p2_z] = 0
        schedule_copy = np.transpose(schedule_copy)
        return schedule_copy

    @staticmethod
    def __swap_shifts(schedule_copy):
        p1 = random.randint(0, schedule_copy.shape[0] - 1)
        p2 = 0
        while p1 == p2:
            p2 = random.randint(0, schedule_copy.shape[0] - 1)

        tmp = schedule_copy[p1]
        schedule_copy[p1] = schedule_copy[p2]
        schedule_copy[p2] = tmp

    def __hash__(self):
        return int(md5(self).hexdigest(), 16)

    def __bool__(self):
        return True

    def hash(self):
        return self.__hash__()

    def dump(self, score):
        with open(f"schedule_{time.strftime('%d_%m_%y-%H:%M:%S')}_{score}.csv", 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(self)


class ScheduleFactory:
    @staticmethod
    def random(imap_placeholder, schedule_prefs: SchedulePreferences) -> Schedule:
        return array([ShiftFactory.random(pz) for pz in schedule_prefs], dtype='int').view(Schedule)

    @staticmethod
    def consecutive(s: Tuple[Schedule, Schedule]) -> Schedule:
        if random.randint(0, 1):
            s1, s2 = s
        else:
            s2, s1 = s
        return array(list(s1[i] if i % 2 == 0 else s2[i] for i in range(s1.shape[0])), dtype='int').view(Schedule)

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
        return array(p1, dtype='int').view(Schedule)

    @staticmethod
    def from_file(path: str) -> Schedule:
        with open(path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            data = [list(map(int, row)) for row in reader]
        return array(data, dtype='int').view(Schedule)


class GeneticAlgorithm:
    def __init__(self, paths=None):
        self.best = INITIAL_BEST_SCORE
        self.best_hashes = []
        self.pool = Pool()
        self.prefs = self.read_preferences()
        log("Read preferences.")
        self._pr = partial(ScheduleFactory.random, schedule_prefs=self.prefs)
        log("Generating instances.")
        self.instances = list(self.pool.imap_unordered(self._pr, [0] * RANDOM_INSTANCES))
        if paths is not None:
            self.instances.extend(ScheduleFactory.from_file(path) for path in paths)
        log("Ranking instances.")
        self.ranking(fuzzy=False)
        del self.instances[GENERATION_INSTANCES:]

    @staticmethod
    def read_preferences() -> SchedulePreferences:
        with open('prefs.csv', newline='') as csvfile:
            reader = csv.reader(csvfile)
            data = [list(map(int, row)) for row in reader]
        return array(data, dtype='int').view(SchedulePreferences)

    def ranking(self, fuzzy=True):
        self.instances = set(self.instances)
        ps = partial(Schedule.score, schedule_prefs=self.prefs)
        scored = self.pool.imap_unordered(ps, self.instances)
        self.instances = [x[0] for x in sorted(scored, key=lambda x: -x[1] + (random.randint(-5, 5) if fuzzy else 0))]

    def run(self):
        for i in range(ALGORITHM_STEPS):
            log(f"Iteration {i} started.")

            # self.instances.extend(self.pool.imap_unordered(self._pr, [0] * int(GENERATION_INSTANCES / 10)))
            self.instances.extend(self.pool.imap_unordered(Schedule.mutation, self.instances[:] * 10))
            self.ranking()
            del self.instances[GENERATION_INSTANCES:]
            log("Instances mutated.")

            pairs = [(self.instances[i], self.instances[i + 1]) for i in range(len(self.instances) - 1)]
            self.instances.extend(self.pool.imap_unordered(ScheduleFactory.consecutive, pairs))
            self.instances.extend(self.pool.imap_unordered(ScheduleFactory.half_by_half, pairs))
            self.ranking()
            del self.instances[GENERATION_INSTANCES:]
            log("Instances crossed.")

            best_score = self.instances[0].score(self.prefs)[1]
            log(f"Best rank: {best_score}\tWorst rank: {self.instances[-1].score(self.prefs)[1]}")
            if best_score > self.best:
                self.instances[0].score(self.prefs, verbose=True)
                self.instances[0].dump(best_score)
                self.best = best_score
            elif best_score == self.best and self.instances[0].hash() not in self.best_hashes:
                self.instances[0].score(self.prefs, verbose=True)
                self.instances[0].dump(best_score)
                self.best_hashes.append(self.instances[0].hash())
        self.ranking(fuzzy=False)
        _, sc = self.instances[0].score(self.prefs, verbose=True)
        self.instances[0].dump(sc)


if __name__ == '__main__':
    with open('paz.txt') as f:
        paths = map(str.strip, f.readlines())
    g = GeneticAlgorithm(paths)
    g.run()
