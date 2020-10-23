import csv
import os
import numpy as np

from main import Schedule

paths = [p for p in os.listdir() if p.startswith("schedule_") and p.endswith("141.csv")]

sd = dict()
schedules = []
for p in paths:
    with open(p, newline='') as F:
        reader = csv.reader(F)
        data = [list(map(int, row)) for row in reader]
    schedule = np.array(data, dtype='int').view(Schedule)
    schedules.append(schedule)
    sd[schedule] = p

scheds = [s for s in schedules if s.sum(0)[1] == 18 and s.sum(0)[9] == 18 and s.sum(0)[4] == 19 and s.sum(0)[5] == 19 and s.sum(0)[6] == 18]
scheds = np.array(scheds)
print(scheds.sum(0))