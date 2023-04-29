import os
import csv
import numpy as np

test_dir = "/MARL-BA/test_log/"

# it_epochs_experimentNr
str2match = "15000_20_17"

patients = []

read_rows = []

for idx, train_log in enumerate(os.listdir(test_dir)):
    if train_log[4:] == str2match:
        patients.append(train_log[:3])
        fname = os.path.join(test_dir, train_log, train_log + ".csv")
        if os.path.isfile(fname):
            with open(fname, "r", newline='') as saved_csv:
                reader = csv.DictReader(saved_csv)
                for row in reader:
                    row_dict = dict(row)
                    row_dict["patient"] = train_log[:3]
                    read_rows.append(row_dict)

# with open(os.path.join(test_dir, str2match + ".csv"), "w+", newline='') as csv_file:
#     csv_writer = csv.writer(csv_file)
#     csv_writer.writerow(list(read_rows[0].keys()))
#     for row in read_rows:
#         csv_writer.writerow(list(row.values()))

all_dists = {}
keys = ["dist_{}".format(i) for i in range(0, 20)]
all_dists_per_agent = {}
for row in read_rows:
    if row["step"] in all_dists.keys():
        all_dists[row["step"]].extend([float(row[key]) for key in keys])
        all_dists_per_agent[row["step"]]["agent {}".format(row["agent"])].extend([float(row[key]) for key in keys])
    else:
        all_dists[row["step"]] = [float(row[key]) for key in keys]
        if int(row["agent"]) == 0:
            all_dists_per_agent[row["step"]] = {"agent 0": [float(row[key]) for key in keys], "agent 1": [],
                                                "agent 2": []}
        elif int(row["agent"]) == 1:
            all_dists_per_agent[row["step"]] = {"agent 0": [], "agent 1": [float(row[key]) for key in keys],
                                                "agent 2": []}
        elif int(row["agent"]) == 2:
            all_dists_per_agent[row["step"]] = {"agent 0": [], "agent 1": [],
                                                "agent 2": [float(row[key]) for key in keys]}

mean_dists = {}
var_dists = {}
med_dists = {}
mean_dists_per_agent = {}
var_dists_per_agent = {}
for key in all_dists.keys():
    mean_dists[key] = np.mean(all_dists[key])
    med_dists[key] = np.median(all_dists[key])
    var_dists[key] = np.var(all_dists[key])

    mean_dists_per_agent[key] = [np.mean(all_dists_per_agent[key]["agent {}".format(i)]) for i in range(3)]
    var_dists_per_agent[key] = [np.var(all_dists_per_agent[key]["agent {}".format(i)]) for i in range(3)]

print("die magische Zahl ist {} bei med {}, var {} und std dev {}".format(mean_dists["75000"], med_dists["75000"],
                                                                          var_dists["75000"],
                                                                          np.sqrt(var_dists["75000"])))
idx = np.argmin(list(mean_dists.values()))
print("die gecheatete Zahl ist {} bei med {}, var {} und std dev {}".format(np.min(list(mean_dists.values())),
                                                                            list(med_dists.values())[idx],
                                                                            list(var_dists.values())[idx],
                                                                            np.sqrt(list(var_dists.values())[idx])))
idx = np.argmin(np.mean(list(mean_dists_per_agent.values()), axis=1))
print("per patient am besten ist {}".format(list(mean_dists_per_agent.values())[idx]))
print("die beste landmarke im mittel ist {}".format(list(mean_dists_per_agent.values())[np.argmin(list(mean_dists_per_agent.values()))//3]))
for i in range(3):
    print("die beste landmarke für agent {} ist {}".format(i, np.min(np.array(list(mean_dists_per_agent.values())),0)[i]))
print(mean_dists)
print(mean_dists_per_agent)
print(patients)
