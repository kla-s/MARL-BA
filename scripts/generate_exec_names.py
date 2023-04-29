date = None
experiment_nrs = [None] * 1000
attention = [None] * 1000
cmd = "CMD "

gpu = 1

# pat_names = ["AAE", "AAK", "ABP", "AKQ", "AKY"]#, "AMB", "AMP", "ANK", "ATH", "AWP"
# pat_names = ['AWP', 'ATH', 'ANK', 'AMP', "AMB"]

# pat_names = ["AMB", 'ATH', "AKY"]
# pat_names = ['ANK', 'AMP']

# pat_names = ["AAE", "AAK", "ABP", "AKQ", "AKY", "AMB", "AMP", "ANK", "ATH", "AWP"]
pat_names = ["AMP"]
num_it = [15000] * len(pat_names)
num_epochs = [20] * len(pat_names)
agents = [3] * len(pat_names)
attention = [22] * len(pat_names)
doc = ["A"] * len(pat_names)
experiment_nrs = [17] * len(pat_names)

# pat_names = ["AAE", "AAK", "ABP", "AKQ", "AKY", "AMB", "AMP", "ANK", "ATH", "AWP"]
# num_it = [100000]*len(pat_names)
# num_epochs = [20]*len(pat_names)
# agents = [3]*len(pat_names)
# experiment_nrs = [0]*len(pat_names)

for idx, pat in enumerate(pat_names):
    if date is not None and experiment_nrs[idx] is not None:
        name = "{0}_{1}_{2}_{3}_{4}".format(pat, num_it[idx], num_epochs[idx], date, experiment_nrs[idx])
    elif date is not None:
        name = "{0}_{1}_{2}_{3}".format(pat, num_it[idx], num_epochs[idx], date)
    elif experiment_nrs[idx] is not None:
        name = "{0}_{1}_{2}_{3}".format(pat, num_it[idx], num_epochs[idx], experiment_nrs[idx])
    else:
        name = "{0}_{1}_{2}".format(pat, num_it[idx], num_epochs[idx])

    if attention[idx] is not None:
        config = "{}_{}_{}".format(num_it[idx], num_epochs[idx], attention[idx])
    else:
        config = "{}_{}".format(num_it[idx], num_epochs[idx])

    cmd += 'python3 /MARL-BA/DQN.py --task train --name "{0}" ' \
           '--algo Dueling --agents {1} --load_config "{7}" --gpu {5} --files "/Landmarks_RL_Data/{6}/Train_Data/data_{2}' \
           '/filenames/image_files.txt" "/Landmarks_RL_Data/{6}/Train_Data/data_{2}/filenames/landmark_files.txt" ' \
           '--mask_roi "/Landmarks_RL_Data/{6}/Train_Data/data_{2}/mask/pat{2}_post1_mask.npy" --coords_init ' \
           '"/Landmarks_RL_Data/{6}/Train_Data/data_{2}/filenames/init_point_box.txt"; '.format(
        name, agents[idx], pat, num_it[idx], num_epochs[idx], gpu, doc[idx], config)
print(cmd[:-2])
