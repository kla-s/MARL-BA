import subprocess
import os
"""
Generates testing accuracy for each agent for each epoch of a series of trained models. Saves them as a Tensorboard log.
Expects to be launched in MARL-BA folder
"""
attention = [None]*1000
test_dir = "test_log/"
train_dir = "train_log/"

gpu = 1

# pat_names = ["AAE", "AAK", "ABP", "AKQ", "AKY", "AMB", "AMP", "ANK", "ATH", "AWP"]
# pat_names = ["AKQ", "ANK", "ATH"]
pat_names = ["AKQ"]
num_it = [15000] * len(pat_names)
num_epochs = [20] * len(pat_names)
agents = [3] * len(pat_names)
experiment_nrs = [5] * len(pat_names)
attention = [None] * len(pat_names)


assert len(pat_names) == len(num_it) == len(num_epochs) == len(agents)


assert os.path.isdir(train_dir)
for idx, train_log in enumerate(os.listdir(train_dir)):
    for idy, pat in enumerate(pat_names):
        try:
            experiment_nr = experiment_nrs[idy]
        except:
            experiment_nr = None
        try:
            date = dates[idy]
        except:
            date = None
        if date is not None and experiment_nr is not None:
            name = "{0}_{1}_{2}_{3}_{4}".format(pat, num_it[idy], num_epochs[idy], date, experiment_nr)
        elif date is not None:
            name = "{0}_{1}_{2}_{3}".format(pat, num_it[idy], num_epochs[idy], date)
        elif experiment_nr is not None:
            name = "{0}_{1}_{2}_{3}".format(pat, num_it[idy], num_epochs[idy], experiment_nr)
        else:
            name = "{0}_{1}_{2}".format(pat, num_it[idy], num_epochs[idy])
        if attention[idy] is not None:
            config = "{}_{}_{}".format(num_it[idy], num_epochs[idy], attention[idy])
        else:
            config = "{}_{}".format(num_it[idy], num_epochs[idy])

        if name == train_log:
            # init_test
            # call_params = 'python3 DQN.py --task eval --test_all_acc --name "{0}" --load "/MARL-BA/train_log/{0}/checkpoint" ' \
            #               '--algo Dueling --agents {1} --gpu {4} --testDir "{2}" --files "/Landmarks_RL_Data/Test_Data/data_{3}' \
            #               '/filenames/image_files.txt" "/Landmarks_RL_Data/Test_Data/data_{3}/filenames/landmark_files.txt" ' \
            #               '--mask_roi "/Landmarks_RL_Data/Test_Data/data_{3}/mask/pat{3}_prevo_mask.npy" --coords_init ' \
            #               '"/Landmarks_RL_Data/Test_Data/data_{3}/filenames/init_test.txt"'.format(
            #     name, agents[idy], test_dir, pat, gpu)

            # point_box
            call_params = 'python3 DQN.py --task eval --test_all_acc --name "{0}" --load_config "{5}" --load "/MARL-BA/train_log/{0}/checkpoint" ' \
                          '--algo Dueling --agents {1} --gpu {4} --testDir "{2}" --files "/Landmarks_RL_Data/A/Test_Data/data_{3}' \
                          '/filenames/image_files.txt" "/Landmarks_RL_Data/A/Test_Data/data_{3}/filenames/landmark_files.txt" ' \
                          '--mask_roi "/Landmarks_RL_Data/A/Test_Data/data_{3}/mask/pat{3}_prevo_mask.npy" --coords_init ' \
                          '"/Landmarks_RL_Data/A/Test_Data/data_{3}/filenames/init_point_box.txt"'.format(
                name, agents[idy], test_dir, pat, gpu, config)


            # tumor box
            # call_params = 'python3 DQN.py --task eval --test_all_acc --name "{0}" --load "/MARL-BA/train_log/{0}/checkpoint" ' \
            #               '--algo Dueling --agents {1} --gpu {4} --testDir "{2}" --files "/Landmarks_RL_Data/Test_Data/data_{3}' \
            #               '/filenames/image_files.txt" "/Landmarks_RL_Data/Test_Data/data_{3}/filenames/landmark_files.txt" ' \
            #               '--mask_roi "/Landmarks_RL_Data/Test_Data/data_{3}/mask/pat{3}_prevo_mask.npy" --coords_init ' \
            #               '"/Landmarks_RL_Data/Test_Data/data_{3}/initialization/tumor_box/pat{3}_prevo_coord.npy"'.format(
            #     name, agents[idy], test_dir, pat, gpu)

            # run
            subprocess.run(call_params, shell=True)