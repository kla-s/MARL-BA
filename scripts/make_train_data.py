import os

"""
Expects Landmarks_RL_Data and MARL-BA to be in the same directory. Modifies all the necessary file paths. 
"""
#docs = ["A", "B"]
docs = ["A"]
train_data_dir_pre = "/Landmarks_RL_Data/"
train_data_dir_post = "Train_Data"
train_data_dirs = []

pat_names = ["AAE", "AAK", "ABP", "AKQ", "AKY", "AMB", "AMP", "ANK", "ATH", "AWP"]

for doc in docs:
    train_data_dir = os.path.join(train_data_dir_pre, doc, train_data_dir_post)
    print(train_data_dir)
    train_data_dirs.append(train_data_dir)
    assert os.path.isdir(train_data_dir)
    for data_pat in os.listdir(train_data_dir):
        assert data_pat[:-3] == "data_"
        pat = data_pat[-3:]
        if pat in pat_names:
            assert os.path.isdir(os.path.join(train_data_dir, data_pat))
            for dir in os.listdir(os.path.join(train_data_dir, data_pat)):

                if dir == "filenames":
                    # image_files
                    # silently deletes existing file
                    f = open(os.path.join(train_data_dir, data_pat, "filenames", "image_files.txt"), "w")
                    f.write(os.path.join(train_data_dir, "data_{}/images/pat{}_post1.nii.gz".format(pat, pat)))
                    f.close()

                    # init_point_box
                    # silently deletes existing file
                    f = open(os.path.join(train_data_dir, data_pat, "filenames", "init_point_box.txt"), "w")
                    f.write(
                        os.path.join(train_data_dir,
                                     "data_{}/initialization/point_box/pat{}_post1_label1.npy\n".format(pat,
                                                                                                         pat)))
                    f.write(
                        os.path.join(train_data_dir,
                                     "data_{}/initialization/point_box/pat{}_post1_label2.npy\n".format(pat,
                                                                                                         pat)))
                    f.write(
                        os.path.join(train_data_dir, "data_{}/initialization/point_box/pat{}_post1_label3.npy".format(pat,
                                                                                                                       pat)))
                    f.close()

                    # landmark_files
                    # silently deletes existing file
                    f = open(os.path.join(train_data_dir, data_pat, "filenames", "landmark_files.txt"), "w")
                    f.write(os.path.join(train_data_dir, "data_{}/landmarks/pat{}_post1.txt".format(pat, pat)))
                    f.close()

                    print("Finished modifying train data files for pat {} doc {}".format(pat, doc))
print("Finished modifying train data files for all patients in {}".format(train_data_dirs))
