import os

"""
Expects Landmarks_RL_Data and MARL-BA to be in the same directory. Modifies all the necessary file paths. 
"""

test_data_dir = "/Landmarks_RL_Data/B/Test_Data"

assert os.path.isdir(test_data_dir)

pat_names = ["AAE", "AAK", "ABP", "AKQ", "AKY", "AMB", "AMP", "ANK", "ATH", "AWP"]
# pat_names = ["AMP"]

for data_pat in os.listdir(test_data_dir):
    assert data_pat[:-3] == "data_"
    pat = data_pat[-3:]
    if pat in pat_names:
        assert os.path.isdir(os.path.join(test_data_dir, data_pat))
        for dir in os.listdir(os.path.join(test_data_dir, data_pat)):

            if dir == "filenames":
                # image_files
                # silently deletes existing file
                f = open(os.path.join(test_data_dir, data_pat, "filenames", "image_files.txt"), "w")
                f.write(os.path.join(test_data_dir, "data_{}/images/pat{}_prevo.nii.gz".format(pat, pat)))
                f.close()

                # init_point_box
                # silently deletes existing file
                f = open(os.path.join(test_data_dir, data_pat, "filenames", "init_point_box.txt"), "w")
                f.write(
                    os.path.join(test_data_dir, "data_{}/initialization/point_box/pat{}_post1_label1.npy\n".format(pat,
                                                                                                                  pat)))
                f.write(
                    os.path.join(test_data_dir, "data_{}/initialization/point_box/pat{}_post1_label2.npy\n".format(pat,
                                                                                                                  pat)))
                f.write(
                    os.path.join(test_data_dir, "data_{}/initialization/point_box/pat{}_post1_label3.npy".format(pat,
                                                                                                                  pat)))
                f.close()

                # init_train_point
                # silently deletes existing file
                f = open(os.path.join(test_data_dir, data_pat, "filenames", "init_train_point.txt"), "w")
                f.write(
                    os.path.join(test_data_dir, "data_{}/initialization/train_point/pat{}_post1_label1.npy\n".format(pat,
                                                                                                                    pat)))
                f.write(
                    os.path.join(test_data_dir, "data_{}/initialization/train_point/pat{}_post1_label2.npy\n".format(pat,
                                                                                                                    pat)))
                f.write(
                    os.path.join(test_data_dir, "data_{}/initialization/point_box/pat{}_post1_label3.npy".format(pat,
                                                                                                                  pat)))
                f.close()

                # init_test
                # silently deletes existing file
                f = open(os.path.join(test_data_dir, data_pat, "filenames", "init_test.txt"), "w")
                f.write(
                    os.path.join(test_data_dir, "data_{}/initialization/init_test/pat{}_post1_label1_coord.npy\n".format(
                        pat, pat)))
                f.write(
                    os.path.join(test_data_dir, "data_{}/initialization/init_test/pat{}_post1_label2_coord.npy\n".format(
                        pat, pat)))
                f.write(
                    os.path.join(test_data_dir, "data_{}/initialization/init_test/pat{}_post1_label3_coord.npy".format(pat,
                                                                                                                  pat)))
                f.close()

                # landmark_files
                # silently deletes existing file
                f = open(os.path.join(test_data_dir, data_pat, "filenames", "landmark_files.txt"), "w")
                f.write(os.path.join(test_data_dir, "data_{}/landmarks/pat{}_prevo.txt".format(pat, pat)))
                f.close()

                print("Finished modifying test data files for pat {}".format(pat))
