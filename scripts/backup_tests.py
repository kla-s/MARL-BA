cmd =""
pat_names = ["AAE", "AAK", "ABP", "AKQ", "AKY", "AMB", "AMP", "ANK", "ATH", "AWP"]

for p in pat_names:
    cmd += "sudo cp -r {0}_{1}_{2}_0 {0}_{1}_{2}_tumor_box; ".format(p, 15000, 20)

print(cmd)