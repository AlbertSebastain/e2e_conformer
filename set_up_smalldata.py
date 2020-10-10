import os
path = "/usr/home/shi/projects/data_aishell/data/test/"
output_path = "/usr/home/shi/projects/data_aishell/data/small_test"
threshold = 100
files = os.listdir(path)
if not os.path.exists(output_path):
    os.makedirs(output_path)
for file in files:
    if not os.path.isdir(os.path.join(path,file)):
        with open(os.path.join(path,file),"r") as f:
            if os.path.exists(os.path.join(output_path,file)):
                os.remove(os.path.join(output_path,file))
            n = 0
            for line in f.readlines():
                if n >= threshold:
                    break
                else:
                    with open(os.path.join(output_path,file),"a") as wf:
                        wf.write(line)
                    n = n+1
