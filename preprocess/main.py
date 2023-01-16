
# import required module
import os
 
# iterate over files in
# that directory
i=0
for filename in os.scandir("./fire_data/data/fire"):
    if filename.is_file():
        print(filename.path)
        os.rename(str(filename.path), "./fire_data/data/fire/fire_nealkotval_" + str(i) + ".jpg")
        i+=1

i=0
for filename2 in os.scandir("./fire_data/data/non_fire"):
    if filename2.is_file():
        print(filename2.path)
        os.rename(str(filename2.path), "./fire_data/data/non_fire/non_fire_nealkotval_" + str(i) + ".jpg")
        i+=1
