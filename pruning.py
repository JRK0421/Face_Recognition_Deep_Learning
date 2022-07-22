import os
import shutil


def prune(dataSet, threshold=15):
    for subdir, dirs, files in os.walk(dataSet):
        if subdir == dataSet:
            continue
        number_img = 0
        for fName in files:
            (imageClass, imageName) = (os.path.basename(subdir), fName)
            if any(imageName.lower().endswith("." + ext) for ext in ["jpg", "png"]):
                number_img += 1
        if number_img < threshold:
            print("Removing {}".format(subdir))
            shutil.rmtree(subdir)
