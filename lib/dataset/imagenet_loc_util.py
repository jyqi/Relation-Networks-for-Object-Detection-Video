
"""
make the ImageSet split files from select class list

"""
import os

num_cls = 100 #214 #298
data_path = "../../data/ILSVRC2015"
synset_file = os.path.join(data_path, 'ImageSets', 'LOC_%d_synset_mapping.txt' % num_cls)

with open(synset_file, 'r') as f:
    lines = f.readlines()

desired_synsets = [line.split()[0] for line in lines]
desired_synsets = set(desired_synsets)

image_sets_infile = os.path.join(data_path, 'ImageSets', 'LOC', 'CLS-LOC', 'train_loc.txt')

with open(image_sets_infile, 'r') as f:
    lines = f.readlines()

image_sets_outfile = os.path.join(data_path, 'ImageSets', 'LOC', 'CLS-LOC', '%d_train_loc.txt' % num_cls)

c = 1
with open(image_sets_outfile, 'w') as f:
    for line in lines:
        if line[:9] in desired_synsets:
            f.write("%s %d\n" % (line.split()[0], c))
            c += 1

