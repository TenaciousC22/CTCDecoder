import numpy as np
import csv
from ctc_decoder import lexicon_search, BKTree
from tqdm import tqdm

offsetMap={
	0:"I840",
	1:"I720",
	2:"I600",
	3:"I480",
	4:"I360",
	5:"I240",
	6:"I060",
	7:"base",
	8:"B060",
	9:"B240",
	10:"B360",
	11:"B480",
	12:"B600",
	13:"B720",
	14:"B840",
	15:"jumble"
}

#mat = np.array([[0.4, 0, 0.6], [0.4, 0, 0.6]])
#chars = 'ab'
chars=" ETOAINSHRLDUYWGCMFBP'VKJXQZ0192856734"

data_path="/home/analysis/Documents/studentHDD/chris/predictiveCodingCharacterExperiment/tensors/"

def createDatasetPaths():
	paths=[]

	for x in range(6):
		for y in range(28):
			for key in offsetMap:
				paths.append(data_path+"speaker"+str(x+1)+"clip"+str(y+1)+"offset"+offsetMap[key]+".npy")

	return paths

data_paths=createDatasetPaths()

with open("dict.csv",newline='') as f:
	reader = csv.reader(f)
	data = list(reader)

data=data[0]

# create BK-tree from a list of words
bk_tree = BKTree(data)

for x in tqdm(range(len(data_paths))):
	arr=np.load(data_paths[x])

	# and use the tree in the lexicon search
	res=lexicon_search(arr, chars, bk_tree, tolerance=5)
	print(res)