import numpy as np
from ctc_decoder import beam_search

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
				paths.append(data_path+"speaker"+str(x+1)+"clip"+str(y+1)+offsetMap[key])

	return paths

data_paths=createDatasetPaths()

for entry in data_paths:
	print(entry)

#print(f'Beam search: "{beam_search(mat, chars)}"')