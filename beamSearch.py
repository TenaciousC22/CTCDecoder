import numpy as np
from ctc_decoder import beam_search, LanguageModel
#from progressbar import progressbar
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

with open("targetsPunctuated.txt") as f:
	lines=[line.rstrip('\n').upper() for line in f]

modelText=lines[0]
for x in range(1,len(lines)):
	modelText=modelText+" "+lines[x]

lm = LanguageModel(modelText, chars)

def createDatasetPaths():
	paths=[]

	for x in range(6):
		for y in range(28):
			for key in offsetMap:
				paths.append(data_path+"speaker"+str(x+1)+"clip"+str(y+1)+"offset"+offsetMap[key]+".npy")

	return paths

data_paths=createDatasetPaths()

for x in tqdm(range(len(data_paths))):
	arr=np.load(data_paths[x])

	print(f'Beam search: "{beam_search(arr, chars,beam_width=25, lm=lm)}"')