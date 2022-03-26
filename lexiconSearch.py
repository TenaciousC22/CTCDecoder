import numpy as np
import csv
from ctc_decoder.best_path import best_path
from ctc_decoder.loss import probability
from ctc_decoder import beam_search, BKTree, LanguageModel
from tqdm import tqdm

def lexicon_search(mat: np.ndarray, chars: str, bk_tree: BKTree, tolerance: int) -> str:
	"""Lexicon search decoder.

	The algorithm computes a first approximation using best path decoding. Similar words are queried using the BK tree.
	These word candidates are then scored given the neural network output, and the best one is returned.
	See CRNN paper from Shi, Bai and Yao.

	Args:
		mat: Output of neural network of shape TxC.
		chars: The set of characters the neural network can recognize, excluding the CTC-blank.
		bk_tree: Instance of BKTree which is used to query similar words.
		tolerance: Words to be considered, which are within specified edit distance.

	Returns:
		The decoded text.
	"""

	with open("targetsPunctuated.txt") as f:
		lines=[line.rstrip('\n').upper() for line in f]

	modelText=lines[0]
	for x in range(1,len(lines)):
		modelText=modelText+" "+lines[x]

	lm = LanguageModel(modelText, chars)

	# use best path decoding to get an approximation
	approx = beam_search(arr, chars, beam_width=25, lm=lm)

	approx=approx.split()

	# get similar words from dictionary within given tolerance
	for word in approx:
		words = bk_tree.query(word, tolerance)
		print(words)

	# # if there are no similar words, return empty string
	# if not words:
	# 	return ''

	# # else compute probabilities of all similar words and return best scoring one
	# word_probs = [(w, probability(mat, w, chars)) for w in words]
	# word_probs.sort(key=lambda x: x[1], reverse=True)
	# return word_probs[0][0]

def createDatasetPaths():
	paths=[]

	for x in range(6):
		for y in range(28):
			for key in offsetMap:
				paths.append(data_path+"speaker"+str(x+1)+"clip"+str(y+1)+"offset"+offsetMap[key]+".npy")

	return paths

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
	#print(res)