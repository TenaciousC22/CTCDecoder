import csv

with open("targetsPunctuated.txt") as f:
	lines=[line.rstrip('\n').upper() for line in f]

modelText=lines[0]
for x in range(1,len(lines)):
	modelText=modelText+" "+lines[x]

lexiconWords=modelText.split()

lexiconWords.sort()

print(lexiconWords)