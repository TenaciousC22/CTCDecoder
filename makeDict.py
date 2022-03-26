import csv

with open("targetsPunctuated.txt") as f:
	lines=[line.rstrip('\n').upper() for line in f]

modelText=lines[0]
for x in range(1,len(lines)):
	modelText=modelText+" "+lines[x]

lexiconWords=modelText.split()

lexiconWords.sort()

x=1
while x < len(lexiconWords):
	if lexiconWords[x-1]==lexiconWords[x]:
		lexiconWords.pop(x)
	else:
		x+=1

with open("dict.csv","w") as f:
	write=csv.writer(f)
	write.writerow(lexiconWords)