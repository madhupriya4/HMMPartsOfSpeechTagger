import json, numpy, math, sys

# snippet to convert to utf-8
def decode(text):
	if isinstance(text, dict):
		return {decode(key): decode(value)
				for key, value in text.iteritems()}
	elif isinstance(text, list):
		return [decode(element) for element in text]
	elif isinstance(text, unicode):
		return text.encode('utf-8')
	else:
		return text


# extract input from HMM stored model files
def getInput():
	tagtext = open("hmmtags.txt", "r").read()
	tags = decode(json.loads(tagtext))

	A = numpy.loadtxt("hmmtransition.txt")

	emitext = open("hmmemission.txt", "r").read()
	B = decode(json.loads(emitext))

	return tags, A, B


# viterbi algorithm
def viterbi(line, A, B, tags):

	obs = line.split(" ")
	t = len(obs)
	n = len(A) - 1  # don't do for start tag

	viterbiScore = numpy.zeros((n, t))

	#initialize backpointer matrix structure
	backpointer = []
	for i in xrange(n):
		backpointer.append([])
		for j in xrange(t):
			backpointer[i].append([0, 0])

	#1st iteration - for start state to all other states
	for tag in tags:

		if tag == 'start':
			continue
		#transition prob for start state to tag
		viterbiScore[tags[tag]][0] = (A[tags['start']][tags[tag]])

		#mutilpy emission prob for 1st word
		word1=obs[0].lower()
		if word1.lower() in B:
			if tag in B[word1]:
				viterbiScore[tags[tag]][0] *= (B[word1][tag])
			else:
				viterbiScore[tags[tag]][0] *= 0.0000001
		else:
			viterbiScore[tags[tag]][0] *= 0.0000001


	# run for remaining words
	for i in xrange(1, t, 1):

		word = obs[i].lower()

		for tag in tags:
			if tag == 'start':
				continue

			# find max of previous viterbi values
			maxV = 0.0

			#find all possible transitions to current state
			for tagPrev in tags:
				if tagPrev == 'start':
					continue

				#mutiply previous viterbi score with current transition prob
				v = viterbiScore[tags[tagPrev]][i - 1] * (A[tags[tagPrev]][tags[tag]])

				#mutiply emission prob
				if word in B:
					if tag in B[word]:
						v *= (B[word][tag])
					else:
						v *= 0.0000001
				else:
					v *= 0.0000001

				#find max
				if v > maxV:
					maxV = v
					maxIndex = [tags[tagPrev], i - 1]
					backpointer[tags[tag]][i] = maxIndex

			#set score to max value
			viterbiScore[tags[tag]][i] = maxV

	# calculate final Viterbi value

	maxV = 0
	maxIndex = [0, t - 1]

	for tag in tags:
		if tag == 'start':
			continue

		v = viterbiScore[tags[tag]][t - 1]

		if maxV <= v:
			maxV = v
			maxIndex = [tags[tag], t - 1]


	# retrive final tag sequence from backpointers

	curMax = maxIndex
	for i in xrange(t - 1, -1, -1):

		#get tag from tag row num(key from value) for each backpointer

		value = curMax[0]
		keyToGet = ""
		for key in tags.keys():
			if tags[key] == value:
				keyToGet = key
				break

		#get correct tag for word
		obs[i] = obs[i] + "/" + keyToGet + " "
		curMax = backpointer[curMax[0]][curMax[1]]

	return "".join(obs)


def main():

	tags, A, B = getInput()
	text = open(sys.argv[1]).read().split("\n")

	file = open("hmmoutput.txt", "w")

	for line in text:
		res = viterbi(line, A, B, tags)
		file.write(res + "\n")
	file.close()

main()