import numpy, json, sys

# get list of tags and tag counts for each word
def getTags(file_lines):

    tags = dict()
    words = dict()

    #extract each token from input text
    count = 0
    for line in file_lines:
        for token in line.split():

            x = token.split("/")
            tag = x[len(x) - 1] #extract tag
            word = x[0].lower() #extract word

            #add tag to dict of tags
            if tag not in tags:
                tags[tag] = count
                count += 1
            #add word to dict of words
            if word not in words:
                temp = dict()
                temp[tag] = 1
                words[word] = temp
            #associate tag to word if word is already present
            else:
                if tag not in words[word]:
                    words[word][tag] = 1
                else:
                    words[word][tag] += 1

    #add special start tag
    n = len(tags)
    tags['start'] = n
    return tags, words


# get transition counts
def getTransCount(file_lines, tags, n):
    trans_count = numpy.zeros((n, n))
    total_count1 = numpy.zeros(n)
    total_count2 = numpy.zeros(n)

    for line in file_lines:
        line = line.strip()
        line = 'starttoken/start ' + line
        tokens = line.split()

        for i in xrange(0, len(tokens) - 1, 1):
            w1 = tokens[i].split("/")[len(tokens[i].split("/")) - 1]
            w2 = tokens[i + 1].split("/")[len(tokens[i + 1].split("/")) - 1]

            if w1 in tags and w2 in tags:
                trans_count[tags[w1]][tags[w2]] += 1
                total_count1[tags[w1]] += 1
                total_count2[tags[w1]] += 1

        i = len(tokens) - 1
        total_count2[tags[tokens[i].split("/")[len(tokens[i].split("/")) - 1]]] += 1

    return trans_count, total_count1, total_count2

#transition matrix (=prob from one tag to another)
def calcTransProb(trans_count, total_count, n):
    for i in xrange(n):
        if total_count[i] > 0:
            trans_count[i] /= total_count[i]

    return trans_count


#emission calculation (=prob of word given tag)
def calcEmissionProb(words, total_count, tags):
    for word in words:
        total = 0.0
        for tag in words[word]:
            if total_count[tags[tag]] > 0:
                words[word][tag] /= total_count[tags[tag]]
    return words

#adding noise to make values non zero
def add_noise(probs):
    mean = numpy.nanmean(probs, axis=0)  # Axis 1
    noise = abs(0.01 * numpy.random.normal(mean, size=probs.shape))
    return probs + noise


# to write the HMM model in a human readable form for reference
def writeHumanReadable(tags, A, B):

    outputFile = open('hmmmodel.txt', "w")
    n = len(A)
    outputFile.write("List of tags with row number:\n")

    for key in tags:
        outputFile.write("{" + str(key) + ":" + str(tags[key]) + "} ,")

    outputFile.write("\n\nTransition matrix between tag indices as above:-\n\n")
    numpy.savetxt(outputFile, A)

    outputFile.write("\n\nEmission matrix:-\n\n")

    for key in B:
        outputFile.write(str(key) + ":" + str(B[key]) + ", ")
    outputFile.close()


def main():

    inputFile = open(sys.argv[1], "r")
    file_lines = inputFile.read().split("\n")

    #get tags and words
    tags, words = getTags(file_lines)
    n = len(tags)

    #get tag counts
    trans_count, total_count1, total_count2 = getTransCount(file_lines, tags, n)

    #calculate emission prob
    emiss_prob = calcEmissionProb(words, total_count2, tags)

    #smooth tag count values
    trans_count = add_noise(trans_count)
    total_count1 = add_noise(total_count1)

    #calculate transition prob
    trans_count = calcTransProb(trans_count, total_count1, n)

    inputFile.close()

    #save HMM mode

    with open('hmmtags.txt', 'w') as inputFile:
        inputFile.write(json.dumps(tags))
    numpy.savetxt('hmmtransition.txt', trans_count)
    with open('hmmemission.txt', 'w') as inputFile:
        inputFile.write(json.dumps(emiss_prob))
    inputFile.close()

    # writing a human readible form for reference
    writeHumanReadable(tags, trans_count, emiss_prob)


main()