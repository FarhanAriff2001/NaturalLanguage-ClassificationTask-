# Farhan Ariff bin Halis Azhan
# farharif
# Assignment4

import sys
import os
import copy 
from collections import Counter
from preprocess import removeSGML, tokenizeText
import math


# TODO
# multiple word occurence in test files DONE
# standard output to terminal DONE
# Check formula in training and test
# Print out a new top 10

def trainNaiveBayes(list_filepaths):
    '''
    input: the list of file paths to be used for training;
    
    output: data structure with class probabilities (or log of probabilities);
    output: data structure with word conditional probabilities (or log of probabilities);
    output: any other parameters required (e.g., vocabulary size).
    
    - preprocess the content of the files provided as input, i.e., tokenize the text (you are encouraged to
        use the functions you implemented for Assignment 1, but you can also use another
        CAEN-compatible tokenizer if you prefer; please also include the related code if you choose to use
        your own processing code) and compute a vocabulary (which is composed of all the tokens in the
        training files, regardless of what class they belong to). In the basic implementation of your function,
        do not remove stopwords, do not use stemming.
    - calculate all the counts required by the Naive Bayes classifier
    - your implementation should be the multinomial Naive Bayes
    '''
    out1 = {"joke" : 0.0, "non-joke" :  0.0}
    out2 =  {"joke" : Counter(), "non-joke" :  Counter()}
    out3 = 0
    sets = set()
    for key in list_filepaths.keys():
        types = "unknown"
        if list_filepaths[key]["original"] == "joke":
            out1["joke"] +=1
            types = "joke"
        else:
            out1["non-joke"] +=1
            types = "non-joke"
        text = list_filepaths[key]["text"]
        out2[types] += Counter(text)
        sets.update(text)


    total = out1["joke"] + out1["non-joke"]
    # Option 1 : Probabilities
    # out1["joke"] = (out1["joke"]/total)
    # out1["non-joke"] = (out1["non-joke"]/total)
    # Option 2 : Log of probabilities
    out1["joke"] = math.log(out1["joke"]/total)
    out1["non-joke"] = math.log(out1["non-joke"]/total)

    # Total Number of words for each types
    typewords = {"joke" : 0.0, "non-joke" :  0.0}
    # typewords["joke"] = out1["joke"]
    # typewords["non-joke"] = out1["non-joke"]
    typewords["joke"] = sum(out2["joke"].values())
    typewords["non-joke"] = sum(out2["non-joke"].values())
    

    # + operation will combine both Counter from joke and non-joke, and from there, 
    # we can get its unique vocabulary words
    out4 = out2["joke"] + out2["non-joke"]
    # Number of unique vocab
    out3 = len(out4)
    
    if out3 != len(sets):
        print("Wait the total number of unique vocab is different")

    for word in out4.keys():
        for types in out2.keys():
            if out2[types][word]:
                out2[types][word] = math.log((out2[types][word] + 1.0) / (typewords[types] + out3))
                # out2[types][word] = ((out2[types][word] + 1.0) / (typewords[types] + out3))
            else:
                out2[types][word] = math.log((0 + 1.0) / (typewords[types] + out3))
                # out2[types][word] = ((0 + 1.0) / (typewords[types] + out3))

    # print(out1)
    # print(out2)
    # print(out3)
    # print(typewords)
    # top_10_joke = out2["joke"].most_common(10)
    # for couple in top_10_joke:
    #     print(couple)
    # print(f"-----------------------------------------")
    # top_10_nonjoke = out2["non-joke"].most_common(10)
    # for couple in top_10_nonjoke:
    #     print(couple)
    return out1, out2, out3, typewords

def testNaiveBayes(testfile, out1, out2, out3, typewords):
    '''
    input: the file path to be used for test;
    input: the output produced by trainNaiveBayes;
    output: predicted class (the string “joke” or the string “non-joke”. You can assume these to be
    the only classes to be predicted)
    
    The tokens that are not in the vocabulary should have smoothing applied.
    '''
    result = "unknown"
    text = testfile["text"]
    text = Counter(text)
    types_prob = {"joke" : 0.0, "non-joke" :  0.0}
    for types in types_prob:
        for word in text:
            value = 0
            if out2[types][word]:
                value = out2[types][word] * text[word]
            else:
                value = math.log((0 + 1.0) / (typewords[types] + out3)) * text[word]
                # value = ((0 + 1.0) / (typewords[types] + out3)) * text[word]
            types_prob[types] += value
        types_prob[types] += out1[types]
    result =  max(types_prob, key=types_prob.get)
    return result

def main(datafolder):
    '''
    Main function.
    The main function should be run using command like :
    python3 naivebayes.py jokes/
    python3 naivebayes.py folder2/
        
    The main program should perform the following sequence of steps:
    # STEP 1
    i. open the folder containing the data files, included in the folder provided as an argument on the
        command line (e.g., jokes/), and read the list of files from this folder. Assuming the folder has N files
        (e.g., jokes/ includes 400 files).
    
    # STEP 2
    For the evaluations on both datasets, use the leave-one-out strategy. That is, assuming there are N
        files in a dataset, train your Naive Bayes classifier on N-1 files, and test on the remaining one file.
        Repeat this process N times, using one file at a time as your test file.

    
    Repeat N times:
    STEP 3
    i. select one file as test, and the remaining N-1 as training (ex: the first iteration should use
        the first file as testing; the second iteration should use the second file as testing; and so on.)
        iteration).
    STEP 4
    ii. apply the trainNaiveBayes followed by the testNaiveBayes functions.
    STEP 5 
    iii. determine if the class assigned by the testNaiveBayes function is correct
    
    It should also display (standard output) the accuracy of your classifier, calculated as the total
        number of files for which the class predicted by your implementation coincides with the correct
        class, divided by the total number of files.
    '''
    files = os.listdir(datafolder)
    i = 0
    dicti = {}
    
    # STEP 1
    for file in files:
        if file.startswith("joke"):
            dicti[file] = {"original" : "joke", "result" : ""}
        else:
            dicti[file] = {"original" : "non-joke", "result" : ""}
        # print(file)
        with open(f"{datafolder}/{file}",'r') as f:
            text = f.read().lower()
            removedText = removeSGML(text)
            token_list = tokenizeText(removedText)
            dicti[file]["text"] = token_list
        i += 1
    # print(i)
    # STEP 2
    # STEP 3 : using file as testfile
    count = 0
    # count_non_joke_error = 0
    for testfile in dicti:
        copy_of_dicti = copy.deepcopy(dicti)
        copy_of_dicti.pop(testfile)
        # STEP 4
        out1, out2, out3, totalwords = trainNaiveBayes(copy_of_dicti)
        result = testNaiveBayes(dicti[testfile], out1, out2, out3, totalwords)
        # STEP 5
        dicti[testfile]["result"] = result
        if dicti[testfile]["original"] != dicti[testfile]["result"]:
            count += 1
    if datafolder[-1] == "/":
        datafolder = datafolder[:-1]
    with open(f'naivebayes.output.{datafolder}', 'w') as f:
        for key in dicti.keys():
            result = dicti[key]["result"]
            f.write(f"{key} {result}\n")
    # Accuracy 
    print(f"{(i-count)/i * 100} accuracy")
    return None


if __name__ == "__main__":
    main(sys.argv[1])
    
    
# if result == "joke":
#     count_non_joke_error += 1
# original = dicti[testfile]["original"]
# print(f"{testfile} has different results! {original} | {result}")
# print(f"{count} difference detected")
# print(count_non_joke_error)
# print(f"{(i-count)/i} accuracy accomplished")