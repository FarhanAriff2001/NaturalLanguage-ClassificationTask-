import sys
from collections import Counter
from preprocess import removeSGML, tokenizeText
import math
import csv

def trainNaiveBayes(list_filepaths):
    '''
    '''
    out1 = {"easy" : 0.0, "medium" :  0.0, "hard" : 0.0}
    out2 =  {"easy" : Counter(), "medium" :  Counter(), "hard" : Counter()}
    out3 = 0
    sets = set()
    for key in list_filepaths.keys():
        types = "unknown"
        if list_filepaths[key]["original"] == "easy":
            out1["easy"] +=1
            types = "easy"
        elif list_filepaths[key]["original"] == "medium":
            out1["medium"] +=1
            types = "medium"
        else:
            out1["hard"] +=1
            types = "hard"
        text = list_filepaths[key]["text"]
        out2[types] += Counter(text)
        sets.update(text)

    total = out1["easy"] + out1["medium"] + out1["hard"]
    out1["easy"] = math.log(out1["easy"]/total)
    out1["medium"] = math.log(out1["medium"]/total)
    out1["hard"] = math.log(out1["hard"]/total)

    # Total Number of words for each types
    typewords = {"easy" : 0.0, "medium" :  0.0, "hard" : 0.0}
    typewords["easy"] = sum(out2["easy"].values())
    typewords["medium"] = sum(out2["medium"].values())
    typewords["hard"] = sum(out2["hard"].values())

    # + operation will combine both Counter from easy, medium and hard, and from there, 
    # we can get its unique vocabulary words
    out4 = out2["easy"] + out2["medium"] + out2["hard"]
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

    return out1, out2, out3, typewords

def testNaiveBayes(testfile, out1, out2, out3, typewords):
    '''
    '''
    result = "unknown"
    text = testfile["text"]
    text = Counter(text)
    types_prob = {"easy" : 0.0, "medium" :  0.0, "hard" : 0.0}
    for types in types_prob:
        for word in text:
            value = 0
            if out2[types][word]:
                value = out2[types][word]
            else:
                value = math.log((0 + 1.0) / (typewords[types] + out3))
                # value = ((0 + 1.0) / (typewords[types] + out3)) * text[word]
            types_prob[types] += value
        types_prob[types] += out1[types]
    result =  max(types_prob, key=types_prob.get)
    return result

def get_train_data():
    trainData = {}
    # dicti = {}
    with open("data/train.tsv", 'r') as f:
        content = f.readlines()
        for line in content:
            vals = line.split("\t")
            # print(vals)
            if not vals[0].isdigit():
                continue
            id = int(vals[0])
            text = vals[1]
            text = text.lower()
            removedText = removeSGML(text)
            token_list = tokenizeText(removedText)
            label = vals[2]
            label = label[:-1]
            trainData[id] = {"original" : label, "text" : token_list}
    return trainData

def get_test_data():
    valData = {}
    # dicti = {}
    with open("data/val.tsv", 'r') as f:
        content = f.readlines()
        for line in content:
            vals = line.split("\t")
            # print(vals)
            if not vals[0].isdigit():
                continue
            id = int(vals[0])
            text = vals[1]
            text = text.lower()
            removedText = removeSGML(text)
            token_list = tokenizeText(removedText)
            label = vals[2]
            label = label[:-1]
            valData[id] = {"original" : label, "text" : token_list}
    return valData

def main(datafolder):
    '''
    '''
    trainData = get_train_data()
    labellingData = get_test_data()
    trainLabel = []
    for train in trainData:
        trainLabel.append({"ID" : train, "LABEL" : train["original"]})
    
    # print(trainData)
    out1, out2, out3, totalwords = trainNaiveBayes(trainData)
    
    # #--------------------------------------------------LABELLING FILE END----------------------------------------------------
    for labelId in labellingData:
        result = testNaiveBayes(labellingData[labelId], out1, out2, out3, totalwords)
        labellingData[labelId]["result"] = result
    labelResult = []
    for labelId in labellingData:
        labelResult.append({"ID" : labelId, "LABEL" : labellingData[labelId]["result"]})
    # #--------------------------------------------------LABELLING FILE END----------------------------------------------------

    
    # field names
    fields = ['ID', 'LABEL']
    
    # name of csv file
    filename = "result.csv"
    
    # writing to csv file
    with open(filename, 'w') as csvfile:
        # creating a csv dict writer object
        writer = csv.DictWriter(csvfile, fieldnames=fields)
    
        # writing headers (field names)
        writer.writeheader()
    
        # writing data rows
        writer.writerows(trainLabel)
        writer.writerows(labelResult)


    return None


if __name__ == "__main__":
    main(sys.argv[1])
    
    
    # files = os.listdir(datafolder)
    # i = 0
    # dicti = {}
    
    # # STEP 1
    # for file in files:
    #     if file.startswith("easy"):
    #         dicti[file] = {"original" : "easy", "result" : ""}
    #     elif file.startswith("medium"):
    #         dicti[file] = {"original" : "medium", "result" : ""}
    #     else:
    #         dicti[file] = {"original" : "hard", "result" : ""}
    #     # print(file)
    #     with open(f"{datafolder}/{file}",'r') as f:
    #         text = f.read().lower()
    #         removedText = removeSGML(text)
    #         token_list = tokenizeText(removedText)
    #         dicti[file]["text"] = token_list
    #     i += 1
    # count = 0
    # for testfile in dicti:
    #     copy_of_dicti = copy.deepcopy(dicti)
    #     copy_of_dicti.pop(testfile)
    #     # STEP 4
    #     out1, out2, out3, totalwords = trainNaiveBayes(copy_of_dicti)
    #     result = testNaiveBayes(dicti[testfile], out1, out2, out3, totalwords)
    #     # STEP 5
    #     dicti[testfile]["result"] = result
    #     if dicti[testfile]["original"] != dicti[testfile]["result"]:
    #         count += 1
    # if datafolder[-1] == "/":
    #     datafolder = datafolder[:-1]
    