# Farhan Ariff bin Halis Azhan
# farharif
# Assignment1

import os
import sys
from bs4 import BeautifulSoup
from collections import Counter

def removeSGML(stri):
    '''
        Function that removes the SGML tags from the input string.
        Input:
        1. string

        Output:
        1. string
    '''
    
    ret = BeautifulSoup(stri, features="html.parser").text
    ret = ret.replace('\n'," ")
    return ret

def check_pronoun(maybe):
    if maybe == "she" or maybe == "he" or maybe == "they" \
    or maybe == "it":
        return True
    return False

def tokenizeText(stri):
    '''
        Function that tokenizes the text.
        Input:
        1. string

        Output:
        1. list (of tokens)

        The tokenizer should not consider punctuation on its own as a token.
    '''
    l_o_token = []
    text = stri.strip()
    text = text.split(' ')
    # print(f"3 : {text}")
    # get word
    for word in text:
        if word != '' and word != '.':
            if ',' in word:
                if word[-1] == ',':
                    n = len(word) - 1
                    word = word[:n]
                else:
                    indexs = word.index(',')
                    if word[indexs-1] != " " and word[indexs+1] != " ":
                        words = list(word)
                        words[indexs] = " "
                        word = ''.join(words)
            elif word[-1] == '.':
                n = len(word) - 1
                word = word[:n]
            if word.isnumeric():
                continue
            l_o_token.append(word)

    # Putting space (start token) to first word
    # l_o_token[0] = " " + l_o_token[0]

    # Putting space (start token) to each word
    # for i in range(len(l_o_token)):
    #     l_o_token[i] = " " + l_o_token[i]
    return l_o_token

### HELPER FUNCTION FOR BPE BELOW

def calculate_character_pair_freq(char_text, dict_token):
    pair_freq = {}
    for k, cs in char_text.items():
        for i in range(len(cs) - 1):
            charac_pair = (cs[i], cs[i+1])
            if charac_pair in pair_freq:
                pair_freq[charac_pair] += dict_token[k]
            else:
                pair_freq.setdefault(charac_pair, dict_token[k])
    return pair_freq

def find_common_pair_func(pair_freq):
    most_common_pair = max(pair_freq, key=pair_freq.get)
    return most_common_pair

def merge_word(v,i,pair_char):
    return v[:i] + pair_char + v[i + 2 :]

def merge_rules_func(char_text, most_common_pair):
    ret = char_text
    for (k,v) in ret.items():
        i = 0
        vlength = (len(v))
        vlength = vlength - 1
        while i < vlength:
            firstbool = v[i] == most_common_pair[0]
            secondbool = v[i+1] == most_common_pair[1]
            if firstbool and secondbool:
                pair_char = [most_common_pair[0] + most_common_pair[1]]
                v = merge_word(v, i, pair_char)
                vlength = (len(v) - 1)
            else:
                i += 1
        ret[k] = v
    return ret

### HELPER FUNCTION FOR BPE ABOVE

def BPE(list_o_token, vocabSize):
    '''
        Function that performs Byte-Pair Encoding tokenization.
        Input:
        1. list (of tokens)
        2. vocabSize

        Output:
        1. list (of subword tokens)
        2. list (of merge rules)

        This tokenizer is going to split the tokens obtained from the tokenizeText function into subwords.
        Please refer to BPE in the lecture slides. Your BPE implementation should contain the following
        parts (either as functions or as code blocks).

        Note: for the purpose of this assignment, 
        you will not perform two separate BPE training and BPE tokenization steps; 
        you will only apply what is referred to in the slides as “BPE training”
    '''
    # initialize the output variables
    l_subword_token = []
    l_merge_rules = []
    ### i) calculate character frequencies, save the initial set of characters as your vocabulary
    # list to string (text)
    text = ''.join(list_o_token)
    # res variable assgined to the counter of text
    res = Counter(text)
    # getting the initial vocab
    vocab = res
    # create a dictionary from token
    dict_token = dict(Counter(list_o_token))
    char_text = {}
    for k in dict_token.keys():
        char_text[k] = [char for char in k]
    i = 0
    while len(vocab) < vocabSize:
        print(f"ROUND {i}")
        ### ii) calculate character pair frequencies
        pair_freq = calculate_character_pair_freq(char_text, dict_token)
        ### iii) merge the most common pair in the character frequencies,
        # save the resulting pair in the vocabulary
        # and save the merge rule
        most_common_pair = find_common_pair_func(pair_freq)
        vocab += Counter({''.join(most_common_pair) : pair_freq[most_common_pair]})
        # print(vocab)
        l_merge_rules.append({most_common_pair : pair_freq[most_common_pair]})
        char_text = merge_rules_func(char_text, most_common_pair)
        i += 1
        # if len(vocab) == vocabSize:
        #     print(f"vocab size of {len(vocab)}/{vocabSize} of  reached")
        # if i == 50:
        #     break
    ### iv) Main loop where you perform steps ii) and iii) until you reach the desired vocabulary size
    l_subword_token = vocab
    return l_subword_token, l_merge_rules

def main(folderpath, vocabsize):
    '''
        Main function.

        It should produce a file called preprocess.output with the following content (tokens are to be listed
        in descending order of their frequency; ties can be listed in any order):

        # i. open the folder containing the data collection, provided as the first argument on the command line
        # (e.g., cranfieldDocs/), and read one file at a time from this folder. Hint: use encoding IS0-8859-1
        # when opening files.
        # ii. for each file, apply, in order: removeSGML, tokenizeText, BPE
        # iii. in addition, write code to determine and list (this is after step ii above):
        # - the total number of merge rules learned by your BPE tokenizer
        # - the first 20 merge rules learned by your BPE tokenizer
        # - most frequent 50 BPE tokens in the collection, along with their frequencies (list in
        #   descending order of their frequency, i.e., from most to least frequent)
    '''
    # Step 1
    # change the directory
    cwd = os.getcwd()
    os.chdir(folderpath)
    list_o_token = []
    # read file
    for file in os.listdir():
        file_path = f"{file}"
        with open(file_path, 'r', encoding="ISO-8859-1") as f:
            text = f.read()
            remSGML = removeSGML(text)
            list_token = tokenizeText(remSGML)
            for token in list_token:
                list_o_token.append(token)
    # BPE training
    l_subword_token, l_merge_rules = BPE(list_o_token, int(vocabsize))
    # print("Done")
    # print(f"Printing merge_rules : {l_merge_rules} : {len(l_merge_rules)}")
    # print(f"Printing vocab : {l_subword_token} : {len(l_subword_token)}" )
    total = 0
    # j = 0
    # done = False
    for values in l_subword_token.values():
        total += values
        # j += 1
        # if (total > 557973.5) and not done:
        #     print("The minimum number of unique BPE tokens \n"
        #           "in the Cranfield collection accounting for 25% \n" 
        #           "of the total number of BPE tokens in \n"
        #           f"the collection is {j} tokens")
        #     done = True
    # Make an output file
    os.chdir(cwd)
    with open('preprocess.output', 'w') as f:
        f.write(f"Tokens {total}")
        f.write(f"\nMerge rules {len(l_merge_rules)}")
        f.write(f"\nThe first 20 merge rules")
        for i in range(20):
            merge_rules = list(l_merge_rules[i].keys())[0]
            merged = ''.join(merge_rules)
            f.write(f"\n{merge_rules} -> {merged}")
        top_bpe = l_subword_token.most_common(50)
        f.write(f"\nTop 50 tokens")
        for token in top_bpe:
            f.write(f"\n'{token[0]}': {token[1]}")
    # return None
    return None

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])