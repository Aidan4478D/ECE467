import nltk as nl
import pandas as pd
import numpy as np
import string
import json

# nl.download()

sentence = "hello my name is bob"

# using punkt_tab tokenizer
# tokens = nl.word_tokenize(sentence)
# print(tokens)

# Read the file into a DataFrame
df = pd.read_csv('TC_provided/corpus1_train.labels', delimiter=' ')

classifiers = {}
total_documents = 0

for c in df[df.columns[1]]:
    if c not in classifiers:
        classifiers[c] = {'instances': 1, 'tokens': {}}
    else:
        classifiers[c]['instances'] += 1

    total_documents += 1

# print(total_documents)


# add instances of terms to classifiers dict
# works on a level of if a term is seen once in the document it's added to the dict
# not # of occurances but # of documents term occurs in
for row in df.itertuples(index=False):
    article = row[0]
    classifier = row[1]

    try:
        with open('TC_provided' + article[1:], 'r') as file:
            content = file.read()
            tokens = nl.word_tokenize(content)
            tokens = [token for token in tokens if token not in string.punctuation and token not in ['``', "''"]]

            seen_tokens = set()

            for token in tokens:
                if token not in seen_tokens:
                    seen_tokens.add(token)

                    if token not in classifiers[classifier]['tokens']:
                        classifiers[classifier]['tokens'][token] = 1
                    else:
                        classifiers[classifier]['tokens'][token] += 1
            # print(f"Tokens for {classifier}: {tokens}")
            # break

    except FileNotFoundError:
        print(f"file {article[1:]} not found.")
        continue


# predict category based off of training
def predict_category(file_path) -> dict:
    
    results: dict = {}

    try:
        with open(file_path, 'r') as file:
            content = file.read()
            tokens = nl.word_tokenize(content)
            tokens = [token for token in tokens if token not in string.punctuation and token not in ['``', "''"]]

            for c in classifiers:
                results[c] = np.log(classifiers[c]['instances'] / total_documents)
    
                for token in tokens:
                    if token in classifiers[c]['tokens']:
                        results[c] += np.log(classifiers[c]['tokens'][token] / classifiers[c]['instances'])
                    else:
                        results[c] += np.log(0.000001)

    except FileNotFoundError:
        print(f"file {article[1:]} not found.")
        return

    return results




final_results = {}
df = pd.read_csv('TC_provided/corpus1_test.labels', delimiter=' ')

for row in df.itertuples(index=False):
    article = row[0]
    correct_classifier = row[1]
    
    category_scores = predict_category('TC_provided' + article[1:])
    predicted_classifier = max(category_scores, key=category_scores.get)
    # final_results[article] = predicted_classifier

    if correct_classifier not in final_results:
        final_results[correct_classifier] = {'instances': 1, 'correct': 0}
    else:
        final_results[correct_classifier]['instances'] += 1


    if(predicted_classifier == correct_classifier):
        final_results[correct_classifier]['correct'] += 1


# print(final_results)

total_correct = sum([class_result['correct'] for class_result in final_results.values()])
total_instances = sum([class_result['instances'] for class_result in final_results.values()])

print(total_correct / total_instances)

