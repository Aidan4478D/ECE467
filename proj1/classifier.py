import nltk as nl
import pandas as pd
import numpy as np
import string
import json




# Read the file into a DataFrame
df = pd.read_csv('TC_provided/corpus1_train.labels', delimiter=' ')

classifiers = {}
total_documents = 0

# get classifiers, their instances, and the total documents
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
    
    results = {}

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
                        # could try different smoothing methods
                        results[c] += np.log(0.001)

    except FileNotFoundError:
        print(f"file at {article[1:]} not found.")
        return

    return results




final_results = {}
df = pd.read_csv('TC_provided/corpus1_test.labels', delimiter=' ')

# Predict and store results
for row in df.itertuples(index=False):
    article = row[0]
    correct_classifier = row[1]
    
    # Get predicted classifier
    category_scores = predict_category('TC_provided' + article[1:])
    predicted_classifier = max(category_scores, key=category_scores.get)

    # Store prediction results in the dictionary
    if correct_classifier not in final_results:
        final_results[correct_classifier] = {}
    
    if predicted_classifier not in final_results[correct_classifier]:
        final_results[correct_classifier][predicted_classifier] = 0

    final_results[correct_classifier][predicted_classifier] += 1

# print(final_results)


# fill contingency table based off final results
categories = list(classifiers.keys())
contingency_table = pd.DataFrame(0, index=categories, columns=categories)

for true_label, predictions in final_results.items():
    for predicted_label, count in predictions.items():
        contingency_table.at[true_label, predicted_label] += count


# calculate precision and recall
precision = {}
recall = {}

total_correct = 0
total_incorrect = 0

for c in categories:

    A = contingency_table.at[c, c]
    total_correct += A
    
    A_B = contingency_table.loc[c].sum()
    A_C = contingency_table[c].sum()

    total_incorrect += A_B - A
    
    # calculate precision while accounting for div by 0
    if A_B > 0:
        precision[c] = round(A / A_B, 2)
    else:
        precision[c] = 0.0
    
    # calculate recall while accoting for div by 0
    if A_C > 0:
        recall[c] = round(A / A_C, 2)
    else:
        recall[c] = 0.0


# Convert the main part of the contingency table to integers
contingency_table.iloc[:, :-1] = contingency_table.iloc[:, :-1].astype(int)

# Add Precision and Recall to the contingency table
contingency_table['PRE'] = pd.Series(precision)
contingency_table.loc['REC'] = pd.Series(recall)

ratio = 0
if total_incorrect + total_correct > 0:
    ratio = round(total_correct/(total_incorrect + total_correct), 3)

print(f'\n{total_correct} CORRECT, {total_incorrect} INCORRECT, RATIO = {ratio}\n')

print("CONTINGENCY TABLE")
print(str(contingency_table) + '\n')

for c in categories:
    precision_val = contingency_table.at[c, 'PRE']
    recall_val = contingency_table.at['REC', c]
    
    if (precision_val + recall_val) > 0:
        f1 = 2 * precision_val * recall_val / (precision_val + recall_val)
    else:
        f1 = 0.0

    print(f'F1 ({c}) = {round(f1, 3)}')

