import nltk as nl
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
import numpy as np
import string
import csv

# stop words and porter stemmer from nltk library
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

train_path = input("Please enter in the labeled training documents path: ")
# unforunate if the path to test set is called 'Y'
test_path = input("Please enter in the unlabeled test documents path. If you would like to split the training set into the test set, please enter 'Y': ")
output_path = input("Please enter in an output file name: ")

# split data into train and test set appropriately (80% train, 20% test)
train_size = 0.8
test_size = 0.2

# split train data into train and test set if it's not specified in the path
df = pd.read_csv(train_path, delimiter=' ', header=None)
if test_path.lower() == 'y':
    train, test = np.split(df.sample(frac=1, random_state=42), [int(train_size * len(df))])
else:
    test = pd.read_csv(test_path, delimiter=' ', header=None)
    train = df


# train the model
classifiers = {}
total_documents = 0

for row in train.itertuples(index=False):
    article = row[0]
    classifier = row[1]
    
    # initialize classifiers and sum total documents
    if classifier not in classifiers:
        classifiers[classifier] = {'instances': 0, 'tokens': {}, 'total_token_count': 0}

    classifiers[classifier]['instances'] += 1
    total_documents += 1

    try:
        with open(article, 'r') as file:
            content = file.read()
            tokens = nl.word_tokenize(content)
            tokens = [ps.stem(token.lower()) for token in tokens if token not in string.punctuation and token not in ["''", "``"] and token.lower() not in stop_words]

            # seen_tokens = set()

            for token in tokens:
                # if token not in seen_tokens:
                    # seen_tokens.add(token)

                if token not in classifiers[classifier]['tokens']:
                    classifiers[classifier]['tokens'][token] = 1
                else:
                    classifiers[classifier]['tokens'][token] += 1

                classifiers[classifier]['total_token_count'] += 1

    except FileNotFoundError:
        print(f"File {article} when training not found.")
        continue


def predict_category(file_path) -> dict:
    results = {}

    try:
       with open(file_path, 'r') as file:
            content = file.read()
            tokens = nl.word_tokenize(content)
            tokens = [ps.stem(token.lower()) for token in tokens if token not in string.punctuation and token not in ["''", "``"] and token.lower() not in stop_words]

            # vocabulary size (unique tokens across all classes)
            vocabulary = set()
            for c in classifiers:
                vocabulary.update(classifiers[c]['tokens'].keys())
            vocabulary_size = len(vocabulary)

            for c in classifiers:
                
                # P(c)
                results[c] = np.log(classifiers[c]['instances'] / total_documents)

                for token in tokens:

                    # if token in classifiers[c]['tokens']:
                        # results[c] += np.log(classifiers[c]['tokens'][token] / classifiers[c]['instances'])
                    # else:

                    token_count = classifiers[c]['tokens'].get(token, 0)

                    # laplace smoothing
                    token_probability = (token_count + 1) / (classifiers[c]['total_token_count'] + vocabulary_size)
                    results[c] += np.log(token_probability)

    except FileNotFoundError:
        print(f"File at {file_path} when testing not found.")
        return None

    return results


# predict and store results and write prediction to output file
final_results = {}

with open(output_path, 'w', newline='') as result_file:
    writer = csv.writer(result_file, delimiter=' ', lineterminator='\n')

    for row in test.itertuples(index=False):
        article = row[0]

        category_scores = predict_category(article)
        predicted_classifier = max(category_scores, key=category_scores.get)

        # print to output file in [article class] format
        writer.writerow([article, predicted_classifier])

        if test_path.lower() == 'y':
            correct_classifier = row[1]
            if correct_classifier not in final_results:
                final_results[correct_classifier] = {}

            if predicted_classifier not in final_results[correct_classifier]:
                final_results[correct_classifier][predicted_classifier] = 0

            final_results[correct_classifier][predicted_classifier] += 1


# exit program (don't create contingency table) if the test set is specified as we do not know correct labels
# this is different than splitting the training data into train and test sets as we DO know the correct testing labels in this case
if test_path.lower() != 'y':
    print(f"Wrote predicted classifiers in [article, class] format to the file: '{output_path}'")
    exit()

################################ create contingency table & calculate stats

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
    
    # calculate recall while accounting for div by 0
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
