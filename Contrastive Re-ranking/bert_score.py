import os
import bert_score
import numpy as np

folder_path = r'/content/drive/MyDrive/SimCLS-main/result/candidate/' --path to candidate summary
folder_path2 = r'/content/drive/MyDrive/SimCLS-main/result/reference/' --path to reference summary
bert_scores = []

def calculate_bert_score(hypotheses, references):
    P, R, F1 = bert_score.score(hypotheses, references, lang='en', rescale_with_baseline=True)
    return F1.tolist()
summary_files = [file for file in os.listdir(folder_path) if file.endswith('.dec')]

for summary_file in summary_files:
        reference_file = summary_file.replace('candidate','reference').replace('dec','ref')
        with open(folder_path+summary_file) as file:
            summaries = file.read().rstrip()

        with open(folder_path2+reference_file) as file:
            references = file.read().rstrip()

        if not isinstance(summaries, list):
            summaries = [summaries]

        if not isinstance(references, list):
            references = [references]

        # Calculate BERT scores
        bert_scores.append(calculate_bert_score(summaries, references))
score = np.mean(bert_scores)
print(f'BERT Score:', score*100)
