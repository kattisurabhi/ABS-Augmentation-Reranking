import re
import pickle

#preprocess article
def modify_article(example):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', example)

    output = ['"{}"{}'.format(sentence.strip(), ',"' if index < len(sentences) - 1 else '",') for index, sentence in enumerate(sentences[:-1])]
    output.append('"{}".'.format(sentences[-1].strip()))

    return output

def modify_pickle_file(input_pickle_path, output_pickle_path):
    with open(input_pickle_path, 'rb') as file:
        examples = pickle.load(file)
    modified_examples = [{"article": modify_article(example)} for example in examples]
    with open(output_pickle_path, 'wb') as file:
        pickle.dump(modified_examples, file)
input_pickle_path = 'article_input_without_mask.pickle'
output_pickle_path = 'modified_article.pkl'

modify_pickle_file(input_pickle_path, output_pickle_path)

#preprocess reference summary
def modify_ref(example):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', example)
    output = ['"{}"{}'.format(sentence.strip(), ',"' if index < len(sentences) - 1 else '",') for index, sentence in enumerate(sentences[:-1])]
    output.append('"{}".'.format(sentences[-1].strip()))

    return output

def modify_pickle_file_ref(input_pickle_path, output_pickle_path):
    with open(input_pickle_path, 'rb') as file:
        examples_ref = pickle.load(file)
    modified_refs = [{"abstract": modify_ref(example)} for example_r in examples_ref]
    with open(output_pickle_path, 'wb') as file:
        pickle.dump(modified_refs, file)
input_pickle_path = 'summary_input_without_mask.pickle'
output_pickle_path = 'modified_ref_summary.pkl'

modify_pickle_file_ref(input_pickle_path, output_pickle_path)

#preprocess generated summaries
import pickle

def modify_sentences(data):
    modified_data = []

    for example in data:
        candidates = example['candidates']
        modified_candidates = []

        for candidate in candidates:
            sentences = candidate[0][0]  # Extracting the sentences from nested lists
            modified_sentences = []

            for sentence_info in sentences.split(". "):
                modified_sentence = f'"{sentence_info.strip()}"'
                modified_sentences.append(modified_sentence)

            modified_candidates.append([modified_sentences, candidate[1]])

        modified_data.append({'candidates': modified_candidates})

    return modified_data
with open('modified_gen_summary.pkl', 'rb') as file:
    input_data = pickle.load(file)
modified_data = modify_sentences(input_data)

with open('preprocess_gen_summary.pkl', 'wb') as file:
    pickle.dump(modified_data, file)

#final preparation 
import pickle

def format_data(article_data, summary_data, candidates_data):
    formatted_data = []

    for article, summary, candidates_dict in zip(article_data, summary_data, candidates_data):
        formatted_example = {
            "article": article["article"],
            "abstract": summary["abstract"],
            "candidates": []
        }

        for candidates_list in candidates_dict['candidates']:
            formatted_example["candidates"].append([candidates_list[0], candidates_list[1]])

        formatted_data.append(formatted_example)

    return formatted_data

with open('modified_article.pkl', 'rb') as f:
    article_data = pickle.load(f)

with open('modified_ref_summary.pkl', 'rb') as f:
    summary_data = pickle.load(f)

with open('preprocess_gen_summary.pkl', 'rb') as f:
    candidates_data = pickle.load(f)

formatted_data = format_data(article_data, summary_data, candidates_data)
output_file_path = 'final_formatted_data.pkl'
with open(output_file_path, 'wb') as f:
    pickle.dump(formatted_data, f)

print(f"Formatted data saved to {output_file_path}")
