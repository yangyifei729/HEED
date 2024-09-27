import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
import string
nltk.download('wordnet')
import re
from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()

print("stop words")
print(stopwords.words("english"))
print("punctuation")
print(string.punctuation)
stop_words_list = []
stop_words = set(stop_words_list + list(string.punctuation) + ['""', "''", '``', "'", "`"])
pattern = r"\(.*?\)|\{.*?\}|\[.*?\]|\<.*?\>"
pattern2 = r"in the (.*)"

green_list = ["apple (company)", "boo (programming language)",
              "question answering (natural language processing)",
              "question answering systems", "visual question answering"]
green_word_list = ["&", ",", ":", "#"]
# from nltk.tokenize.treebank import TreebankWordDetokenizer
def untokenize(words):
    """
    Untokenizing a text undoes the tokenizing operation, restoring
    punctuation and spaces to the places that people expect them to be.
    Ideally, `untokenize(tokenize(text))` should be identical to `text`,
    except for line breaks.
    """
    text = ' '.join(words)
    step1 = text.replace("`` ", '"').replace(" ''", '"').replace('. . .',  '...')
    step2 = step1.replace(" ( ", " (").replace(" ) ", ") ")
    step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
    step4 = re.sub(r' ([.,:;?!%]+)$', r"\1", step3)
    step5 = step4.replace(" '", "'").replace(" n't", "n't").replace(
         "can not", "cannot")
    step6 = step5.replace(" ` ", " '")
    step7 = step6.replace(" ’ ", "’")
    return step7.strip()

with open("./data/quora_label_space_merged_clean_v2.tsv", 'w', encoding='utf8') as fout:
    with open("./data/quora_label_space_merged.tsv", 'r', encoding='utf8') as f:
        data = f.readlines()
        for index, line in enumerate(data):
            # if index > 100:
            #     break
            line = line.rstrip()
            topic, count = line.split("\t")
            # topic = "books for c#"
            if topic in green_list:
                topic2 = topic
            else:
                topic1 = re.sub(pattern, "", topic).rstrip()
                topic2 = re.sub(pattern2, "", topic1).rstrip()

            tokenized_word = nltk.word_tokenize(topic2)
            if topic in green_list:
                cleaned_word = topic.split(" ")
            else:
                cleaned_word = [word for word in tokenized_word if (word not in stop_words) or (word in green_word_list)]
            untokenize_word = untokenize(cleaned_word)
            # print(untokenize_word)
            # if topic != untokenize_word:
                # fout.write("{}\t{}\t{}\n".format(topic, count, untokenize_word))
            fout.write("{}\t{}\t{}\n".format(topic, count, untokenize_word))
