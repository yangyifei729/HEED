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

# print("stop words")
# print(stopwords.words("english"))
# print("punctuation")
# print(string.punctuation)
# stop_words_list = []
# stop_words = set(stop_words_list + list(string.punctuation) + ['""', "''", '``', "'", "`"])
pattern = r"\(.*?\)|\{.*?\}|\[.*?\]|\<.*?\>"
pattern2 = r"in the (.*)"

# stop_words_list = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll",
#                    "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's",
#                    'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs',
#                    'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am',
#                    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
#                    'did', 'doing', 'a', 'an', 'the', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
#                    'once', 'here', 'there', 'when', 'where',
#                    'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
#                    'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just',
#                    'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren',
#                    "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn',
#                    "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't",
#                    'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't",
#                    'won', "won't", 'wouldn', "wouldn't", "where's", "how's", "what's", "'s"]

# topic = '"u.s. is in decline" theme'
# topic1 = re.sub(pattern, "", topic).rstrip()
# topic2 = re.sub(pattern2, "", topic1).rstrip()
# tokenized_word = nltk.word_tokenize(topic2)
# cleaned_word = [word for word in tokenized_word if word not in stop_words]
# print(tokenized_word)
#
# print(topic1)
# print(topic2)
# print(tokenized_word)
# print(cleaned_word)


with open("./data/quora_label_space_merged_clean.tsv", 'w', encoding='utf8') as fout:
    with open("./data/quora_label_space_merged.tsv", 'r', encoding='utf8') as f:
        data = f.readlines()
        for index, line in enumerate(data):
            # if index > 100:
            #     break
            line = line.rstrip()
            topic, count = line.split("\t")
            # topic = "the emperor's new groove (2000 movie)"
            # count=1
            # topic = 'homes and houses (self home)'
            topic1 = re.sub(pattern, "", topic).rstrip()
            topic2 = re.sub(pattern2, "", topic1).rstrip()
            tokenized_word = nltk.word_tokenize(topic2)
            cleaned_word = [word for word in tokenized_word if word not in stop_words]
            lemma = [lmtzr.lemmatize(word) for word in cleaned_word]
            lemma = " ".join(lemma)
            if not lemma:
                lemma = topic
                print("no lemma", topic)

            # print("~~~~~~~~~~~")
            # print(topic)
            # print(lemma)
            # if topic != lemma:
            # print(lemma)
            if "covid" in lemma and "19" in lemma:
                lemma="covid-19"
            fout.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(topic, count, topic1, topic2, ' '.join(cleaned_word), lemma))

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
            if ("covid" in untokenize_word and "19" in untokenize_word) or "covid-19" in untokenize_word:
                untokenize_word = "covid-19"
            # print(untokenize_word)
            # if topic != untokenize_word:
                # fout.write("{}\t{}\t{}\n".format(topic, count, untokenize_word))
            fout.write("{}\t{}\t{}\n".format(topic, count, untokenize_word))
            # print(cleaned_word)
            # print(untokenize(cleaned_word))

            # gg
            # gg
            # gg
# pattern2 = r'\(.*?\)|\{.*?\}|\[.*?\]|\<.*?\>'
# pattern2 = r"in the (.*)"
# topic = "job search advice in the united states of america"
# clean_topics = re.sub(pattern2, "", topic).rstrip()
# print(clean_topics)
# import re
# s = "我是中国人(住在北京)666[真的]bbbb{确定}<kkk>"
# pattern = r"\(.*?\)|\{.*?\}|\[.*?\]|\<.*?\>"
# # a = re.sub(, "", s)
#
# with open("./data/quora_label_space_merged.tsv", 'r', encoding='utf8') as f:
#     data = f.readlines()
#     count = 0
#     for index, line in enumerate(data):
#         line = line.rstrip()
#         topic, rank = line.split("\t")
#         if "(" in topic and ")" in topic:
#             print("~~~~~~~~~")
#             print(topic)
#             print(re.sub(pattern, "", topic).rstrip())
#             count += 1
#     print(count)

