import spacy
import re
import nltk
from nltk.corpus import stopwords
import gensim
from nltk.corpus import stopwords
import string


class NameEntityRecognition:
    def __init__(self, document):
        self.general_word = ['Mr.', 'Ms.', 'The', 'and', 'email']
        self.stop_word = stopwords.words('english') + self.general_word
        self.exclude_punctuation = set(string.punctuation)
        self.document = document
        self.hobbies = ['fishing', 'Art', 'Biking', 'Bingo', 'game', 'Boating', 'Camping',
                        'Carving', 'Cooking', 'Dancing', 'Gambling', 'Games',
                        'Hiking', 'Hunting', 'skating', 'Jogging', 'Magic', 'Mountaineering',
                        'Painting', 'Reading', 'Riding', 'climbing', 'Shopping',
                        'shooting',  'Sports', 'Surfing', 'Yoga', 'tennis', 'baseball',
                        'basketball', 'puck', 'Snooker']
        self.exclude_punctuation = set(string.punctuation)
        self.model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz',
                                                                     binary=True, encoding="ISO-8859-1")

    def pre_processing(self, document):
        document = ' '.join([i for i in document.split() if i not in self.stop_word])
        sentences = nltk.sent_tokenize(document)
        sentences = [nltk.word_tokenize(sent) for sent in sentences]
        sentences = [nltk.pos_tag(sent) for sent in sentences]
        return sentences

    def extract_email_addresses(self):
        r = re.compile(r'[\w\.-]+@[\w\.-]+')
        return r.findall(self.document)

    def entity_extract_person_location_birthday(self):
        person = ''
        location = ''
        birthday = ''
        nlp2 = spacy.load('en')
        doc2 = nlp2(self.document)
        information = [(X.text, X.label_) for X in doc2.ents]
        for item in information:
            if 'PERSON' in item[1]:
                person = person + item[0]
            elif 'GPE' in item[1]:
                location = location + item[0]
            elif 'DATE' in item[1]:
                birthday = birthday + item[0]
        return person, location, birthday

    # language backup words
    def entity_extract_language(self):
        language_en = ''
        nlp2 = spacy.load('en')
        doc2 = nlp2(self.document)
        information = [(X.text, X.label_) for X in doc2.ents]
        for item in information:
            if 'NORP_LANGUAGE' in item[1]:
                language_en = language_en + item[0]
        print(language_en)
        return language_en

    # training engine by your self
    def entity_extract_position(self):
        # new entity
        nlp = spacy.load('NER_position')
        doc = nlp(self.document)
        information = [(X.text, X.label_) for X in doc.ents]
        for item in information:
            if 'Position' in item[1]:
                return item[0]
        return 'no position entity in this sentence.'

    # test it by word embedding distance
    def hobbies_similarity_pro(self, list_hobby):
        hobbies = list_hobby
        sum_similarity_value = 0
        for item in range(len(hobbies) - 1):
            for i in range(item, len(hobbies) - 1):
                sum_similarity_value = sum_similarity_value + self.model.similarity(hobbies[item], hobbies[i + 1])
        hobbies_similarity_average_value = sum_similarity_value / (float(len(hobbies) * (len(hobbies) - 1)) / 2.0)
        return hobbies_similarity_average_value

    def hobbies_similarity(self, list_hobby):
        hobbies = list_hobby
        sum_similarity_value = 0
        for item in range(len(hobbies) - 1):
            sum_similarity_value = sum_similarity_value + self.model.similarity(hobbies[item], hobbies[item + 1])
        hobbies_similarity_average_value = sum_similarity_value / (float(len(hobbies) - 1))
        return hobbies_similarity_average_value
        # about 0.18

    def noun_chunk(self):
        nlp = spacy.load('en')
        doc = nlp(self.document)
        list_noun_chunk = ' '.join(str(chunk) for chunk in doc.noun_chunks)
        return list_noun_chunk

    def entity_extract_hobbies_processing(self, list_chunk, extracted_entity):
        string_no_stop = " ".join([i for i in list_chunk.split() if i not in self.stop_word and len(i) > 2])
        document_clean = " ".join(ch for ch in string_no_stop.split() if ch not in self.exclude_punctuation)
        for item in extracted_entity:
            if item in document_clean:
                document_clean = document_clean.replace(item, '')
        return document_clean.split(' ')

    def similarity_value(self, backup_words):
        for item in backup_words:
            sum_value = 0
            for i in self.hobbies:
                sum_value = sum_value + self.model.similarity(item, i)
            # hobbies_similarity 0.18
            if sum_value/float(len(self.hobbies)) > 0.18:
                print('The candidate word to describe bobbies:', item)


if __name__ == '__main__':
    test_corpus = 'Hello, give me some help. I want to know who is Mr. Tony Fank. He is CEO,  who speak German.' \
                  'And he is working in Frankfurt, working email is liang2jay@gmail.com. ' \
                  'I am searching someone play football. '
    start = NameEntityRecognition(test_corpus)
    email = start.extract_email_addresses()
    person, location, birthday = start.entity_extract_person_location_birthday()
    position = start.entity_extract_position()
    language = start.entity_extract_language()
    list_entity = [person, location, birthday, position] + language + email
    backup_word = start.entity_extract_hobbies_processing(start.noun_chunk(), list_entity)
    start.similarity_value(backup_word)





