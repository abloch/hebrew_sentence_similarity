
import logging  # Setting up the loggings to monitor gensim
from itertools import combinations
import matplotlib.lines as mlines
import abc
import multiprocessing
import os
import string
import pandas as pd
from scipy import spatial
from time import time, strftime
import progressbar
import datetime
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import seaborn as sn

from sklearn.metrics import classification_report, confusion_matrix

DEBUG = True
DEBUG_PRINT_SENTENCE = (DEBUG and False)
DEBUG_PRINT_ABOVE_RATIO = (DEBUG and False)


JSON_ENDING = 'json'
CSV_ENDING = 'csv'
JSON_TOKES_STRING = 'tokens'
JSON_SINGLE_TOKEN_STRING = 'token'
CSV_YAP_TOKEN_STRING = 'yap_tokens'

PARSE_TYPE_YAP = 'yap'
PARSE_TYPE_BLANK_TEXT = 'simple_text'
PARSE_TYPE_BLANK = 'blank'

OUTPUT_FOLDER_NAME = 'output'

heb_letters_list = ['ם', 'ץ', 'ף', 'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י',
                 'כ', 'ל', 'מ', 'נ', 'ס', 'ע', 'פ', 'צ', 'ק', 'ר', 'ש', 'ת', 'ך', 'ן']

english_letters_list = list(string.ascii_letters) # 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
digits_list = list(string.digits) # '0123456789'
punctuation_list = list(string.punctuation) #'!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'

class Sentence_Iterator(object):
    def __init__(self, dirname, split_validation=0):
        self._dirname = dirname
        self._split_validation = split_validation # ratio to split by

    @abc.abstractmethod
    def __iter__(self):
        pass

    @staticmethod
    def is_mixed_language(token):
        '''
        Check that a token has mixed hebrew or english or digits
        Will also clean tokens with not supported chars

        '''
        list_chars = set(list(token))

        len_punctuation_intr = len(set(punctuation_list)& list_chars)

        len_heb_intr = len(set(heb_letters_list) & list_chars)
        if len_heb_intr and len(set(token)) == len_heb_intr+len_punctuation_intr:
            return False

        len_eng_intr = len(set(english_letters_list) & list_chars)
        if len_eng_intr and len(set(token)) == len_eng_intr+len_punctuation_intr:
            return False

        len_num_intr = len(set(digits_list) & list_chars)

        if len_num_intr and len(set(token)) == len_num_intr+len_punctuation_intr:
            return False

        return True

    @staticmethod
    def _clean_sentence(sentence):
        '''
        Clean sentence from all punctuations except "
        :param sentence:
        :return:
        '''
        sentence = sentence.replace('-', ' ')
        sentence = sentence.translate(
            sentence.maketrans("", "", ''.join(char for char in string.punctuation if char != '"')))

        return sentence

class Sentences(Sentence_Iterator):
    def __init__(self, dirname, split_validation=0):
        super().__init__(dirname, split_validation)

    def __iter__(self):
        if self._dirname.split('.')[-1] == JSON_ENDING:
            tweets = pd.read_json(self._dirname, encoding="utf8")
        else:
            tweets = pd.read_msgpack(self._dirname, encoding="utf8")
        end_test = len(tweets) - len(tweets)* self._split_validation  # Remove the validation tweets
        for sentence in tweets['clean_text'][:end_test]:
            sentence_list = self._clean_sentence(sentence).split(" ")
            final_sentence_list = []
            for token in sentence_list:
                # Check that this token is not mixed with characters and numbers from different languages and sigits
                if not self.is_mixed_language(token):
                    final_sentence_list.append(token)

            # Removing all punctuation except " since it is used a lot in hebrew for shortenings
            yield final_sentence_list

class Sentences_Yap(Sentence_Iterator):
    def __init__(self, dirname, split_validation=0):
        super().__init__(dirname, split_validation)

    def __iter__(self):

        if not os.path.exists(self._dirname):
            raise RuntimeError("The file path {} dosn't exist".format(self._dirname))

        tweets = pd.read_json(self._dirname, encoding="utf8")

        end_test = len(tweets) - len(tweets)* self._split_validation  # Remove the validation tweets

        # Read all JSON tokens
        for sentence in tweets[JSON_TOKES_STRING][:end_test]:
            # from the sentence dictionary construct the sentence list
            # Parse the tokens and clean them, won't take tokens longer than 15, since most languages don't have
            # such long words, and especially not hebrew
            # Clen  punctuation from the token, except ""
            sentence_list = [token[JSON_SINGLE_TOKEN_STRING].translate(token[JSON_SINGLE_TOKEN_STRING]
                        .maketrans("","", ''.join(char for char in string.punctuation if char!='"')))
                         for token in sentence if len(token[JSON_SINGLE_TOKEN_STRING])< 15]

            final_sentence_list = []
            for token in sentence_list:
                # Check that this token is not mixed with characters and numbers from different languages and sigits
                if not self.is_mixed_language(token):
                    final_sentence_list.append(token)

            yield final_sentence_list


class Sentences_Yap_CSV(Sentence_Iterator):
    def __init__(self, dirname, split_validation=0):
        super().__init__(dirname, split_validation)

    def __iter__(self):

        if not os.path.exists(self._dirname):
            raise RuntimeError("The file path {} dosn't exist".format(self._dirname))

        tweets = pd.read_csv(self._dirname, encoding="utf8")

        end_test = len(tweets) - len(tweets) * self._split_validation  # Remove the validation tweets

        # Read all yap tokens
        for i, sentence in enumerate(tweets[CSV_YAP_TOKEN_STRING][:end_test]):
            # from the sentence dictionary construct the sentence list
            # Parse the tokens and clean them, won't take tokens longer than 15, since most languages don't have
            # such long words, and especially not hebrew
            # Clen  punctuation from the token, except ""
            if type(sentence) != str: # Skip a blank line
                continue
            sentence_list = self._clean_sentence(sentence).split(" ")

            final_sentence_list = []
            for token in sentence_list:
                # Check that this token is not mixed with characters and numbers from different languages and sigits
                if not self.is_mixed_language(token):
                    final_sentence_list.append(token)

            yield final_sentence_list

class Sentences_csv(Sentence_Iterator):
    '''
    This iterator is used for similarity comparision, and will yield two sentences at a time
    '''
    def __init__(self, dirname):
        super().__init__(dirname)

    def __iter__(self):
        sentences = pd.read_csv(self._dirname, header=None, encoding="utf8")

        # For cases when csv is built by sentence in every row
        if len(sentences.values)> 1:
            sentences_list = [sentence[0] for sentence in sentences.values]
        else: # Every column
            sentences_list = [sentence for sentence in sentences]

        # Read every two sentences
        for sentence in sentences_list:
            sentence_list = self._clean_sentence(sentence).split(" ")
            final_sentence_list = []
            for token in sentence_list:
                # Check that this token is not mixed with characters and numbers from different languages and sigits
                if not self.is_mixed_language(token):
                    final_sentence_list.append(token)

            # Removing all punctuation except " since it is used a lot in hebrew for shortenings
            yield final_sentence_list


class SentenceSim:

    _ID_NAME = 'id'

    def __init__(self,
                 use_type,
                 train_path,
                 output_path,
                 input_type='tweets',
                 validation_path=None,
                 log_level = logging.ERROR,
                 batch_size = 500,
                 sim_threshold = 0.9,
                 distance_type='cos',
                 jec_threshold = 0.5,
                 epoch_train = 30,
                 split_validation=0,
                 min_count=5,   # Vectors will not be created for tokens bellow this number
                 window=3,
                 size=300,
                 sample=6e-5,
                 alpha=0.03,
                 min_alpha=0.0007,
                 negative=20,
                 ):

        logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt='%H:%M:%S', level=log_level)

        self._current_path = os.path.dirname(__file__)

        # Validate input provided by user
        self._validate_input(
                        use_type,
                        train_path,
                        output_path,
                        validation_path)

        self._word_vec_dict = {}

        self._data_path = train_path
        self._batch_size = batch_size
        self._similarity_threshold = sim_threshold
        self._jeccard_threshold = jec_threshold
        self._distance_type = distance_type
        self._outputh_path = output_path
        self._validation_path = validation_path
        self._existing_model = None
        self._epochs = epoch_train
        self.split_validation = 0
        self._min_count = min_count
        self._window = window
        self._size = size
        self._sample = sample
        self._alpha = alpha
        self._min_alpha = min_alpha
        self._negative = negative

        if use_type == 'train':
            # The output of YAP comes as JSON file, in a different format than the general tweets
            # So we choose a different iterator, but the output of the iterator will be the same for all types
            if input_type == 'yap':
                self._type = PARSE_TYPE_YAP
                self._text_key = JSON_SINGLE_TOKEN_STRING
                self._sentenceses_iter = Sentences_Yap_CSV(self._data_path, split_validation)
            else:
                self._type = PARSE_TYPE_BLANK
                self._text_key = PARSE_TYPE_BLANK_TEXT
                self._sentenceses_iter = Sentences(self._data_path ,split_validation)

        if use_type == 'similarity':
            self._sentenceses_iter = Sentences_csv(self._data_path)

        self._cores = multiprocessing.cpu_count()  # Count the number of cores in a computer

    def _validate_input(self,
                        use_type,
                        train_path,
                        output_path,
                        validation_path
                        ):

        if use_type == 'train' or train_path != None:
            # Input None
            if not train_path:
                raise RuntimeError("The path to input dataset was not provided")
            # Bad path

            if use_type != 'train' and not os.path.exists(train_path):
                raise RuntimeError(
                    "The path to train dataset {} does'nt exist, please review it and try again".format(train_path))
            if not output_path:
                raise RuntimeError("The path to model output was not provided")

            try:
                # In case the output path doesn't exist we will create an output folder in current folder
                if not os.path.exists(output_path):
                    self._outputh_path = os.path.join(os.path.join(self._current_path, OUTPUT_FOLDER_NAME))
                    raise RuntimeWarning(
                        "The output path provided by user {} does'nt exist, and will be created in {}".
                            format(output_path, self._outputh_path ))
            except:
                if not os.path.exists(self._outputh_path):
                    os.makedirs(self._outputh_path)

    @staticmethod
    def get_jaccard_sim(sentence_1, sentence_2):
        '''
        Find the number of similar words between two sentences
        '''
        a = set(sentence_1.split())
        b = set(sentence_2.split())
        c = a.intersection(b)
        if (len(a) + len(b) - len(c)) == 0:
            return 0
        return float(len(c)) / (len(a) + len(b) - len(c))

    def calculate_sentence_similarity(self, sentence_1, sentence_2, similarity_threshold=0.85):
        '''
        Calculate the distance between two sentences vectors
        '''

        # Using world movers similarity function
        if self._distance_type == 'wm':
            return 1 - self._model.wmdistance(sentence_1, sentence_2)

        # Cosine distance
        sentence_1_vec, unknown_1 = self._get_setnence_vector(sentence_1)
        sentence_2_vec, unknown_2 = self._get_setnence_vector(sentence_2)

        if unknown_1:
            print('For sentence {} these tokens were unknown by the model {}'.format(sentence_1, unknown_1))

        if unknown_2:
            print('For sentence {} these tokens were unknown by the model {}'.format(sentence_2, unknown_2))

        # If [0] was returned, it means didn't find any vector
        if np.all(sentence_1_vec == [0]) or np.all(sentence_2_vec == [0]):
            return 0

        result = 1 - spatial.distance.cosine(sentence_1_vec, sentence_2_vec)

        result = 0 if result < 0 else result # Setting results bellow 0 to 0

        return result

    def _get_setnence_vector(self, sentence:str):
        '''
        Calculate sentence vector by averaging over every token vector
        :param sentence: String representing the sentence we want to calculate vector for
        :return: Sentence vector, and list of unknown word
        '''

        # Clean punctuation from the string
        sentence = sentence.translate(
            sentence.maketrans("", "", ''.join(char for char in string.punctuation if char != '"')))
        sentence = sentence.split(" ")

        sentence_avg = 0
        unknown_list = [] # Unknown tokens for the model
        sentence_exist = []  # Collect the tokens that are known to the model
        for word in sentence:

            if word in self._model:
                # Vectors are normalized to unit length before they are used for similarity calculation
                # For word embedding the actual length of the vector is irrelevant
                norm = np.linalg.norm(self._model[word])
                if norm == 0:
                    sentence_avg += self._model[word]
                else:
                    sentence_avg += self._model[word] / norm
                sentence_exist.append(word)
            else:
                unknown_list.append(word)

        # If there is a case where all words of one the sentences is unknown we return 0 for similarity
        if len(sentence_exist) - len(unknown_list) > 0:
            # Average only over the known words
            sentence_vec = np.divide(sentence_avg, len(sentence_exist) - len(unknown_list))
        else:
            return np.array([0]), unknown_list
            print("Sentence 1 Unknown word {}".format(word))
        return sentence_vec, unknown_list

    def find_similar_in_all_original_tweets(self, pairs_to_stop = -1, start_point=1):
        if self._type == PARSE_TYPE_YAP:
            raise RuntimeError("Json find_similar_in_all is not supported")
        else:
            tweets = pd.read_msgpack(self._data_path, encoding="utf8")

        tweets.set_index(self._ID_NAME)
        # Create a list from all the tweets, and devide it to sublist of size batch
        list_df = [tweets[i:i + self._batch_size] for i in range(start_point, tweets.shape[0], self._batch_size)]
        sim_res = []
        id = -1
        for batch in list_df[1:]:

            tweets_comb = list(combinations(batch['clean_text'],2))

            sim_res += [(batch[batch['clean_text']==c[0]]['id'].iloc[0],
                c[0],
                batch[batch['clean_text']==c[1]]['id'].iloc[0],
                c[1],
                self.calculate_sentence_similarity(self.model, sentence_1=c[0], sentence_2=c[1]))
                        for c in tweets_comb
                        if batch[batch['clean_text']==c[0]]['name'].iloc[0] != batch[batch['clean_text']==c[1]]['name'].iloc[0]
                        and self.get_jaccard_sim(sentence_1=c[0],sentence_2=c[1]) < self._jeccard_threshold
                        and self.calculate_sentence_similarity(self.model, sentence_1=c[0], sentence_2=c[1]) >= self._similarity_threshold]

            id += len(sim_res)
            print('Pairs found until now: {}'.format(id))
            if id >= pairs_to_stop:
                if not os.path.exists(self._outputh_path):
                    os.makedirs(self._outputh_path)
                print('Final found {} pairs'.format(len(sim_res)))
                pd.DataFrame(sim_res).to_csv(self._outputh_path + '/pairs_similar_old_w2v.csv')


    def find_similar_in_all_sentences(self, pd_sentences, use_jeccard_threshold=False):
        '''
        Going over all sentences and finding similarity between them
        :param pd_sentences: data frame of sentences
        :return:
        '''

        bar = progressbar.ProgressBar()

        tweets_comb = list(combinations(pd_sentences.values, 2))
        bar(range(len(tweets_comb)))

        # We can refine our similarity search by using these two threshold
        # Jeccard in order to get sentences with less shared words
        jeccard_threshold = 1
        if use_jeccard_threshold:
            jeccard_threshold = self._jeccard_threshold

        similarity_threshold = self._similarity_threshold

        sim_res = []
        for (sentence_1_l, sentence_2_l) in tweets_comb:

            bar.next()
            sentence_1 = sentence_1_l[0]
            sentence_2 = sentence_2_l[0]

            if sentence_1 != sentence_2:
                if self.get_jaccard_sim(sentence_1=sentence_1, sentence_2=sentence_2) < jeccard_threshold:

                    similarity = self.calculate_sentence_similarity(sentence_1=sentence_1, sentence_2=sentence_2)
                    if similarity >= similarity_threshold:
                        sim_res.append((sentence_1, sentence_2, similarity))

        return sim_res

    def _calculate_sentence_similarity_from_file(self,
                                                 comparison_type='file_pairs',
                                                 sentence_to_compare = None,
                                                 sort_by_similarity=False):
        '''
        Running sentences similarity from csv file

        There are three options:

        file_pairs: will compare every two sentences in a chronological order
        file_all: will compare all permutations of all sentences in the file
        file_best_match: will output the most two similar sentences in the file


        :return: path to output file
        '''

        output_file = type(self._model).__name__ + '_'+comparison_type+'_out_' + \
                      (datetime.datetime.now().strftime('%H-%M-%S_%m-%d-%Y')) + '.csv'
        output_path = os.path.join(self._outputh_path ,output_file)
        columns = ['sentence_1', 'sentence_1', 'similarity']

        if not os.path.exists(self._outputh_path):
            os.makedirs(self._outputh_path)

        output = []
        # Compare every two sentences in chronological order
        if comparison_type == 'file_pairs':

            # Collect all sentences and their similarity score
            output = []

            for i, sentence in enumerate(self._sentenceses_iter):
                if i % 2 == 0:
                    sentence_1 = ' '.join(sentence)
                else:
                    sentence_2 = ' '.join(sentence)
                    output.append((sentence_1, sentence_2, self.calculate_sentence_similarity(sentence_1, sentence_2)))

        # Compare all with all
        if comparison_type == 'file_all':
            all_cleaned_Sentences = []

            # Collect all cleand sentences from iterator

            for sentence_1 in self._sentenceses_iter:
                all_cleaned_Sentences += [' '.join(sentence_1)]

            output = self.find_similar_in_all_sentences(pd.DataFrame(all_cleaned_Sentences))

        # Find 2 most similar
        if comparison_type == 'file_best_match':


            if not sentence_to_compare:
                raise RuntimeError('For option file_best_match pleas provide a sentence to compare to' )
            max_sim = 0
            best_match = ''
            for sentence in self._sentenceses_iter:
                sentence_str= ' '.join(sentence)
                output.append((sentence_to_compare, sentence_str, self.calculate_sentence_similarity(sentence_to_compare, sentence_str)))

        # Sort by similarity
        if sort_by_similarity:
            output.sort(key=lambda x: x[2], reverse=True)

        pd.DataFrame(output, columns=columns).to_csv(output_path)

    def _create_sentences_txt_file(self, file_name='sentences.txt'):
        '''
        Creates out of the senteces files a simple text file where every line is a new sentence
        Mostly used for training sent2Vec model

        Will save the file in the output path
        '''

        # Will overwrite the file if exists
        file_path = os.path.join(self._outputh_path, file_name)
        with open(file_path, 'w+', encoding="utf8") as f:
            for sentence in self._sentenceses_iter:
                f.write(' '.join(sentence)+'\n')


    def _build_vocab(self):
        '''
        Creates the model dictionary from all the words in the train data
        '''

        t = time()
        self.model.build_vocab(self._sentenceses_iter, progress_per=10000)
        print(self.model)
        print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

    def _train(self, save_model=True):
        '''
        Train words embedding model

        :param save_model: IF we want to save the model to storage device
        :return:
        '''
        t = time()

        self.model.train(self._sentenceses_iter, total_examples=self.model.corpus_count, epochs=self._epochs, report_delay=1)

        print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

        if not os.path.exists(self._outputh_path):
            os.makedirs(self._outputh_path)

        if save_model:
            self.model.save(os.path.join(self._outputh_path, 'epochs_'+str(self._epochs)+'__' +strftime("%Y%m%d-%H%M%S") + '.bin'))

    def validate(self, prediction_path, perfect_path, classes=2, plot_results=True):
        '''
        Validate the model by receiving the predicitons and the ground truth and the number of classes to be compared
        to, that will dived the predicitons to number of classes groups using k-mens and plot the results if user
        selected to
        '''

        assert(prediction_path)
        assert(perfect_path)

        if not os.path.exists(prediction_path):
            raise RuntimeError('The path {} for prediction dosn\'t exist'.format(prediction_path))

        if not os.path.exists(perfect_path):
            raise RuntimeError('The path {} for perfect dosn\'t exist'.format(perfect_path))

        if perfect_path.split('.')[-1] != 'csv':
            raise RuntimeError('we only support CSV files for validation, and not {}'.format(perfect_path.split('.')[-1]))

        if prediction_path.split('.')[-1] != 'csv':
            raise RuntimeError('we only support CSV files for validation, and not {}'.format(prediction_path.split('.')[-1]))

        pd_pred = pd.read_csv(prediction_path, index_col=False)

        pd_perf = pd.read_csv(perfect_path, index_col=False)

        if len(pd_perf.values) != len(pd_pred.values):
            raise RuntimeError('The number of prediction {} and perfects {} is not equal'.
                               format(len(pd_pred.values), len(pd_perf.values)))
        # Binary classification
        if classes == 2:
            kmeans = KMeans(n_clusters=classes, init=np.array([[1],[0]]), n_init=1)
            predictions = [1-val for val in kmeans.fit_predict(np.reshape(pd_pred['similarity'].values,
                                                                          (pd_pred['similarity'].values.shape[0],1)))]
        elif classes == 4:
                # Selecting the start centroids as if the range is divided equally to 4
                kmeans = KMeans(n_clusters=classes, init=np.array([[0.87], [0.62], [0.38], [0]]), n_init=1)
                predictions = [3 - val for val in kmeans.fit_predict(np.reshape(pd_pred['similarity'].values,
                                                     (pd_pred['similarity'].values.shape[0], 1)))]
        else:
            raise RuntimeError('The number of classes {} is not supported only 2 or 4 are supported'.
                               format(classes))
#
        #Split the predicitons by perfects class
        predictions_by_class_dict = {}
        for prediciton, class_type in zip(pd_pred['similarity'].values, pd_perf['similarity'].values):
            if class_type not in predictions_by_class_dict:
                predictions_by_class_dict[class_type] = [prediciton]
            else:
                predictions_by_class_dict[class_type].append(prediciton)

        # Find the threshold that was selected
        threshold = [0] * (classes - 1)
        for i, c_type in enumerate(predictions):
            if int(c_type) < len(threshold):
                threshold[int(c_type)] = max(threshold[c_type], pd_pred['similarity'].values[i])

        if classes == 2:
            print("Threshold that was selected is {} above it sentences considered similar\n\n".format(threshold))
        if classes == 4:
            print("Thresholds that were selected are {}\n\n".format(threshold))

        print(classification_report(pd_perf['similarity'].values, predictions))
        print(confusion_matrix(pd_perf['similarity'].values, predictions))
        confusion_m = confusion_matrix(pd_perf['similarity'].values, predictions)

        if plot_results:
            self.plot_results(predictions_by_class_dict, threshold, confusion_m, classes)

    def plot_results(self, predictions_by_class_dict, threshold, confusion_m, classes):
        fig = plt.figure()
        ax = fig.add_subplot()
        colors = ("green", "blue", 'yellow', "pink")
        data = ()
        color = ()
        for i in reversed(range(classes)):
            data += ((np.array(range(len(predictions_by_class_dict[i]))), np.array(predictions_by_class_dict[i])),)
            color += (colors[i],)

        for data, color in zip(data, color):
            x, y = data
            ax.scatter(x, y, alpha=0.6, c=color, edgecolors='none', s=50)
        x1, x2, y1, y2 = plt.axis()

        for i in range(0, classes-1, 1):
            l = mlines.Line2D([x1, x2], [threshold[i], threshold[i]],color='red', alpha=0.5)
            ax.add_line(l)
            plt.yticks(np.append(plt.yticks()[0], round(threshold[i], 3)),np.append(plt.yticks()[1], str(round(threshold[i], 3))))


        plt.yticks(np.append(np.append(plt.yticks()[0], 0),1), np.append(np.append(plt.yticks()[1], str(0)),str(1)))
        plt.show()

        plt.figure(figsize=(3, 3))
        sn.heatmap(confusion_m, annot=False ,cmap='Blues', fmt='g')
        plt.show()

    def merge_datasets_to_one_dataframe(self,
                                     train_path,
                                     validatin_path,
                                     test_path,
                                     dimensions,
                                     test_separator = ',',
                                     validation_separator = ',',
                                     train_separator = '\n'):
        '''
        Draw the distribution of train/validation and test sets
        :return:
        '''

        pd_validation = pd.read_csv(validatin_path, sep=validation_separator, encoding="utf8")
        pd_train = pd.read_csv(train_path,encoding="utf8" , sep=train_separator, header=None)
        pd_test = pd.read_csv(test_path, encoding="utf8", sep=test_separator)


        # Create vectors out of test/train and validation
        train_vectors = [np.array(self._get_setnence_vector(str(sentence))[0]) for sentence in pd_train.sample(10000).values ]
        train_vectors = [np.reshape(vector,(dimensions,1)) for vector in train_vectors if vector.size == dimensions]

        test_vectors = [self._get_setnence_vector(str(sentence))[0] for sentence in pd_test.values]
        test_vectors = [np.reshape(vector, (dimensions, 1)) for vector in test_vectors if vector.size == dimensions]

        validation_vectors = [self._get_setnence_vector(str(sentence))[0] for sentence in pd_validation.values]
        validation_vectors = [np.reshape(vector, (dimensions, 1)) for vector in validation_vectors if vector.size == dimensions]

        # Create equal number of columns to vectors dimensions
        feat_cols = ['dimension_' + str(i) for i in range(len(train_vectors[0]))]

        # Create the data-frames for all datasets
        pd_train_vectors = pd.DataFrame.from_records(train_vectors, columns=feat_cols)
        pd_test_vectors = pd.DataFrame.from_records(test_vectors, columns=feat_cols)
        pd_validation_vectors = pd.DataFrame.from_records(validation_vectors, columns=feat_cols)

        pd_labels = pd.DataFrame(columns=feat_cols+['label'])

        pd_train_set = pd.concat([pd_train_vectors, pd_labels], ignore_index=True, sort=False)
        pd_train_set.loc[:, 'label'] = '0'


        pd_test_set = pd.concat([pd_test_vectors, pd_labels], ignore_index=True, sort=False)
        pd_test_set.loc[:, 'label'] = '1'

        pd_validation_set = pd.concat([pd_validation_vectors, pd_labels], ignore_index=True, sort=False)
        pd_validation_set.loc[:, 'label'] = '2'

        self._pd_final_res = pd.concat([pd_train_set, pd_test_set, pd_validation_set], ignore_index=True, sort=False)

    def draw_sentences_distribution(self, use_pca=True):
        '''
        Will plot the dataset in a reduce dimensions
        :return:
        '''
        import pandas as pd
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        from sklearn.cluster import KMeans

        np.random.seed(42)

        # Lower to 80 dimensions using PCA, since t-SNE is not working well with very high dimensions
        pca_80 = PCA(n_components=80)
        pca_result_80 = pca_80.fit_transform(self._pd_final_res[self._pd_final_res.columns[0:300]].values)

        print('Cumulative explained variation for 50 principal components: {}'.format(
            np.sum(pca_80.explained_variance_ratio_)))

        self._pd_final_res['tsne-2d-one'] = pca_result_80[:, 0]
        self._pd_final_res['tsne-2d-two'] = pca_result_80[:, 1]



        time_start = time()
        tsne = TSNE(n_components=2, verbose=2, perplexity=50, learning_rate=300, n_iter=500)
        if use_pca:
            tsne_results = tsne.fit_transform(pca_result_80)
        else:
            tsne_results = tsne.fit_transform(self._pd_final_res[self._pd_final_res.columns[2:]].values)

        print('t-SNE done! Time elapsed: {} seconds'.format(time() - time_start))

        self._pd_final_res['tsne-2d-one'] = tsne_results[:, 0]
        self._pd_final_res['tsne-2d-two'] = tsne_results[:, 1]


        fig, ax = plt.subplots()
        color_dict = {'0': 'red', '1': 'blue', '2': 'green'}
        color = [color_dict[label] for label in  self._pd_final_res['label']]


        ax.scatter(
            x=self._pd_final_res['tsne-2d-one'], y=self._pd_final_res['tsne-2d-two'],
            c=color,
            s=1,
            alpha=0.3
        )
        if not os.path.exists(self._outputh_path):
            os.makedirs(self._outputh_path)
        plt.savefig(os.path.join(self._outputh_path,'tsne.png'))
        print('Saved {}'.format(self._outputh_path))

        exit()


