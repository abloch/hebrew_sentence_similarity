import random
import os
import pandas as pd
from general_config import *

class text_utills:
    def __init__(self,
                 path_to_open_folder,
                 path_to_save_folder=None):

        assert(path_to_open_folder)
        self._path_to_open_file = path_to_open_folder

        self._path_to_save_folder = path_to_save_folder
        # User didn't provide the path to save, so we will save at the current project directory
        if not self._path_to_save_folder:
            self._path_to_save = os.path.dirname(__file__)



    def open_file_as_dataframe(self, file_to_open):
        '''
        Open a file and return the content in dataframe
        :return: the data frame of the file
        '''

        if not file_to_open:
            raise RuntimeError("No file was passed")

        file_type = file_to_open.split('.')[-1]

        if file_type == 'json':
            file_path = os.path.join(self._path_to_open_file, file_to_open)
            if not os.path.exists(file_path):
                raise RuntimeError("The file path {} dosn't exist".format(file_path))

            pd_f_content = pd.read_json(file_path, encoding="utf8")
        else:
            raise RuntimeError("Files of type {} are not supported".format(file_type))

        return pd_f_content

    def clean_content_file(self, pd_content, labels = ['id_1', 'id_2', 'tweet_1', 'tweet_2', 'match']):
        '''
        Clean file of all unnecessary data
        '''

        return pd.DataFrame.from_records([(tweet_1['id'], tweet_2['id'], tweet_1['tweet'], tweet_2['tweet'] ,0)
        for tweet_1, tweet_2 in zip(pd_content['source'], pd_content['matches'])],columns=labels)

    def convert_validation_json_to_csv(self, file_name):
            data = self.open_file_as_dataframe(file_name)

            for sentence in data['clean_text'][:end_test]:
                # clean Arabic tweets
                #if len([word for word in sentence if detect_language(word) == 'Arabic']) > 0:
                #    continue
                sentence_list = self._clean_sentence(sentence).split(" ")
                final_sentence_list = []
                for token in sentence_list:
                    # Check that this token is not mixed with characters and numbers from different languages and sigits
                    if not self.is_mixed_language(token):
                        final_sentence_list.append(token)


    def split_translation(self, file_name):
        with open(os.path.join(self._path_to_open_file, file_name), 'r', encoding="utf8") as f:
            data = f.read().replace('\n', ' ')
            data = data.replace(',', ' ')
            data = data.replace('.', '\n')
            data = data.replace('?', '\n')
            data = data.replace('!', '\n')
            text_file = open(os.path.join(self._path_to_save_folder, file_name+"_output.txt"), "w", encoding="utf8")
            text_file.write(data)

    def create_merged_csv_from_translation(self, file_name_1, file_name_2, out_name, add_random=False):
        '''
        Merge two text files into one csv file every even number will be sentence from file 1 and odd from file 2
        :param file_name_1:
        :param file_name_2:
        :param out_name:
        :param add_random:
        :return:
        '''
        data_df_1 = pd.read_csv(os.path.join(self._path_to_open_file, file_name_1), sep="\n", header=None)
        data_df_2 = pd.read_csv(os.path.join(self._path_to_open_file, file_name_2), sep="\n", header=None)
        sentence_list= []
        for (sentence_1, sentence_2) in zip(data_df_1.values, data_df_2.values):
            sentence_list.append(sentence_1[0])
            sentence_list.append(sentence_2[0])
        if add_random:
            i=0
            neg_example = []
            random.seed(42)
            # Don't want the same sentence or the similar to it
            while i < len(sentence_list) // 2:
                sen = random.sample(range(len(sentence_list)), 2)
                if abs(sen[0]-sen[1]) > 1:
                    neg_example.append(sentence_list[sen[0]]) # Read two random from positive
                    neg_example.append(sentence_list[sen[1]])
                    i+=1

            sentence_list += neg_example

        data_df = pd.DataFrame(sentence_list)
        data_df.to_csv(os.path.join(self._path_to_save_folder, out_name), index=False, header=False, encoding="utf8")


    def create_validation_set_from_translation(self, file_name_1, file_name_2, out_name, add_random=False):

        data_df_1 = pd.read_csv(os.path.join(self._path_to_open_file, file_name_1), sep="\n", header=None)
        data_df_2 = pd.read_csv(os.path.join(self._path_to_open_file, file_name_2), sep="\n", header=None)
        sentence_list= []
        tup_sentence_list = []
        for (sentence_1, sentence_2) in zip(data_df_1.values, data_df_2.values):
            sentence_list.append(sentence_1[0])
            sentence_list.append(sentence_2[0])
            tup_sentence_list.append((sentence_1[0], sentence_2[0], 1))
        if add_random:
            i=0
            random.seed(42)  # so we will have same random as create_merged_csv_from_translation
            neg_example = []
            # Don't want the same sentence or the similar to it
            while i < len(sentence_list) // 2:
                sen = random.sample(range(len(sentence_list)), 2)
                if abs(sen[0]-sen[1]) > 1:
                    neg_example.append((sentence_list[sen[0]], sentence_list[sen[1]] , 0))
                    i+=1

            tup_sentence_list += neg_example

        data_df = pd.DataFrame(tup_sentence_list, columns = ['sentence_1', 'sentence_1', 'similarity'])
        data_df.to_csv(os.path.join(self._path_to_save_folder, out_name), index=False, encoding="utf8")

