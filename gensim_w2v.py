from gensim.models import Word2Vec
import multiprocessing
import ast
from general_config import SentenceSim
import logging  # Setting up the loggings to monitor gensim



class sentence_sim_w2v(SentenceSim):

    _ID_NAME = 'id'
    def __init__(self,
                 use_type,
                 alpha=None,
                 input_type=None,
                 train_path=None,
                 output_path=None,
                 logging=logging.ERROR,
                 distance_type=None,
                 pretrained_model_path=None,
                 workers = -1,
                 batch_size = 500,
                 sim_threshold = 0.9,
                 jec_threshold = 0.5,
                 epoch_train = 30,
                 min_count=10,
                 size=None,
                 sample=None,
                 window=None,
                 split_validation=0
                 ):

        super().__init__(
            log_level=logging,
            use_type=use_type,
            alpha=alpha,
            window=window,
            size=size,
            distance_type=distance_type,
            sample=sample,
            input_type=input_type,
            train_path=train_path,
            output_path=output_path,
            batch_size=batch_size,
            sim_threshold=sim_threshold,
            jec_threshold=jec_threshold,
            epoch_train=epoch_train,
            min_count=min_count,
            split_validation=split_validation)

        if use_type != 'validation':
            if pretrained_model_path:
                   self._model = Word2Vec.load(pretrained_model_path)
            else:
                if workers == -1:
                    workers = self._cores -1
                self.model = Word2Vec(min_count=self._min_count,
                                      window=self._window,
                                      size=self._size,
                                      sample=self._sample,
                                      alpha=self._alpha,
                                      min_alpha=self._min_alpha,
                                      negative=self._negative,
                                      workers=workers)

        self._validation_th_wm = 0.65
        if input_type == 'yap':
            self._validation_th_cos = 0.73
        else:
            self._validation_th_cos = 0.5





    def open_bag_of_words_json(json_path):
        '''
        Open the bag of words dictionary in a given path
        :param json_path:
        :return:
        '''
        with open(json_path, encoding="utf8") as json_file:
            s = json_file.read()
            print(ast.literal_eval(s))


    def train(self):

        self._build_vocab()
        self._train()

