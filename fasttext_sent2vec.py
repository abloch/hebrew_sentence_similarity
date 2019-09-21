import subprocess
import pyximport
import sent2vec
from scipy import spatial
pyximport.install()

from general_config import SentenceSim
import logging


class Sentence2VectorSim(SentenceSim):

    def __init__(self,
                 use_type,
                 window=None,
                 size=None,
                 sample=None,
                 input_type=None,
                 alpha=None,
                 train_path=None,
                 output_path=None,
                 pretrained_model_path=None,
                 validation_path=None,
                 logging=logging.ERROR,
                 distance_type=None,
                 batch_size=500,
                 sim_threshold=0.9,
                 jec_threshold=0.5,
                 epoch_train=30,
                 workers = -1,
                 min_count=10,
                 split_validation=0
                 ):

        super().__init__(
            use_type=use_type,
            train_path=train_path,
            output_path=output_path,
            validation_path=validation_path,
            log_level=logging,
            batch_size=batch_size,
            input_type=input_type,
            alpha=alpha,
            sample=sample,
            size=size,
            distance_type=distance_type,
            window=window,
            sim_threshold=sim_threshold,
            jec_threshold=jec_threshold,
            epoch_train=epoch_train,
            min_count=min_count,
            split_validation=split_validation)

        if use_type != 'validation':
            if pretrained_model_path:
                self._model = sent2vec.Sent2vecModel()
                self._model.load_model(pretrained_model_path)
            else:
                subprocess.call(['/home/local/Tasks/nlp/sent2vec/sent2vec/fasttext'])
                if workers == -1:
                    workers = self._cores -1
                self.model = FastText(min_count=self._min_count,
                                      window=self._window,
                                      size=self._size,
                                      sample=self._sample,
                                      alpha=self._alpha,
                                      min_alpha=self._min_alpha,
                                      negative=self._negative,
                                      sg=0,  # CBOW
                                      min_n=2,  # minimum ngram
                                      workers=workers)

        self._validation_th_wm = 0.65
        self._validation_th_cos = 0.25


    def train(self):

        self._build_vocab()
        self._train()


    def calculate_sentence_similarity(self, sentence_1, sentence_2, similarity_threshold=0.85):
        '''
        Calculate the distance between two sentences vectors
        '''

        embs = self._model.embed_sentences([sentence_1, sentence_2])
        result = (1 - spatial.distance.cosine(embs[0], embs[1]))

        result = 0 if result < 0 else result
        return result
