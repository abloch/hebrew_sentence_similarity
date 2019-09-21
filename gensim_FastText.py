from gensim.models import FastText
from general_config import SentenceSim
import logging


# https://subscription.packtpub.com/book/big_data_and_business_intelligence/9781789130997/app01/app01lvl1sec40/gensim-fasttext-parameters
class FastTextSentenceSim(SentenceSim):

    def __init__(self,
                 use_type,
                 window=None,
                 size=None,
                 sample=None,
                 input_type=None,
                 alpha=None,
                 distance_type=None,
                 train_path=None,
                 output_path=None,
                 pretrained_model_path=None,
                 validation_path=None,
                 logging=logging.ERROR,
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
            distance_type=distance_type,
            batch_size=batch_size,
            input_type=input_type,
            alpha=alpha,
            sample=sample,
            size=size,
            window=window,
            sim_threshold=sim_threshold,
            jec_threshold=jec_threshold,
            epoch_train=epoch_train,
            min_count=min_count,
            split_validation=split_validation)

        if use_type != 'validation':
            if pretrained_model_path:
                self._model = FastText.load(pretrained_model_path)
                # World movers
                if distance_type == 'wm':
                    self._model.init_sims(replace=True)
            else:
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




    def train(self):

        self._build_vocab()
        self._train()


