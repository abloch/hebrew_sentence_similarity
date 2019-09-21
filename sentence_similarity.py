import logging  # Setting up the loggings to monitor gensim
import argparse
from gensim_w2v import sentence_sim_w2v
from fasttext_sent2vec import Sentence2VectorSim
from gensim_FastText import FastTextSentenceSim


class_initiation_dict = {'w2v':sentence_sim_w2v, 'fasttext':FastTextSentenceSim, 'sent2vec':Sentence2VectorSim}

verbossity_dict = {'0':logging.ERROR, '1':logging.DEBUG}
file_sim_choises = ['file_pairs', 'file_all', 'file_best_match','input']

def parse():

    def size_type(x):
        x = int(x)
        if x < 50 or x > 300:
            raise argparse.ArgumentTypeError("The size argument must be between 50 and 300 and not {}".format(x))
        return x

    def minimum_type(x):
        x = int(x)
        if x < 2 or x > 100:
            raise argparse.ArgumentTypeError("The minimum argument must be between 2 and 100 and not {}".format(x))
        return x

    def sample_type(x):
        x = float(x)
        if x < 0 or x > 1e-5:
            raise argparse.ArgumentTypeError("The sample argument must be between 0 and 1e-5 and not {}".format(x))
        return x

    def lra_type(x):
        x = float(x)
        if x < 0.01 or x > 0.05:
            raise argparse.ArgumentTypeError("The alpha argument must be between 0.01 and 0.05 and not {}".format(x))
        return x


    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(help='Choose a command Train new model or Find similarity on existing model')

    similarity_parser = subparsers.add_parser('similarity', help='"Run a predictions of given input and selected model"')
    similarity_parser.add_argument("-st", "--similarity_type", choices=['file_pairs', 'file_all', 'file_best_match','input'],
                                   type=str,
                                   help='Choose the type of similarity comparisson we want to perform, from file or by user input'
                                   'Sentences in the file should be devided by new-line\n'
                                   'file has three options: \n file_pairs: will compare every two '
                                   'sentences in a chronological order.\n'
                                   'file_all: will compare all permutations of all sentences in the file\n'
                                   'file_best_match: For a given sentence as input and a file will output all similarity results'
                                        'ranked by most to less similar',required=True,
                                   default='file')
    similarity_parser.add_argument("-dt", "--distance_type", choices=['wm', 'cos'], help='The distance funtion we want to run',
                                                                        required=False, default='cos')
    similarity_parser.add_argument("-model", "--model_type", type=str, help='The model we want to run',
                              choices=['w2v', 'w2v_numpy','fasttext', 'sent2vec'], default='wv2', required=False)
    similarity_parser.add_argument("-if", "--input_file", type=str, help='Path to file with all sentences we want to compare',
                                   default=None, required=False)
    similarity_parser.add_argument("-im1", "--sentence1", type=str,
                                   help='The first sentences we want to compare', required=False)
    similarity_parser.add_argument("-im2", "--sentence2", type=str,
                                   help='The second sentences we want to compare', required=False)
    similarity_parser.add_argument("-mp", "--model_path", type=str,
                                   help='Path to model we want to compare by, if not provided a default pretrained '
                                        'w2v model will be selected', required=False)
    similarity_parser.add_argument("-op", "--output_path", type=str, help='The output for the similarity results. Results will also be sent to stdout',
                              required=False, default=None)
    similarity_parser.add_argument("-v", "--verbosity", type=str,
                                   help='Logging verbosity',
                                   required=False, default='0')
    similarity_parser.add_argument("-tr", "--threshold", type=float,
                                   help='Look at sentence above threshold',
                                   required=False, default=0)
    similarity_parser.add_argument("-sr", "--sort_result", type=float,
                                   help='Sort by similarity the output result',
                                   required=False, default=False)


    train_parser = subparsers.add_parser('train', help='"Train a new model" help')

    train_parser.add_argument("-tp", "--train_path", type=str, help='Train dataset full path', required=True,
                        default=None)

    train_parser.add_argument("-it", "--input_type", type=str, help='Input for training comes in YAP json format or'
                                                                    ' general tweets format', required=True,
                              choices=['yap', 'tweets'],
                              )

    train_parser.add_argument("-model", "--model_type", type=str, help='The model we want to run',
                              choices=['w2v','fasttext', 'sent2vec'], default='wv2', required=False)


    train_parser.add_argument("-op", "--output_path", type=str, help='The output model', required=True, default=None)
    train_parser.add_argument("-b", "--batch_size", type=int, help='The batch size for training', required=False,
                              default=500)
    train_parser.add_argument("-e", "--epochs_number", type=int, help='The number of epochs for training', required=False,
                              default=100)
    train_parser.add_argument("-mc", "--minimum_count", type=minimum_type, help='Ignores all words with total absolute '
                                                                       'frequency lower than this',
                              required=False, default=15)
    train_parser.add_argument("-w", "--window", type=int, help='The n-grams window for FastText and Sent2Vec and '
                                                               'prediction for word2vec',
                              required=False, default=3)

    train_parser.add_argument("-sz", "--size", type=size_type, help=' Dimensionality of the feature vectors (50, 300)',
                              required=False, default=300)

    train_parser.add_argument("-sm", "--sample", type=sample_type, help='The threshold for configuring which'
                                                                    'higher-frequency words are randomly downsampled',
                              required=False, default=6e-5)

    train_parser.add_argument("-lr", "--alpha", type=lra_type, help='The initial learning rate',
                              required=False, default=0.03)

    train_parser.add_argument("-wr", "--workers", type=int, help='The number of workers, -1 for using all cores',
                              required=False, default=-1)
    train_parser.add_argument("-v", "--verbosity", type=str, help='Logging verbosity', required=False, default='0')


    validate_parser = subparsers.add_parser('validation', help='Validate results using predicted and prefect csv files')
    validate_parser.add_argument("-it", "--input_type", type=str, help='Input for training comes in YAP json format or'
                                                                    ' general tweets format', required=False,
                              choices=['yap', 'tweets'], default='tweets'
                              )

    validate_parser.add_argument("-pp", "--predicted_path", type=str, help='Predicted csv full path', required=True,
                        default=None)
    validate_parser.add_argument("-pf", "--perfects_path", type=str, help='Perfects csv full path', required=True,
                        default=None)
    validate_parser.add_argument("-nc", "--number_classes", type=int, help='Number of classes to devide to', required=False,
                        default=2)

    validate_parser.add_argument("-model", "--model_type", type=str, help='The model we want to validate, every model '
                                                'has different threshold for similar/not-similar',
                              choices=['w2v', 'w2v_numpy','fasttext', 'sent2vec'], default='wv2', required=False)

    validate_parser.add_argument("-vs", "--visualize", type=bool, help='Visualize the output as a confusion and scatter plot',
                              required=False, default=True)

    distribution_parser = subparsers.add_parser('visualize', help='Visualize the distribution of tran/validation and '
                                                                  'test dataset')

    distribution_parser.add_argument("-td", "--train_dataset", type=str, help='Train dataset full path', required=True,
                        default=None)
    distribution_parser.add_argument("-vd", "--validation_dataset", type=str, help='Validation dataset full path', required=True,
                        default=None)
    distribution_parser.add_argument("-tsd", "--test_dataset", type=str, help='Test dataset full path', required=True,
                        default=None)

    distribution_parser.add_argument("-ts", "--train_separator", type=str, help='Train dataset separator', required=False,
                        default='\n')

    distribution_parser.add_argument("-vs", "--validation_separator", type=str, help='Validation dataset separator', required=False,
                        default=',')
    distribution_parser.add_argument("-tsp", "--test_separator", type=str, help='Test dataset separator', required=False,
                        default=',')

    distribution_parser.add_argument("-vdm", "--vector_dimensions", type=int, help='Dimensions of the vector '
                                        'in the given model', required=False, default=300)

    distribution_parser.add_argument("-model", "--model_type", type=str, help='The model we want to run',
                              choices=['w2v', 'w2v_numpy','fasttext', 'sent2vec'], default='wv2', required=False)

    distribution_parser.add_argument("-mp", "--model_path", type=str,
                                   help='Path to model we want to use', required=False)

    distribution_parser.add_argument("-op", "--output_path", type=str, help='The output for the visualization results.',
                              required=True, default=None)

    args = parser.parse_args()

    # Is similarity was chosen
    if 'similarity_type' in args:

        if args.similarity_type == 'file' and args.input_file == None:
            parser.error('--similarity_type was selected from a file, please provide a file path')

        if args.similarity_type == 'file' and ('sentence1' in args or 'sentence2' in args):
            parser.error('--similarity_type was selected to be from file, but also input sentence was provided, please select one of the options')

        if args.similarity_type == 'input' and (args.sentence1 == None or args.sentence2 == None):
            parser.error('--similarity_type was selected to be input, please provide two sentences to compare')

    return args

if __name__ == '__main__':

    config  = parse()

    if 'similarity_type' in config:
        model_class = class_initiation_dict[config.model_type] \
            (pretrained_model_path=config.model_path,
             train_path = config.input_file,
             output_path = config.output_path,
             use_type = 'similarity',
             distance_type = config.distance_type,
             sim_threshold=config.threshold,
             logging=verbossity_dict[config.verbosity])


        # Only comparing two sentences from user
        if config.similarity_type == 'input':
            print("\n\nSentence 1: {} \nSentence 2: {} \nSimilarity score: {}".format(
                config.sentence1,
                config.sentence2,
                model_class.calculate_sentence_similarity(config.sentence1, config.sentence2)))

        # Comparing sentence in a file
        if config.similarity_type in file_sim_choises:
            # if 'file_best_match' sentence_to_compare will not be None
            model_class._calculate_sentence_similarity_from_file(comparison_type=config.similarity_type ,
                                                                 sentence_to_compare= config.sentence1,
                                                                 sort_by_similarity=config.sort_result)

    if 'train_path' in config:

        model_class = class_initiation_dict[config.model_type](
             train_path = config.train_path,
             input_type=config.input_type,
             output_path = config.output_path,
             use_type = 'train',
             batch_size=config.batch_size,
             epoch_train=config.epochs_number,
             min_count=config.minimum_count,
             window=config.window,
             size=config.size,
             sample=config.sample,
             alpha=config.alpha,
             workers=config.workers,
             logging=verbossity_dict[config.verbosity])

        model_class.train()

    if 'perfects_path' in config:
        model_class = class_initiation_dict[config.model_type] \
            (
             input_type=config.input_type,
             use_type='validation')

        model_class.validate(config.predicted_path, config.perfects_path , config.number_classes, plot_results=config.visualize)

    if 'train_separator' in config:
        model_class = class_initiation_dict[config.model_type] \
            (output_path=config.output_path,
             pretrained_model_path=config.model_path,
             use_type='visualize')

        model_class.merge_datasets_to_one_dataframe(
                                                    train_path=config.train_dataset,
                                                    validatin_path=config.validation_dataset,
                                                    test_path=config.test_dataset,
                                                    dimensions=config.vector_dimensions,
                                                    test_separator=config.train_separator,
                                                    validation_separator=config.validation_separator,
                                                    train_separator=config.train_separator
                                                    )
        model_class.draw_sentences_distribution()