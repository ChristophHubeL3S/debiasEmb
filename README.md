## Debiasing Word Embeddings from Sentiment Associations in Names

This is the code for the debiasing approach as presented in our paper "Debiasing Word Embeddings from Sentiment Associations in Names".

If you are interested in the news data and the trained embeddings that we used, you can find them here: https://data.uni-hannover.de/dataset/128e47f6-4cae-4b20-8815-4bb94abb8df5

Feel free to use our approach and/or data and please cite our paper.

What follows is a brief description of how to use this code to create debiased word embeddings using a text dataset and a list of specified words to debias (e.g. names) and how to evaluate them.

### Training an Oracle Sentiment Classifier
First, we have to train the oracle sentiment classifier and save the weights. For this step you need skip-gram embeddings of the input text. If you do not have them yet,
you can use the train.py script as described in the next step with the option -m w2v. You can also use your own lists of positive and negative words and names (or other words to debias) or you 
use the ones that we provide in the data folder.

Example call:

```
python ./evaluation/sent_classifier.py -e <path_to_your_embeddings> -p <path_to_your_positive_words> -n <path_to_your_negative_words> -ev <path_to_your_names> -o <path_to_your_output_folder>
```

The output directory will contain the saved model and the weights.

### Training Debiased Word Embeddings
The script ```train.py``` can be used to train two types of word representations: (i) skip-gram embeddings, and (ii) debiased word embeddings. 

The required parameters for training word representations are the following. First, it is necessary to specify your ``input text file``, ``the output directory``, and the ``path to the oracle weights .txt`` file that you created in the previous step. Namely, the ```train.py``` has the following parameters:

* ```-i``` is the parameter for the ```input text file``` which is used as the data for training the embeddings.
* ```-o``` is the parameter that specifies the ```output directory```.
* ```-w``` is the parameter that specifies the ```path to the oracle weights``` (see **Training an Oracle Sentiment Classifier**). _This parameter is necessary only when training for **debiased word embeddings**_.
* ```-m``` is the parameter that specifies the **type** of word representations. Two options are valid: (i) _w2v_, and (ii) _debiasw2v_. 
* ```-es``` is the parameter that specifies the path to the standard **word2vec representations** which are used to initialize the **debiased word representations**. _Note that this is the way we have trained the debiased representations in our experimental evaluation. We debias the pretrained **skip-gram** word representations for several iterations (less than 5, based on the loss convergence)_.
* ```-b``` is the parameter determining the batch size.
* ```-n``` is the parameter pointing to the ```file containing the names``` for which we want to debias their representations.

Example call:

```
python train.py -i <path_to_your_text_file_for_training> -o <path_to_your_output_directory> -w <path_to_your_oracle_weights> -m debias -n <path_to_your_list_of_words_to_be_debiased> -b 64 [-es <path_to_the_w2v_embeddings>]
```

The output directory will contain the trained model after the last epoch and the embeddings in standard w2v format.

### Evaluating on Word-level
To evaluate the embeddings on the sentiment classifier, simply run the eval_emb.py script.

Example call:

```
python ./evaluation/eval_emb.py -e <path_to_your_embeddings> -m <path_to_the_oracle_classifier> -o <path_to_output_directory>
```

### Evaluating on a Downstream Task
If you want to evaluate your embeddings on a downstream sentence-level classifier, your can use the downstream_eval.py script.
You need to provide a dataset containing labeled sentences. In our paper, we used the dataset that you find under /data/labeled_news_sentences.

Example call:

```
python ./downstream_eval.py -e <path_to_your_embeddings> -d <path_to_labeled_sentences>
```


If you have any questions or you are experiencing any issues do not hesitate to let me know: hube@l3s.de





