## Debiasing Word Embeddings from Sentiment Associations in Names

This is the code for the debiasing approach as presented in our paper "Debiasing Word Embeddings from Sentiment Associations in Names".

If you are interested in the news data and the trained embeddings that we used, you can find them here: https://data.uni-hannover.de/dataset/128e47f6-4cae-4b20-8815-4bb94abb8df5

Feel free to use our approach and/or data and please cite our paper.

What follows is a brief description of how to use this code to create debiased word embeddings using a text dataset and a list of specified words to debias (e.g. names) and how to evaluate them.

### Training Debiased Word Embeddings
Use the train.py script. You have to specify your input text file and the output directory by using the -i and -o parameters.
To active debiasing you also have to select the debiasEmb model by calling -m debias. Otherwise the plain skip-gram approach will
be used for training. You can specify further parameters like the path to the name list, embedding dimensions, batch size etc.

Example call:

python train.py -i <path_to_your_text_file_for_training> -o <path_to_your_output_directory> -m debias -n 
<path_to_your_list_of_words_to_be_debiased> -b 64

The output directory will contain the trained model after the last epoch and the embeddings in standard w2v format.

### Evaluating Debiased Word Embeddings
To evaluate your word embeddings, you can use the evaluate.py script.



If you have any questions or you are experiencing any issues do not hesitate to let me know: hube@l3s.de





