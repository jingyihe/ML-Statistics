1. how to get sklearn.datasets

   1. datasets.load_*() for smal dataset in datasets
   2. datasets.fetch_*(data_home=None,subset = 'train'/'test'/'all') for big datasets, need to download, use data_home to save dataset, the default is ~/scikit_learn_data/
   3. the data from load/fetch is datasets.base.Bunch(dictionary)
      1. data: 2d numpy.ndarray
      2. target
      3. Descr: description
      4. feature_names
      5. target_names

2. Split datasets to trainning set and test set: sklean.model_selection.train_test_split(array, *options)

3. Feature extraction(sclera.feature_extraction):sklearn.feature_extraction.DictVectorizer(sparse = True)

   1. DictVectorizer.fit_trasform(X) X: dictionary return a sparse matrix
   2. DictVectorizer.inverse_trasform(X) X: array or matrix return to the original data type 

   1. sklearn.feature_extraction.text.CountVectorizer(stop_words = []): count frequency for each word; 
      1. return sparse matrix, need use .toarray() to get 2d matrix
      2. for chinese, we have to use space to separate each word or other library
      3. sklearn.feature_extraction.text.TfidVectorizer()
      4. stop_words are those words will not effect the performance
   2. if we just use the frequency of the words, then for some of words like I, we, is…etc, will have higher value/weight; the words we want to find (say key words) are: only popular for some types of topics, not all of the topics ,—— TF-IDF
      1. TF-IDF的主要思想：如果某个词货短语在一篇文章中出现的概率高， 并且在其他文章中很少出现，则认为此词具有很好的区分能力，适合用来分类
      2. TF-IDF : measure the importance of the words
      3. TF: word frequncy
      4. IDF: inverse document frequency=总文件数目除以包含该词语之文件的数目，再讲得到的商取以10为底的对数得到( ln ( total # of files / # of files with this word ))
      5. tfidf_i,j = tf_i,j * idf_i
   3. sklearn.feature_extraction.text.TfidfVectorizer(): the bigger value the better

4. Data standarize:

   1. MinMaxScaler: not good if there is missing value, which effect the max or min
   2. StandardScaler(): more stable 

5. reduce dimensions:

   1. select feature: 
      1. filter: 
         1. variance/sd (remove feature that variance is small ); 
            1. sklearn.feature_selection.VarianceThreshold(threshold = 0.0)
               1. delete features that variance is low
               2. Variance.fit_transform(X)
         2.  correlation:
            1. pearson correlation coefficient: chose one of them if several features are higher correlated ; or weighted those features
      2. Embedded: decision tree, regularization; deep learning
   2. PCA:
      1. reduce the dimension, may create new features or not, lose as least as information
      2. sklearn.decomposition.PCA(n_components = None): n_components — with decimal(keep % of informaiton) — with integer(keep # of features)

6. 

































