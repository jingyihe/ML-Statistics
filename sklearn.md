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
   1. 字典特征抽取DictVectorizer.fit_trasform(X) X: dictionary return a sparse matrix
      1. DictVectorizer.inverse_trasform(X) X: array or matrix return to the original data type 
   2. 文本特征抽取
      1. sklearn.feature_extraction.text.CountVectorizer(stop_words = []): count frequency for each word; 
         1. return sparse matrix, need use .toarray() to get 2d matrix
         2. for chinese, we have to use space to separate each word or other library
      2. sklearn.feature_extraction.text.TfidVectorizer(): stop_words are those words will not effect the performance; if we just use the frequency of the words, then for some of words like I, we, is…etc, will have higher value/weight; the words we want to find (say key words) are: only popular for some types of topics, not all of the topics ,—— TF-IDF
         1. TF-IDF的主要思想：如果某个词货短语在一篇文章中出现的概率高， 并且在其他文章中很少出现，则认为此词具有很好的区分能力，适合用来分类
         2. TF-IDF : measure the importance of the words
         3. TF: word frequncy
         4. IDF: inverse document frequency=总文件数目除以包含该词语之文件的数目，再讲得到的商取以10为底的对数得到( ln ( total # of files / # of files with this word ))
         5. tfidf_i,j = tf_i,j * idf_i
         6. sklearn.feature_extraction.text.TfidfVectorizer(): the bigger value the better
4. 特征预处理
   1. Data standarize:
      1. MinMaxScaler: not good if there is missing value, which effect the max or min
      2. StandardScaler(): more stable 
   2. reduce dimensions:
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
         2. features are non correlated
         3. sklearn.decomposition.PCA(n_components = None): n_components — with decimal(keep % of informaiton) — with integer(keep # of features)
5. Classification
   1. sklearn transformer and estimator
      1. 实例化一个estimator
      2. estimator.fit(x_train, y_train) - model fitted
      3. performance:
         1. y_predict == y_test
         2. calculate accuracy = estimator.score(x_test, y_test)
   2. KNN: predict its class according to its neighbors
      1. if k = 1, will affect by outlier
      2. if k is too small, will effect by the outlier, if the k is too larger, will effect by the unbalanced dataset
      3. KNN API : sklearn.neighbors.KNeighborsClassifier(n_neighbors = 5, algorithm = 'auto')
      4. advantages: easy to understand to use, and implement
      5. disadvantages: have to select k, value k will affect the performance; time comsuming for big data and use a lot of memory( less < 10,000)
   3. sklearn.model_selection.GridSearchVC(estimator, param_grid = None, cv = None)
      1. best_params_
      2. best_score_
      3. best_estimator_
      4. cv_results_
   4. naive baysian : naive(independent) + baysian
      1. assume features are independent,
      2. popular for text classification: assume words are independent
      3. P(C): 每个文档的概率（某文档类别/总文档数量）
      4. P(W/C) :给定类别下特征（被预测文档中出现的词）的概率：
         1. 计算方法： P(F1|C) = N_i/N (训练文档中去计算)： N_i为该F1词在C类别文档中出现的次数； N为所属类别C下的文档所有词出现的次数和
         2. when P(F1|C) = 0, we use Laplace smoothing coefficient  P(F1|C)  = (N_i + alpha)/(N + alpha*m), where alpha = 1 and m = # of features
      5. P(F_1, F_2, …) 预测文档中每个词的概率 
      6. sklearn.naive_bayes.MultinomialNB()
      7. advantage : stable, more torenrance to missing values, fast
      8. disadvantage: performance is not good for correlated dataset
   5. Decision Tree：如何高效地进行决策
      1. information entropy: H(x) = - $\sum_i^n P(x_i) * log(P(x_i))$
      2. information gain:特征A对训练集D的information gain g(D, A) is defined as H(D) - H(D|A)
      3. conditional entropy: $H(D|A) = \sum_i^n \frac{|D_i|}{|D|}H(D_i) = - \sum_i^n \frac{|D_i|}{|D|}\sum_i^K \frac{|D_{ik}|}{|D_i|} log \frac{|D_{ik}|}{|D_i|}
      4. sklearn.tree.DecisionTreeClassifier(criterion='gini', max_depth = None, random_state = None)
      5. visulization for the decision tree:
         1. sklearn.tree.export_graphviz(estimator, out_file='tree.dot', feature_name=["",""])
         2. copy .dot file to  http://www.webgraphviz.com/ to generate the graph
      6. advantages: easy to explain and understand, visilization the tree
      7. disadvantages: overfit
      8. improve: cart algorithm(cut the leaves), random forest
   6. random forest
      1. random select dataset with the  sample size as the original sample size
      2. random select features 
      3. why randomly select trainning sets: if not, every tree would be the same, then the final result would be the same
      4. why we sample with replacement? If sample without replacement, the every training set is different, no overlap, then every tree might be biased, larger variance
      5. sklearn.ensemble.RandomForestClassifier(n_estimators = 10, criterion = 'gini', max_depth = None, bootstrap = True, random_state = None, min_samples_split = 2); max_features = 'auto'/'sqrt'/'log2'
      6. advantage: do not need to reduce dimensions, effective on big data, feature importance
   7. Linear regresssion:
      1. sklearn.linear_model.LinearRegression(fit_intercept = True)
      2. sklearn.linear_model.SGDRegressor(loss = 'squared_loss',fit_intercept = True,learning_rate ='invscaling',eta0 = 0.01); invscaling: eta = eta0/pow(t, power_t), power_T= 0.25
      3. SGDRegressor.coef_: coefficients
      4. SGDRegressor.intercept_: intercept
      5. sklearn.linear_model.Ridge(): 
      6. measure performance: MSE; 
   8. 模型保持与加载：
      1. from sklearn.externals import joblib
      2. save: joblib.dump(rf, 'test.pki')
      3. load: estimator = joblib.load('test.pki')
   9. unsuperwised learning:
      1. K - means:
         1. 1) pre define K; 2) grid search to find the K
         2. for each data points, calculate the distance to K centers, assign the point to the closed center
         3. recalculate the center with the new groups
         4. stop unitl converge
         5. sklearn.cluster.KMeans(n_clusters = 8, init = "k-means++")
         6. measure Kmeans performance: Silhouette Coefficient：$sc_i = \frac{b_i - a_i}{max(b_i, a_i)}$
         7. $bi$ is the minimum distance from i to other data points with different data centers
         8. $a_i$ is the average distance of i to all the data points with the same center
         9. if $sc_i = 1$ means the performance is good, if $sc_i = -1$ means bad
         10. sklearn.metrics.silhouette_score()
         11. advantage: iterative algorithm, easy to use and understand, can be used without target
         12. disadvantage: get the local optimizer not global
   10. C

