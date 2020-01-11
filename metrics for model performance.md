For different model types, we use different metircs:

1. classification (predict category): 
   1. accuracy (# of samples predicted correct / total # of samples), precision(true positive/predicted positive), recall(predicted positive/true positive)/sensitivity, F1, ROC(Receiver Operator Characteristic) and AUC (the area under the curve) (sensitivity/TPR vs specificity/FPR): we want to have higher TPR and lower FPR
   2. important notes: for some example like disease detection, we may focus on recall, we want to find all the positives from true positives; another thing is when the data is unbalanced, we found precision and recall are not good measurements, then use ROC/AUC, we may have threshold for FPR
   3. Q: would it be a good idea if we add confidence interval for precision/recall? The idea is from precision is measuring the accuracy for the prediction
2. Regression(predict value): 1. SSE, MAE(mean absolute error), RMS