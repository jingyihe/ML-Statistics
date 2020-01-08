Cramer's V  for calculating correlation of large dataset:

The smaller means week relationship.
Reference : http://www.utstat.toronto.edu/~brunner/oldclass/378f16/readings/CohenPower.pdf
df=5   (small=.05,     medium=.13,   large=.22)
https://www.researchgate.net/post/How_can_I_intepret_the_effect_sizes_of_Cramers_V_when_DF_3



import numpy as np
import scipy.stats as ss
def cramers_corrected_stat(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))
