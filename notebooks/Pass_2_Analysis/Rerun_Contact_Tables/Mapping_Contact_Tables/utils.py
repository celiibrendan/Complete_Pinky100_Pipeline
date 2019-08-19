#calculates the pearson correlation and cosine similarity while accounting for the corner cases
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
import warnings

def check_nan(v1,v2):
    if np.isscalar(v1):
        if np.isnan(v1) or v1 == None:
            return True
    else:
        if True in np.isnan(v1) or len(v1) <= 0:
            return True
        
    if np.isscalar(v2):
        if np.isnan(v2) or v2 == None:
            return True
    else:
        if True in np.isnan(v2) or len(v2) <= 0:
            return True
    
    return False

def find_pearson_old(v1,v2):
    v1 = v1.astype("float")
    v2 = v2.astype("float")
    print("v1 = " + str(v1))
    print("v2 = " + str(v2))
    if check_nan(v1,v2):
        return np.NaN
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if np.array_equal(v1,v2):
            return 1
        elif abs(sum(v1 - v2)) >= v1.size:
            return -1
        else:
            #perform the pearson correlation
            corr_conversion, p_value_conversion = pearsonr(v1, v2)
            return corr_conversion

def find_cosine_old(v1,v2):
    v1 = v1.astype("float")
    v2 = v2.astype("float")
    if check_nan(v1,v2):
        return np.NaN
    if np.array_equal(v1,v2):
        return 1
    elif abs(sum(v1 - v2)) == v1.size:
        return 0
    else:
        v1 = v1.reshape(1,len(v1))
        v2 = v2.reshape(1,len(v2))
        return cosine_similarity(v1, v2)[0][0]



def find_pearson(v1,v2):
    v1 = v1.astype("float")
    v2 = v2.astype("float")
#     print("v1 = " + str(v1))
#     print("v2 = " + str(v2))
    if check_nan(v1,v2):
        return np.NaN
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        #perform the pearson correlation
        if v1.size <= 1 or v2.size <= 1:
            return np.NaN
        if np.array_equal(v1,v2):
            return 1
        elif abs(sum(v1 - v2)) >= v1.size:
            return -1
        else:
            corr_conversion, p_value_conversion = pearsonr(v1, v2)
            return corr_conversion

def find_cosine(v1,v2):
    v1 = v1.astype("float")
    v2 = v2.astype("float")
    if check_nan(v1,v2):
        return np.NaN
    if v1.size <= 1 or v2.size <= 1:
            return np.NaN
    if np.array_equal(v1,v2):
        return 1
    elif abs(sum(v1 - v2)) == v1.size:
        return 0
    else:
        v1 = v1.reshape(1,len(v1))
        v2 = v2.reshape(1,len(v2))
        return cosine_similarity(v1, v2)[0][0]

def find_binary_sim(v1,v2):
    v1 = v1.astype("float")
    v2 = v2.astype("float")
    if check_nan(v1,v2):
            return np.NaN
    a = np.dot(v1,v2)
    b = np.dot(1-v1,v2)
    c = np.dot(v1,1-v2)
    d = np.dot(1-v1,1-v2)
    
    return (a)/(a + b + c + d),(a + d)/(a + b + c + d)