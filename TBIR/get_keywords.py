from sklearn.feature_extraction.text import CountVectorizer
import nltk
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.stem import PorterStemmer
from tqdm import tqdm

def get_descriptions(file):

	descriptions_data = dict()

	file = open(file).read()

	descriptions = file.split('\n')

	ps = PorterStemmer()

	for description in descriptions:

		id_plus_descriptions = description.split(' ')

		idx = id_plus_descriptions[0]

		descriptions_list = id_plus_descriptions[1:]

		descriptions_list = [ps.stem(desc) for desc in descriptions_list]

		description = ' '.join(descriptions_list) 

		descriptions_data[idx] = description

	return descriptions_data

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
 
def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]
 
    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
 
    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results

stop_words = nltk.corpus.stopwords.words("english")

#Editing max_df and min_df would give different answers
Count_Vectorizer = CountVectorizer(max_df=0.3,min_df=0.005,stop_words=stop_words)

#get descriptions as dict of ID and description
descriptions_data = get_descriptions('descriptions.txt')

descriptions = [descriptions_data[description] for description in descriptions_data]

word_count_vectors = Count_Vectorizer.fit_transform(descriptions)
feature_names = Count_Vectorizer.get_feature_names()
tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
tfidf_transformer.fit(word_count_vectors)

file = open('keyword_description.txt','a+')

for key in tqdm(descriptions_data):

	description = descriptions_data[key]

	tf_idf_vector = tfidf_transformer.transform(Count_Vectorizer.transform([description]))

	sorted_vector = sort_coo(tf_idf_vector.tocoo())

	keywords = extract_topn_from_vector(feature_names,sorted_vector,5)

	text = key+ '\t' +' '.join(list(keywords))+'\n'

	if len(list(keywords)) >2:
		file.write(text)

file.close()