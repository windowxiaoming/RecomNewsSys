from sklearn.feature_extraction.text import TfidfVectorizer
documents = []
f = open('165.pkl','rb')
import pickle
D = pickle.load(f)

tfidf_model = TfidfVectorizer(lowercase=False)
X = tfidf_model.fit_transform(D['documents'])
