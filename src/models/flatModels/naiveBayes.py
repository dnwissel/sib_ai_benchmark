from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB


from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import loguniform

from models.wrapper import Wrapper

params = dict(
        name='NaiveBayes',
        # model=ComplementNB(force_alpha=True),
        model=MultinomialNB(force_alpha=True),
        preprocessing_steps=[('StandardScaler', MinMaxScaler())],

        # model=GaussianNB(),
)

# Please don't change this line
wrapper = Wrapper(**params)


if __name__ == "__main__":
    pass