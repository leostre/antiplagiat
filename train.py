import random
import pickle
import sklearn
import compare
import os
from sklearn.model_selection import train_test_split
class Reader:
    def __init__(self, orig='', plag1='', plag2=''):
        self.path_to_original = os.path.join(os.getcwd(), 'plagiat', 'files') if not orig else orig
        self.path_to_plag1 = os.path.join(os.getcwd(), 'plagiat', 'plagiat1') if not plag1 else plag1
        self.path_to_plag2 = os.path.join(os.getcwd(), 'plagiat', 'plagiat2') if not plag2 else plag2
        self.plagiarism = []
        self.original = []

    def get_plagiarism_pairs(self):
        pairs_list = []
        for file in os.listdir(self.path_to_original):
            orig = os.path.join(self.path_to_original, file)
            pairs_list.append((orig, os.path.join(self.path_to_plag1, file)))
            pairs_list.append((orig, os.path.join(self.path_to_plag2, file)))
        return pairs_list

    def get_non_plagiarism(self):
        pairs_list = []
        import random
        files = os.listdir(self.path_to_original)
        def random_not_equal(path):
            for file in files:
                while True:
                    name = files[random.randint(0, len(files) - 1)]
                    if name != file:
                        break
                pairs_list.append((os.path.join(self.path_to_original, file), os.path.join(path, name)))
        random_not_equal(self.path_to_plag1)
        random_not_equal(self.path_to_plag2)
        return pairs_list

class Trainer:

    def __init__(self):
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
        self.model = None

    def matrix(self, need_to_recount=False):
        reader = Reader()
        mtr = []
        try:
            with open('vectors.pkl', 'rb') as file:
                mtr = pickle.load(file)
        except:
            pass
        if not need_to_recount:
            return mtr

        def to_vectors(is_plagiarism):
            pairs = reader.get_plagiarism_pairs() if is_plagiarism else reader.get_non_plagiarism()
            print('General number is ', len(pairs))
            counter = 0
            for pair in pairs:
                print(counter)
                counter += 1
                try:
                    vector = []
                    cmp = compare.Comparer(*pair)

                    parameters = cmp.matches('body')
                    for type in compare.Comparer.TYPES_OF_INTEREST:
                        parameters[f'susp_{type}'] = cmp.suspicious_by_type(type)
                        parameters[f'overlap_{type}'] = cmp.overlapping_by_type(type)
                    for key in sorted(parameters):
                        vector.append(parameters[key])

                    mtr.append((vector, float(is_plagiarism)))
                except:
                    print('Failed to compare: ', pair)
        to_vectors(is_plagiarism=True)
        to_vectors(is_plagiarism=False)
        with open('vectors.pkl', 'wb') as p:
            pickle.dump(mtr, p, -1)
        return mtr


    def distribute(self, matrix):
        x, y = zip(*matrix)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2)

    def linear_regression(self):
        model = sklearn.linear_model.LogisticRegression()
        model.fit(self.x_train, self.y_train)
        self.model = model

    def predict(self):
        return self.model.predict_proba(self.x_test)

    def save_model(self):
        with open('model.pkl', 'wb') as out:
            pickle.dump(self.model, out, -1)





if __name__ == '__main__':
    tr = Trainer()
    tr.distribute(tr.matrix())
    tr.linear_regression()
    scores = sklearn.model_selection.cross_validate(tr.model, tr.x_train, tr.y_train,
                                                    cv=5,
                                                    scoring=('r2', 'neg_mean_squared_error'))
    print(scores)
    # print(tr.x_train)
    # print(tr.y_test)

    print(tr.predict())
    tr.save_model()



