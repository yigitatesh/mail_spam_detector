import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, hstack

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.linear_model import LogisticRegression

# LOAD DATA
spam_df = pd.read_csv("spam.csv")

# ENCODİNG TARGETS
# ham = 0, spam = 1
target_dict = {0: "ham", 1: "spam"}
spam_df["target"] = np.where(spam_df["target"] == "ham", 0, 1)

# FUNCTIONS
def add_col(X, column_df):
    if isinstance(column_df, pd.Series):
        column_sparse = csr_matrix(column_df.values.reshape(-1, 1))
    else:
        column_sparse = csr_matrix(column_df)
    return hstack((X, column_sparse))

# predictor function
def predict(mail, model, cv):
	## prepare mail to prediction
	mail_df = pd.Series(data=[mail], name="text")
	mail_vec = cv.transform(mail_df)
	# len of email
	mail_vec = add_col(mail_vec, mail_df.apply(lambda x: len(x)))
	# max word len of email
	mail_vec = add_col(mail_vec, mail_df.apply(lambda x: np.max([len(i) for i in x.split()])))	
	# count of uppercase letters
	mail_vec = add_col(mail_vec, mail_df.apply(lambda x: len([1 for i in x if i.isupper()])))
	# count of digits
	mail_vec = add_col(mail_vec, mail_df.apply(lambda x: len([1 for i in x if i.isdigit()])))

	## make the prediction
	prediction = model.predict(mail_vec)
	
	return np.squeeze(prediction)

# main function
def main(model, cv):
	run = True
	while run:
		mail = input("Enter the mail to check spam or not (q to quit): ")

		if mail == "q":
			run = False
			continue

		prediction = predict(mail, model, cv)

		# inform the user
		if prediction == 0: # not spam
			print("It is not spam.")
		else: # 1 = spam
			print("It is spam.")


# CREATE THE MODEL
X = spam_df["text"]
y = spam_df["target"]

cv = CountVectorizer(stop_words=None, ngram_range=(1, 1))
X_vec = cv.fit_transform(X)

# len of emails
X_vec = add_col(X_vec, X.apply(lambda x: len(x)))

# max word len of emails
X_vec = add_col(X_vec, X.apply(lambda x: np.max([len(i) for i in x.split()])))

# count of uppercase letters
X_vec = add_col(X_vec, X.apply(lambda x: len([1 for i in x if i.isupper()])))

# count of digits
X_vec = add_col(X_vec, X.apply(lambda x: len([1 for i in x if i.isdigit()])))

# fit the model
lr = LogisticRegression(max_iter=1000).fit(X_vec, y)

if __name__ == "__main__":
	main(lr, cv)
