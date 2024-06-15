from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# load the iris dataset

iris = load_iris()
X, y = iris.data, iris.target

# train the logistic regression model
model = LogisticRegression()
model.fit(X, y)


def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    features = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(features)
    return iris.target_names[prediction[0]]
