
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

from load import loadImages

if __name__ == "__main__":
    images, labels = loadImages("./img")

    svm = LinearSVC(max_iter=10000)

    images = PCA(n_components=2).fit_transform(images)

    accuracy, kfold = [], KFold(n_splits=5, shuffle=True)
    for train_indices, test_indices in kfold.split(images):
        train_data, test_data = images[train_indices], images[test_indices]
        train_labels, test_labels = labels[train_indices], labels[test_indices]

        svm.fit(train_data, train_labels)

        accuracy.append(accuracy_score(
            test_labels, svm.predict(test_data), normalize=True)*100)

    print(accuracy)
