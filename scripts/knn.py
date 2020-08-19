
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier

from load import load_images

if __name__ == "__main__":
    images, labels = load_images("./img", size=None)

    print(images.shape)

    knn = KNeighborsClassifier(n_neighbors=1)

    images = PCA(n_components=2).fit_transform(images)

    accuracy, kfold = [], KFold(n_splits=5, shuffle=True)
    for train_indices, test_indices in kfold.split(images):
        train_data, test_data = images[train_indices], images[test_indices]
        train_labels, test_labels = labels[train_indices], labels[test_indices]

        knn.fit(train_data, train_labels)

        accuracy.append(accuracy_score(
            test_labels, knn.predict(test_data), normalize=True)*100)

    print(accuracy)
