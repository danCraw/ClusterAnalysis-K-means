import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# читаем дейтафрейм и создаем экземпляр label encoder
label_encoder = preprocessing.LabelEncoder()
dataframe = pd.read_csv('tourist_trips.csv', delimiter=';')
samples = list(dataframe.values)
samples_transformed = []

for sample in samples:
    label_encoder.fit(sample)
    samples_transformed.append(label_encoder.fit_transform(sample))

# выполняем иерархическую кластеризацию
mergings = linkage(np.array(samples_transformed), method='complete')

# строим дендрограмму и сохраняем в файл
plt.figure(figsize=(20, 8), dpi=300)
labels = [i for i in range(0, 82)]  # создаем массив уникальных лейблов для каждого эксперимента
label_encoder.fit(labels)
labels_list = labels

dendrogram(mergings,
           labels=np.array(labels),
           leaf_rotation=82,
           leaf_font_size=6,
           )
plt.savefig('./Dendrogram_sweets')

# создаем метод главных компонент с двумя компонентами
pca = PCA(2)

# делаем сопоставление и трансформацию
transformed_data = pca.fit_transform(samples_transformed)
cluster_labels = ['Days', 'ExtrCharges', 'Tips', 'USD']

# инициализируем экземпляр метода средних
kmeans = KMeans(n_clusters=len(cluster_labels))

# предсказываем лейблы к трансформированным данным
label = kmeans.fit_predict(transformed_data)

# получаем центроиды от метода средних
centroids = kmeans.cluster_centers_

# выводим результаты метода средних
plt.figure(figsize=(14, 7), dpi=300)

for i in range(0, len(cluster_labels)):
    plt.scatter(transformed_data[label == i, 0], transformed_data[label == i, 1], label=cluster_labels[i])

plt.scatter(centroids[:, 0], centroids[:, 1], s=80, color='k')
plt.legend()
plt.savefig("./KMeans_sweets")

# рисуем график каменистой осыпи
distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
K = range(1, 20)
for k in K:
    # строим модель для каждого значения кластеров в диапазоне от 1 до 20 и строим график
    kmeanModel = KMeans(n_clusters=k).fit(transformed_data)
    kmeanModel.fit(transformed_data)
    distortions.append(sum(np.min(cdist(transformed_data, kmeanModel.cluster_centers_,
                                        'euclidean'), axis=1)) / transformed_data.shape[0])
    inertias.append(kmeanModel.inertia_)
    mapping1[k] = sum(np.min(cdist(transformed_data, kmeanModel.cluster_centers_,
                                   'euclidean'), axis=1)) / transformed_data.shape[0]
    mapping2[k] = kmeanModel.inertia_

plt.figure(figsize=(14, 7), dpi=300)
plt.plot(K, distortions, 'bx-')
plt.xlabel('Значения K')
plt.title('График каменистой осыпи ')
plt.savefig("./Elbow_sweets")