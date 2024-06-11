import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from py2neo import Graph

uri = 'http://localhost:7474'
user = 'neo4j'
password = 'zj19960522'
graph = Graph(uri, auth=(user, password))

query_features = """
MATCH (ds:Dataset)-[r1]-(sm:Surrogate_model)-[r2]-(smm:Surrogate_modeling_method)
RETURN ds.dim AS dim, ds.size AS size, ds.nonlinearity AS nonlinearity, ds.mean AS mean, ds.harmonic_mean AS harmonic_mean, ds.standard_deviation AS standard_deviation, ds.median AS median, ds.MSE_PRSM_1 AS MSE_PRSM_1, ds.RMSE_PRSM_1 AS RMSE_PRSM_1, ds.MAE_PRSM_1 AS MAE_PRSM_1, ds.MSE_PRSM_2 AS MSE_PRSM_2, ds.RMSE_PRSM_2 AS RMSE_PRSM_2, ds.MAE_PRSM_2 AS MAE_PRSM_2, ds.gradient1 AS gradient1, ds.gradient2 AS gradient2, ds.gradient3 AS gradient3, ds.gradient4 AS gradient4, sm.accuracy AS accuracy, smm.name AS smm
"""

features_data = graph.run(query_features).data()

features_df = pd.DataFrame(features_data)

features_df_grouped = features_df.groupby(list(features_df.columns[:-2])).agg(
    max_accuracy=pd.NamedAgg(column="accuracy", aggfunc="max"),
    best_smm=pd.NamedAgg(column="accuracy", aggfunc=lambda x: features_df.loc[x.idxmax(), "smm"])
).reset_index()

feature1 = features_df_grouped.iloc[:, :-2].values

metrics1 = features_df_grouped["max_accuracy"].values
label1 = features_df_grouped["best_smm"].values


#归一化
min_max_scaler = preprocessing.MinMaxScaler()
feature1 = min_max_scaler.fit_transform(feature1)

feature2 = feature1[:, :3]


for j in range(feature1.shape[0]):

    feature3 = feature1[j].reshape(1, -1)
    feature4 = feature2[j].reshape(1, -1)

    distances = np.linalg.norm(feature2 - feature4, axis=1)
    num_samples = int(feature1.shape[0] * 0.3)
    nearest_indices = np.argsort(distances)[:num_samples]

    feature1_selected = feature1[nearest_indices]
    label1_selected = [label1[idx] for idx in nearest_indices]

    knn = KNeighborsClassifier(n_neighbors= 5)
    knn.fit(feature1_selected, label1_selected)
    predict_label2 = knn.predict(feature3)[0]
    print(predict_label2)





