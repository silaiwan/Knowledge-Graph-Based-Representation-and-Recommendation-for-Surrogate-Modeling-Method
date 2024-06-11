import pandas as pd
from py2neo import Graph, Node, Relationship

def read_csv(file_path):
    df = pd.read_csv(file_path)
    return df

def create_nodes_and_relationships_of_data_sets(graph, df):
    for index, row in df.iterrows():
        node_dataset = Node("Dataset", name=row['dataset'], dim=row['dim'], size=row['size'], nonlinearity = row['nonlinearity'],
                            mean = row['mean'], harmonic_mean = row['harmonic_mean'], standard_deviation = row['standard_deviation'],
                            median = row['median'], MSE_PRSM_1 = row['MSE_PRSM_1'], RMSE_PRSM_1 = row['RMSE_PRSM_1'], MAE_PRSM_1 = row['MAE_PRSM_1'],
                            MSE_PRSM_2 = row['MSE_PRSM_2'], RMSE_PRSM_2 = row['RMSE_PRSM_2'], MAE_PRSM_2 = row['MAE_PRSM_2'],
                            gradient1 = row['gradient1'], gradient2 = row['gradient2'], gradient3 = row['gradient3'],gradient4 = row['gradient4'])
        graph.merge(node_dataset, "Dataset", "name")

        node_BF = Node("BF", name = row["BF"])
        graph.merge(node_BF , "BF", "name")

        relationship = Relationship(node_dataset, "belong to", node_BF)
        graph.merge(relationship)

def create_nodes_and_relationships_of_surrogate_modeling_methods(graph, df):
    for index, row in df.iterrows():
        node_Surrogate_modeling_method = Node("Surrogate_modeling_method", name=row['Surrogate modeling method'], model_type=row['model type'], function_type=row['function type'])
        graph.merge(node_Surrogate_modeling_method, "Surrogate_modeling_method", "name")


def create_nodes_and_relationships_of_surrogate_models(graph, df):
    for index, row in df.iterrows():
        node_Surrogate_model = Node("Surrogate_model", name=row['Surrogate model'], accuracy=row['accuracy'], robustness=row['robustness'], time = row['time'])
        graph.merge(node_Surrogate_model, "Surrogate_model", "name")

        node_dataset = graph.nodes.match("Dataset", name=row['dataset']).first()

        if node_dataset:
            relationship1 = Relationship(node_dataset, "hasSurrogate-model", node_Surrogate_model)
            graph.merge(relationship1)

        node_Surrogate_modeling_method = graph.nodes.match("Surrogate_modeling_method", name=row['Surrogate modeling method']).first()

        if node_Surrogate_modeling_method:
            relationship2 = Relationship(node_Surrogate_model, "hasSurrogate-modeling-method", node_Surrogate_modeling_method)
            graph.merge(relationship2)

if __name__ == "__main__":
    file_path1 = "Data sets.csv"
    file_path2 = "Surrogate modeling methods.csv"
    file_path3 = "Surrogate models.csv"
    uri = 'http://localhost:7474'
    user = 'neo4j'
    password = 'zj19960522'

    df1 = read_csv(file_path1)
    df2 = read_csv(file_path2)
    df3 = read_csv(file_path3)

    graph = Graph(uri, auth=(user, password))

    create_nodes_and_relationships_of_data_sets(graph, df1)
    create_nodes_and_relationships_of_surrogate_modeling_methods(graph, df2)
    create_nodes_and_relationships_of_surrogate_models(graph, df3)
