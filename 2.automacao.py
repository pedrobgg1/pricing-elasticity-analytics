#%%
# Modelo de automatização de faturamento para possiveis aumentos de preço
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Carregando as duas base de dados
df_base = pd.read_csv("data/retail_sales.csv", sep=',')
df_clusters = pd.read_csv("data/TbClusters.csv", sep=';')
#%%
# Unindo a base de dados para descobrir os clusters da base original
df_geral = df_base.merge(how="left", right=df_clusters, on=["store_id","item_id"], suffixes=[{"geral":"cluster"}])
df_geral.value_counts(df_geral["cluster"])

df_geral = df_geral[["store_id", "item_id","sales","price","promo","month","cluster"]]

# Realizando a regreção de cada cluster para descobrir a elasticidade do preço da demanda

elasticidades = {}

df_regressao = df_geral[(df_geral['sales'] > 0) & (df_geral['price'] > 0)].copy()

for i in range(4):

    df_clust = df_regressao[df_regressao['cluster'] == i]

    # Sales + 1 pois contem valores 0 o qual quebra o log
    modelo = smf.ols(formula='np.log(sales + 1) ~ np.log(price) + promo + C(month)', data=df_clust).fit()
    
    elasticidades[f'Cluster_{i}'] = modelo.params['np.log(price)']
    print(f"Cluster {i}")

print(elasticidades)

elast_cl0 = float(elasticidades["Cluster_0"])
elast_cl1 = float(elasticidades["Cluster_1"])

# Automatização. 
def calculadora_faturamento():
    cluster = int(input("Escolha se o produto esta no cluster 0 ou 1"))
    if cluster == 0 or cluster == 1:
        preco = float(input("Insira o preço do produto"))
        qnt = int(input("Insira a quantidade vendida"))
        aumento = float(input("Insira o aumento de preço proposto. EX: 0.10 para 10% de aumento"))
        if cluster == 0:
            clustertype = elast_cl0
        else:
            clustertype = elast_cl1
        # Calular o novo preço e quantidade
        preco_novo = preco * (1+aumento)
        qnt_nova = qnt * (1+(aumento * clustertype))
        # Calcular os faturamentos
        faturamento_antg = preco * qnt
        faturamento_novo = preco_novo * qnt_nova
        faturamento_final = faturamento_novo - faturamento_antg
        # Apresentar os resultados finais
        print("-" * 40)
        print(f"Para um produto no cluster {cluster}")
        print(f"Com um acréscimo de: {aumento*100}% no preço.")
        print(f"O faturamento passaria de: R${faturamento_antg:.2f}, para: R${faturamento_novo:.2f}.")
        print(f"Resultando em um impacto de: R${faturamento_final:.2f}, no faturamento final")
        print("-" * 40)
    else: print("Escolha um cluster válido")

calculadora_faturamento()

# %%
