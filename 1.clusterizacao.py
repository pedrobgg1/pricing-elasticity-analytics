#%%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

#%%

# Carregar base de dados

df = pd.read_csv("data/retail_sales.csv", sep=',')

df

#%%

# Agrupar por IDs
# Realizar a média do preço e promoção e somar quantidade

df_groupby = (df.groupby(by=["store_id","item_id"]).agg(
                                     total_vendas = ("sales",'sum'),
                                     media_preco = ("price",'mean'),
                                     media_promocao = ("promo",'mean')
                                     ).reset_index()
)
df_groupby


#%%
# Normalização dos dados

X = df_groupby[["total_vendas","media_preco","media_promocao"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#%%
# Descobrir o melhor número de clusters usando o método Elbow 

teste = []
for i in range(1, 11): # Testa k de 1 a 10
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    teste.append(kmeans.inertia_)

plt.plot(range(1, 11), teste)
plt.title('Método do Cotovelo')
plt.xlabel('Número de clusters (k)')
plt.ylabel('teste')
plt.show()


#%%

# Realizar a clusterização

modelo = KMeans(n_clusters=4,random_state=42,n_init='auto')

grupos = modelo.fit_predict(X_scaled)

df_groupby["cluster"] = grupos

# Enviar a nova tabela para csv, separando melhor os códigos

df_groupby.to_csv("TbClusters.csv", sep=';', index=False)


#%%

# agrupar por cluster para análisar as estátisticas de cada um

df_groupclusters = (df_groupby.groupby(by='cluster').agg(
                                    total_vendas = ("total_vendas",'sum'),
                                    media_preco = ("media_preco",'mean'),
                                    media_promocao = ("media_promocao",'mean')
))

