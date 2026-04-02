# Automação de Pricing: Simulando o Impacto de Aumentos nos Preços

O objetivo deste projeto é automatizar uma das etapas mais cruciais na tomada de decisão da manutenção de preço de produtos, **o cálculo da elasticidade-preço da demanda**. Este processo é essencial para entender o quão disposta sua base de clientes está a aceitar váriações nos preços. A automatização dessa etapa garante que o foco da equipe se volte em especial para **estratégia de mercado** e **percepção de valor**.

## Base de dados

Para este projeto, foi utilizada uma base de dados pública do *Kaggle* chamada *Store Item Demand Forecasting Dataset*. Sua escolha se deu por conter as informações necessárias para a automação e pelo grande volume de dados, chegando a aproximadamente **4 milhões** de observações.

Esta base contém **50 produtos vendidos** em **50 lojas**, com preço e quantidade vendida, além da data e se o item estava em promoção no dia.

A Base de Dados está disponível em: [Acesse a Base De Dados](https://www.kaggle.com/datasets/dhrubangtalukdar/store-item-demand-forecasting-dataset)

## Metodologia

A análise foi separada em dois principais focos:

### Machine Learning - K-Means
Para analisar a elasticidade de uma base de dados com **4 milhões** de observações, é necessário um poder computacional elevado. Por esse motivo, escolhi agrupar cada combinação de Loja-Produto, calculando as médias de preço e promoção e somando a quantidade de vendas.

Após isso, utilizando o algoritmo *K-Means* de *machine learning*, foi possível separar cada agrupamento em grupos distintos.

Para uma assertividade maior, utilizei o *Elbow Method*, garantindo a melhor escolha do número de clusters, o que resultou em **4 clusters**.

<p align="center">
<img src="Img\Elbow.png" width="400px" alt="Gráfico do Elbow Method">
</p>

O código final de clusterização foi:
 ```python       
    X = df_groupby[["total_vendas","media_preco","media_promocao"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    modelo = KMeans(n_clusters=4,random_state=42,n_init='auto')
    grupos = modelo.fit_predict(X_scaled)
    df_groupby["cluster"] = grupos
```

### Elasticidade

O princípio do cálculo da elasticidade-preço da demanda é entender o quanto uma modificação no preço impacta na demanda do produto, ou seja, na quantidade vendida dele. 

Esta conta permite testar várias possibilidades, descobrindo se o aumento no preço diminuiria tanto a quantidade vendida que o faturamento final se tornaria menor que o atual.

Sua fórmula consiste na variação da quantidade demandada, dividida pela variação do preço.

Ela é expressada desta forma:
    $$
    EPD = \frac{\frac{ΔQuantidade}{Quantidade}}{\frac{ΔPreco}{Preco}}
    $$

### Regressão

Uma das melhores formas de descobrir a elasticidade é utilizando a regressão de Mínimos Quadrados Ordinários (MQO), já que ela consegue calcular a influência do preço na quantidade demandada. Em outras palavras, ela nos dá o resultado de quanto um acréscimo no preço modifica a quantidade.

A fórmula de uma regressão de MQO consiste em uma variável dependente *($Y$)*, que em nosso caso será a quantidade vendida, e um número de variáveis independentes *($X$)*. 

Para garantir uma confiabilidade maior na regressão, adicionei outras variáveis que podem estar relacionadas à variação da quantidade, como os meses e se houve promoção. Isso se torna importante para diminuir a influência de fatores externos que também afetam a quantidade, melhorando o cálculo do impacto real que o preço tem sobre as vendas.

Além disso, os valores de quantidade e preço foram transformados em logaritmos. Isso foi feito tanto para evitar que outliers distorcessem a regressão, quanto pelo fato de que, em um modelo Log-Log, o coeficiente do preço passa a representar diretamente a elasticidade-preço da demanda.

A fórmula final da regressão ficou:
    $$
    Log(Quantidade) = \beta_0 + \beta_1 Log(Preco) + \beta_2 Promocao + \beta_3 Mes_2 + \dots + \beta_{13} Mes_{12} + e
    $$
    
(O mês 1 ficou de fora para não cair na armadilha das dummies.)

A regressão foi calculada para cada cluster, separando assim a elasticidade do preço encontrada em cada grupo de produtos.

O resultado final das elasticidades foi:

* **Cluster 0:** -0.13922

* **Cluster 1:** -0.03366

* **Cluster 2:** 0.02824

* **Cluster 3:** 0.16965


Segundo a teoria econômica, na maioria dos casos a elasticidade-preço da demanda deve ser negativa, já que a curva de demanda é descendente (quando o preço sobe, a quantidade demandada cai). Porém, por se tratar de uma base pública com algumas distorções nos dados, os clusters **2** e **3** apresentaram valores positivos. Por esse motivo, utilizarei apenas os clusters **0** e **1** para a automação final.


## Automação: Simulador de Faturamento

Com as elasticidades calculadas, a base de dados original de **4 milhões** de registros foi previamente preparada, mapeando o cluster correspondente para cada combinação de Loja-Produto.

Para simular o impacto financeiro de decisões de preço, foi desenvolvida a função calculadora_faturamento(). Ao executar o código, o sistema solicitará os seguintes parâmetros de forma interativa:

* **Cluster:** Escolha entre o cluster **0** ou **1**.

* **Preço:** O valor atual do produto.

* **Quantidade:** A quantidade vendida do item.

* **Aumento:** A porcentagem de aumento proposta em formato decimal (ex: **0.10** para **10%** de aumento).

**Código da Calculadora:**
```python
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
```

## Proximos Passos e Avanços do Projeto

Embora a versão atual já automatize o motor de cálculo, meu objetivo é futuramente transformar este estudo em uma ferramenta para as equipes de negócio. Para isso, os próximos passos de desenvolvimento focarão em:

* **Automatização do Pipeline:** Habilitar o usuário a realizar apenas o upload do arquivo .csv bruto, com o programa já executando a custerização e a regressão de forma automática, habilitando pessoas fora do conhecimento de dados e python, trabalhar com a ferramenta

* **Interface Interativa com Streamlit:** Desenvolver um aplicativo web interativo utilizando a biblioteca Streamlit. Isso permitirá que gerentes e analistas de pricing utilizem barras deslizantes (sliders) para simular aumentos de preço facilmente, sem necessidade de interagir com o código.

* **Recomendação de Preço Ótimo e Ponto de Equilíbrio:** Com disponibilidade de uma variáveis contendo o custo do produto fixo e váriavel, é possivel calcular o **ponto de equilíbrio**. Isso possibilitará que o modelo não apenas simule o faturamento, mas entregue uma recomendação do melhor preço possível para maximizar as margens e a lucratividade de cada item.

