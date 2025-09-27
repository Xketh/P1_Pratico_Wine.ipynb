# **📘 Projeto Prático – Mineração de Dados (Wine Dataset)**

**Disciplina:** ISW-039 – Mineração de Dados - FATEC - Mauá   
**Professor:** Lucas B. Henrique  
**Data:** 27/09/2025

<br>

# 🥇 **Objetivo**

Esse projeto é a parte prática da P1. A ideia é pegar o dataset Wine (vinhos com 13 atributos químicos) e aplicar o que aprendemos:

* **Limpeza de dados**
* **Estatística descritiva**
* **PCA**
* **Balanceamento**
* **Classificação com k-NN**
* **Naive Bayes**
* **Métricas de avaliação**
  
O objetivo é **entender os dados** e **como criar modelos de classificação simples**.

<br>

# 🚀 **Como rodar**

**1.**  Abra o notebook no **Google Colab** (mais fácil).

**2.** Execute célula por célula (seta “play") > (Você pode clicar em "executar tudo", porém, roda tudo de uma vez, ai o notebook cospe um monte de coisa de uma vez e você não entende nada. Rodando uma célula por vez, você consegue ver: “ah, aqui ele carregou os dados, aqui ele limpou, aqui fez o gráfico”. Pelo menos pra mim, assim fez mais sentido).

**3.**  Se preferir, pode rodar no **VSCode** ou **Jupyter Notebook** local.

**4.** Bibliotecas usadas: **pandas, numpy, matplotlib, scikit-learn**.

**5.**  Sempre usar **random_state=42** para manter resultados iguais.

  <br>

  # **📂 Estrutura**

* P1_Pratico_Kethilyn_1131392223013 _Wine.ipynb → Notebook com todas as etapas

* README.md → Esse arquivo com explicações

* (opcional) resultados/ → salvar imagens ou tabelas

<br>

# **📑 Explicando cada etapa:**


### **1) Setup**

* Aqui pegamos todas as “ferramentas” que vamos usar: manipular tabelas, gráficos, criar modelos, medir acertos, etc.

* Só precisa rodar a célula.

<br>

### **2) Carregar o dataset Wine**

* Não precisa criar nada, só rodar o código.

O dataset já vem no sklearn*:

***"from sklearn.datasets import load_wine"*** > Importa a função que carrega o dataset Wine.

### **Resumo em linguagem simples:**

* X_raw = as características químicas do vinho

* y_raw = tipo de vinho (classe)

* target_names = nomes dos tipos de vinho


* 13 colunas químicas: álcool, magnésio, fenóis…

* Rótulo = tipo de vinho (0, 1, 2)

<br>

* **Obs.:** Sklearn é uma biblioteca Python que ajuda a criar modelos de Machine Learning, preparar dados e avaliar resultados.

### **3) Dicionário de dados**

Explicamos o que descrevemos atributos químicos

**Exemplo:**

* álcool = quantidade de álcool
* proline = quantidade de prolina, etc.

<br>

### **4) Qualidade dos dados**

Nessa etapa, checamos se tem valores faltando ou linhas duplicadas.
Se tiver, temos que decidir se vamos apagar ou preencher.

 **Pergunta 1:** Existem valores ausentes?

 **Resposta:** Não

 <br>
 
 **Pergunta 2:** Há duplicatas relevantes?
 
 **Resposta:** Não tem linhas duplicados relevantes, ou seja os dados estão limpos e á estão pronto pra gente analisar.

<br>

### **5) Estatística descritiva**

Na estatística descritiva, a **média, mediana, mínimo, máximo** ajudam a ver se tem valores esquisitos (muito grandes ou muito pequenos).

**O código:**

X_raw.describe().T → mostra números importantes de cada coluna: média, mediana, mínimo, máximo e quanto os valores se espalham (desvio padrão).

X_raw.hist(...) → faz gráficos que mostram como os números estão distribuídos.

<br>

**Pergunta 1:** Compare média vs mediana e dispersão de pelo menos 2 variáveis. 

**Resposta:** Se você observar, vai ver que a média (soma tudo e divide pela quantidade (ex.: notas 7, 8, 9 → média = 8, porque 7+8+9= 24 / 3 = 8) e a mediana (que é o valor do meio quando você coloca em ordem (ex.: notas 7, 8, 9 → 8) do "álcool" ficaram quase iquais, então os valores estão equilibrados. Já no "ácido málico", a média ficou bem maior que a mediana, ou seja, tem valores que estão puxando pra cima, o que faz a gente pensar nos **outliers** que nada mais é que os "valores estranhos".

<br>
  
**Pergunta 2:** A escala diferente entre atributos muda sua interpretação?

**Resposta:** A escala diferente muda a interpretação sim. Um atributo com números maiores por exemplo, pode parecer mais importante, mas isso é só por causa da escala. Por isso é preciso padronizar os dados antes de comparar.

<br>

### **6) Correlação**

A Correlação mostra se duas variáveis mudam juntas.

Pode ser positiva (uma aumenta, a outra também aumenta) ou negativa (uma aumenta, a outra diminui).

Valores:

**+1 →** perfeita correlação positiva

**0 →** sem correlação

**-1 →** perfeita correlação negativa

No caso, proline e alcohol têm correlação alta, porque quando o álcool aumenta no vinho, o proline também tende a aumentar.

<br>

**Pergunta 1:** Aponte duas correlações altas (em módulo) e discuta impacto na modelagem.

**Resposta:** Achei uma correlação alta entre proline e alcohol. Isso quer dizer que essas variáveis andam juntas, tipo “cola”.

<br>

**Pergunta 2:** Discuta impacto na modelagem:

**Resposta:** O Impacto pode dar informação repetida, fazendo com que a modelagem fique mais pesada e demore mais.

<br>

### **7) Divisão treino/teste**

Separar dados em:

* treino (80%) → o modelo aprende

* teste (20%) → avaliamos se ele aprendeu bem

<br>

### **8) Pré-processamento**

Colunas estão em escalas diferentes.
A gente padroniza pra colocar tudo na mesma régua.

### **9) Baseline (Dummy)**

Modelo super bobo que só chuta a classe mais comum.
Serve de comparação: se k-NN ou Naive Bayes não forem melhores, deu ruim.

<br>

### **10) k-NN**

O k-NN é um modelo que olha quem são os vizinhos mais próximos de um ponto para decidir a classe dele.

**k =** número de vizinhos que o modelo vai considerar.

Ex.: k=5 → olha os 5 vizinhos mais próximos.

**weights = distance →** significa que os vizinhos mais próximos têm mais importância do que os mais longe.

**p = 2 (euclidiana) →** distância “reta” entre pontos

* **GridSearchCV:** Ferramenta do scikit-learn que ajuda a achar os melhores parâmetros para um modelo.



Treinar modelo que classifica baseado nos vizinhos mais próximos.
Ajustamos parâmetros (número de vizinhos, distância, peso).

<br>

**Pergunta 1 :** Qual configuração de k, weights e p foi escolhida? Interprete o resultado considerando a escala dos atributos.

**Resposta:** O melhor foi **k=5**, **weights=distance**, **p=2 (euclidiana** >. O modelo acabou dando mais peso pros que estavam mais perto.

 <br>
 
**Pergunta 2:** Interprete o resultado considerando a escala dos atributos:

**Resposta:** Cada atributo (álcool, magnésio e etc) tem números em escalas que são diferentes. Se a gente não padroniza os atributos, o k-NN fica muito confuso.


<br>

### **11) Naive Bayes**

* **Probabilístico >** Ele não só olha os dados, mas calcula a probabilidade de algo ser de uma classe ou outra.
* Ele assume que cada coluna não depende da outra.
* Rápido e eficiente, mas menos preciso que k-NN nesse dataset, justamente porque ele assume que uma característica química não depende da outra, e não percebe que algumas andam juntas. O k-NN verifica isso.
* Ele funciona muito bem para texto e e-mails (tipo SPAM)
  
* **GaussianNB** é uma versão de Naive Bayes que assume que os atributos numéricos seguem a distribuição normal (sino).

* No Wine, todos os 13 atributos químicos são contínuos, então GaussianNB é apropriado.
  
<br>

### **12) PCA**

* Ela reduz a quantidade de colunas mas pode perder detalhes importantes
Ex.: transforma 13 colunas em 2 componentes principais pra gente visualizar

**Pergunta 1:** Duas PCs são suficientes para visualização?

**Resposta:** Duas ja ajudam a visualizar os grupos de vinhos no gráfico, só que para modelagem não é muito suficiente porque não guardam toda informação.

<br>

**Pergunta 2:** Usaria PCA para modelagem?

**Resposta:** Não sei ao certo se usaria, porque ela reduz muita informação que pode ser importante. 

<br>

### **13) Balanceamento**

* O Wine não é desbalanceado.
* O **SMOTE** é usado quando uma classe tem poucos exemplos.

**Pergunta:** Em que tipo de problema aplicaríamos SMOTE ou undersampling? Dê um exemplo.

**Resposta:** Usamos o SMOTE é quando uma classe tem muito mais exemplos que a outra. Por exemplo: "Se 3% dos clientes cancelaram conta no banco, logo 97% não cancelaram. Mas não da pra ficar "chutando" que não cancelaram. No caso, como a classe 3% é muito pequena. Usamos o SMOTE. Porque ele cria clientes parecidos com os que cancelaram o que aumenta a quantidade de clientes que cancelaram, o que equilibra. Assim o sistema vai identificar melhor os cancelamnetos e não se deixar enganar com os clientes que não cancelaram.
Já o Undersampling, nesse exemplo, reduziria os clientes que não cancelaram, mas isso perde informação, então SMOTE é mais indicado.

<br>

### **14) Comparação de modelos**

* Aqui Juntamos métricas (**accuracy, precision, recall, F1**) de **Dummy, k-NN e Naive Bayes**.
* Escolhemos qual modelo mandou melhor.

**Pergunta:** Qual modelo você recomendaria **neste dataset** e **por quê**? Considere desempenho, simplicidade e sensibilidade à escala.

**Resposta:** Eu recomendaria k-NN, porque o desempenho é muito bom, é simples de entender e funciona bem quando os atributos são padronizados.


<br>

# **Conclusão**

## **Os dados mostraram (EDA):** 

Os dados do vinho possuem 13 características químicas e mostraram que não tinha nenhum dado faltando, mas algumas colunas tinham números bem diferentes entre si, por exemplo, o valor do álcool que é aproximadamente 10 e o valor do proline aproximadamente 1000.


<br>

## **O que foi feito no pré-processamento:**

* Padronizamos todos os números pra ficarem na mesma escala.
* Também vimos se tinha outliers ou informações repetidas.

<br>

## **Qual modelo foi melhor:**

O k-NN foi o que se saiu melhor com k=5 e distância euclidiana, porque conseguiu separar melhor os tipos de vinho. O Naive Bayes também funcionou, mas não tão bem.

<br>

## **Limitações e próximos passos:**

O dataset é pequeno, então o modelo pode não funcionar igual com outros vinhos. Daria também para testar com mais dados ou tentar outras técnicas de redução de variáveis, tipo PCA, só que sem perder informações importante.
