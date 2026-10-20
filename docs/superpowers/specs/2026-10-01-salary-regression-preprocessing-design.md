
# Design: preprocessing para regressão de salários

## Objetivo

Preparar `archive/ai_jobs_salaries_clean.csv` para uma rede neural manual que
prevê o valor contínuo de `salary_in_usd`. O processamento deve remover
variáveis inadequadas ou redundantes, preservar informações úteis para o
salário e reduzir o efeito das classes categóricas raras sem contaminar o
conjunto de teste.

## Decisões aprovadas

- Alvo: `salary_in_usd` como regressão, sem transformar o alvo em classes.
- Moeda: usar o `salary_in_usd` já existente como valor confiável; não
  recalcular taxas de câmbio.
- Balanceamento: oversampling com reposição, aplicado somente ao treino.
- Categoria a balancear: `work_mode`.
- Níveis de experiência: balancear `experience_level`.
- Família de cargo: balancear `role_family`.
- Saída principal: preencher `DataProcessing.ipynb` e salvar CSVs processados.
- Divisão: treino/teste com semente fixa e distribuição original preservada no
  teste.

## Colunas

### Entradas mantidas

- `work_year`
- `experience_level`
- `job_title`
- `remote_ratio`
- `work_mode`
- `role_family`

### Alvo

- `salary_in_usd`

### Colunas removidas

- `salary`
- `salary_currency`
- `employee_residence`
- `company_location`
- `company_size`
- `employment_type`
- `employment_type_label`
- `experience_level_label`
- `salary_outlier_flag`
- `isco_group_hint`

`salary` e `salary_currency` são removidas porque o alvo em USD já está
disponível. `salary_outlier_flag` é derivada do salário e poderia causar
vazamento. Os rótulos textuais duplicados são removidos em favor das colunas
codificadas. Localização, tamanho e tipo de emprego seguem a solicitação de
limpeza.

## Fluxo de dados

1. Carregar o CSV e validar a presença das colunas obrigatórias.
2. Validar tipos, valores ausentes do alvo e valores infinitos; falhar com uma
   mensagem explícita quando o alvo não for utilizável.
3. Remover duplicatas exatas.
4. Separar `X` e `y`, mantendo `salary_in_usd` fora das entradas.
5. Dividir os dados em treino e teste com `random_state=42`.
6. Ajustar imputação, codificação one-hot e normalização usando somente o
   treino.
7. Aplicar oversampling controlado no treino para `work_mode`,
   `experience_level` e `role_family`. O método deve limitar a expansão para
   evitar replicação excessiva da classe majoritária `Other / Unclassified`.
8. Salvar `train_processed.csv` e `test_processed.csv` com as mesmas colunas,
   além de um relatório de distribuição antes e depois do balanceamento.

O teste não será reamostrado, normalizado com estatísticas próprias nem usado
para ajustar vocabulário categórico. Isso evita vazamento de informação.

## Codificação e escala

Variáveis categóricas serão convertidas por one-hot encoding. Variáveis
numéricas serão imputadas, quando necessário, pela mediana do treino. A escala
min-max será ajustada no treino; as mesmas estatísticas serão aplicadas ao
teste. Colunas constantes terão denominador substituído por `1.0` para evitar
divisão por zero. O resultado final não poderá conter `NaN` ou `inf`.

## Balanceamento

O balanceamento será aproximado e controlado, não uma equalização ingênua até a
maior contagem. Para cada coluna selecionada, categorias raras serão
reamostradas com reposição até um limite configurado, mantendo a classe
majoritária sob controle. As contagens de cada coluna serão medidas antes e
depois, e o limite será documentado no relatório.

Como o objetivo é regressão, nenhuma amostra será criada alterando
artificialmente `salary_in_usd`; uma linha reamostrada conserva todas as suas
características e o respectivo alvo.

## Artefatos e validação

O notebook produzirá:

- `archive/train_processed.csv`
- `archive/test_processed.csv`
- `archive/preprocessing_report.json`

O relatório registrará tamanho dos conjuntos, colunas mantidas/removidas,
limiar de oversampling, contagens categóricas antes/depois, semente usada e
estatísticas de escala. As verificações mínimas serão:

- `salary_in_usd` presente no alvo e ausente de `X`;
- mesmas colunas de entrada no treino e no teste;
- nenhum valor ausente ou infinito;
- nenhuma coluna removida no resultado;
- teste sem duplicação introduzida pelo balanceamento;
- transformações ajustadas somente no treino.

## Escopo

Esta especificação cobre o pré-processamento e os artefatos CSV. Não altera a
arquitetura, a função de perda ou o algoritmo de treinamento da MLP manual.
