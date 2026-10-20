# dado o perfil de um emprego, o salário dele fica acima ou abaixo da mediana do dataset?
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split # o sklearn está servindo apenas para dividir os dados

CSV_PATH = "ai_jobs_salaries_clean.csv"

# coluna usada para criar o alvo binário, nulo usa deteccao automática
TARGET_COLUMN = "salary_in_usd"

# 1 se o valor >= mediana, senão 0, TARGET_THRESHOLD para definir um limiar fixo
TARGET_THRESHOLD = None
DROP_COLUMNS = [
    "salary",                  # mesmo salário em moeda local (vazamento do alvo)
    "salary_outlier_flag",     # derivada do salário (vazamento do alvo)
    "salary_currency",         # redundante com país de residência/empresa
    "experience_level_label",  # duplica experience_level
    "employment_type_label",   # duplica employment_type
    "work_mode",               # duplica remote_ratio
    "role_family",             # agrupamento de job_title
    "isco_group_hint",         # derivada do cargo, com valores vazios
]
 # colunas para n usar
MAX_CATEGORIES = 500 # colunas com mais que esse valor são descartadas

TEST_SIZE = 0.20 # 20% para teste e 80% para treino
SEED = 42
NUM_HIDDEN_LAYERS = 2 #num de camadas intermediarias
NEURONS_PER_LAYER = 4
LEARNING_RATE = 0.1
EPOCHS = 100 # não sei se é uma boa quantidade, talvez seja muito
LOSS = "bce"# "bce" (Log Loss) ou "mse" (Erro Quadrático Médio)
BATCH_SIZE = 256

# grade de experimentos para o artigo
RUN_GRID = True
GRID = [
    (0, 1, 0.1), # para tirar a camada oculta e fazer regressão logistica
    (1, 4, 0.01),
    (1, 4, 0.1),
    (1, 8, 0.1),
    (1, 16, 0.1),
    (2, 8, 0.1),
    (2, 16, 0.1),
    (2, 16, 0.5),
    (3, 16, 0.1),
    (3, 16, 0.5),
]

#repete com semestes diferentes
GRID_REPETITIONS = 3

# -------------- pre processamento

def find_target_column(df):
    #Escolhe a coluna numerica de salario caso TARGET_COLUMN não tenha sido definida
    #não sei se é uma boa prática, mas pensei nisso para tratar falta de informação do usuário
    if TARGET_COLUMN is not None:
        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"TARGET_COLUMN '{TARGET_COLUMN}' não existe. Colunas: {list(df.columns)}")
        return TARGET_COLUMN
    numeric = list(df.select_dtypes(include=np.number).columns)
    for col in numeric:  # para pegar a coluna em dolar
        if "usd" in col.lower():
            return col
    for col in numeric:
        if "salary" in col.lower() or "salario" in col.lower() or "salário" in col.lower():
            return col
    raise ValueError(f"Defina TARGET_COLUMN. Colunas: {list(df.columns)}")  # caso a coluna não seja encontrada automaticamente
 
# carrega o csv, cria o alvo binário, divice e normaliza
def load_and_preprocess():
    df = pd.read_csv(CSV_PATH)
    #debug
    print(f"Dataset carregado: {df.shape[0]} linhas x {df.shape[1]} colunas")
    print(f"Colunas: {list(df.columns)}")
 
    # resposta q a rede tenta acertar (0 ou 1, um para salário alto)
    target_col = find_target_column(df)
    threshold = TARGET_THRESHOLD if TARGET_THRESHOLD is not None else df[target_col].median()
    y = (df[target_col] >= threshold).astype(int).to_numpy().reshape(-1, 1) # define o ponto de corte
    print(f"Alvo: '{target_col}' >= {threshold:,.2f}  ->  1 | caso contrário 0  "
          f"(proporção de 1s: {y.mean():.2%})")
 
    # remove a coluna alvo e as colunas que são inúteis
    X_df = df.drop(columns=[target_col] + [c for c in DROP_COLUMNS if c in df.columns])
 
    # descarta colunas com muitas categorias -> colunas não numéricas, a MLP só aceita números
    for col in X_df.select_dtypes(exclude=np.number).columns:
        if X_df[col].nunique() > MAX_CATEGORIES:
            print(f"Aviso: coluna '{col}' descartada ({X_df[col].nunique()} categorias)")
            X_df = X_df.drop(columns=[col])
 
    # caso a base de dados tenha um NaN, esse NaN vai se espalhar na mult de matrizes. Para dados numericos eu usei a mediana da coluna e texto é colocado Unknown
    for col in X_df.columns:
        if pd.api.types.is_numeric_dtype(X_df[col]):
            X_df[col] = X_df[col].fillna(X_df[col].median())
        else:
            X_df[col] = X_df[col].fillna("Unknown")
 
    # cada categoria vira uma coluna própria de 0 ou 1, então se tem a propriedade vai ter 1, senão 0
    X_df = pd.get_dummies(X_df, dtype=float)
    print(f"num de atributos de entrada após one hot: {X_df.shape[1]}")
    X = X_df.to_numpy(dtype=float)
 
    # divide o dataset para treino e teste, essa divisão é aleatória
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, shuffle=True
    )
 
    # normaliza min-max. E calculado so no treino para não ir info do teste e n prejudicar o resultado
    x_min = X_train.min(axis=0)
    x_max = X_train.max(axis=0)
    denom = np.where((x_max - x_min) == 0, 1.0, x_max - x_min)# para n ter div por 0
    X_train = (X_train - x_min) / denom # coloca a coluna no intervalo 0 e 1
    X_test = np.clip((X_test - x_min) / denom, 0.0, 1.0) # o clip força os valores entre 0 e 1
 
    print(f"Treino: {len(X_train)} amostras | Teste: {len(X_test)} amostras\n") # mostra a quantidade de dados para treino e teste -> apenas por bonito
    return X_train, X_test, y_train, y_test

# ------------- mlp em si

def sigmoid(z):
    # sigmoide, usei o clip para colocar no range
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
 
 
def sigmoid_derivative_from_activation(a):
    # ativa a sigmoide
    return a * (1.0 - a)
 
 
class MLP:
    def __init__(self, n_inputs, n_hidden_layers, n_neurons, learning_rate, loss="bce", seed=0):
        self.lr = learning_rate
        self.loss_name = loss
        rng = np.random.default_rng(seed)
 
        self.sizes = [n_inputs] + [n_neurons] * n_hidden_layers + [1]
 
        # inicialização para sigmoid
        self.W, self.b = [], []
        for n_in, n_out in zip(self.sizes[:-1], self.sizes[1:]):
            limit = np.sqrt(6.0 / (n_in + n_out))
            self.W.append(rng.uniform(-limit, limit, size=(n_in, n_out)))
            self.b.append(np.zeros((1, n_out)))
 
    # propaga x pela rede e retorna o número de ativação de cada camada
    def forward(self, X):
        activations = [X]
        for W, b in zip(self.W, self.b):
            z = activations[-1] @ W + b
            activations.append(sigmoid(z))
        return activations
 
    def predict_proba(self, X):
        return self.forward(X)[-1]
 
    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
 
    # calcula o erro
    def compute_loss(self, y_true, y_pred):
        if self.loss_name == "bce":
            eps = 1e-12
            p = np.clip(y_pred, eps, 1 - eps)
            return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))
        return float(np.mean((y_true - y_pred) ** 2))
 
    #backward pass -> retorna os gradientes dW e db de cada camada, calculados de trás para frente
    def backward(self, activations, y_true):
        m = y_true.shape[0]
        n_layers = len(self.W)
        dW = [None] * n_layers
        db = [None] * n_layers
 
        a_out = activations[-1]
        if self.loss_name == "bce":
            delta = a_out - y_true
        else:
            delta = (a_out - y_true) * sigmoid_derivative_from_activation(a_out)
 
        for l in range(n_layers - 1, -1, -1):
            dW[l] = activations[l].T @ delta / m
            db[l] = delta.sum(axis=0, keepdims=True) / m
            if l > 0:  #propaga o erro para a camada anterior
                delta = (delta @ self.W[l].T) * sigmoid_derivative_from_activation(activations[l])
        return dW, db
 
    #func para atualizar os pesos da matriz
    def update(self, dW, db):
        for l in range(len(self.W)):
            self.W[l] -= self.lr * dW[l]
            self.b[l] -= self.lr * db[l]
 
    #loop de treinamento da mlp
    def fit(self, X, y, epochs, batch_size=None, seed=0, verbose=True):
        rng = np.random.default_rng(seed)
        n = X.shape[0]
        history = []
        for epoch in range(1, epochs + 1):
            if batch_size is None:
                batches = [np.arange(n)]
            else:
                idx = rng.permutation(n)
                batches = [idx[i:i + batch_size] for i in range(0, n, batch_size)]
 
            for b_idx in batches:
                acts = self.forward(X[b_idx])
                dW, db = self.backward(acts, y[b_idx])
                self.update(dW, db)
 
            loss = self.compute_loss(y, self.predict_proba(X))
            history.append(loss)
            if verbose and (epoch == 1 or epoch % max(1, epochs // 10) == 0):
                print(f"  Época {epoch:5d}/{epochs} | erro ({self.loss_name}): {loss:.6f}")
        return history


#retorna os acertos totais
def evaluate(model, X, y):
    hits = int((model.predict(X) == y).sum())
    total = len(y)
    return hits, total, 100.0 * hits / total
 
# roda a mlp
def run_experiment(data, n_hidden_layers, n_neurons, learning_rate, epochs=EPOCHS, loss=LOSS, batch_size=BATCH_SIZE, seed=SEED, verbose=True):
    X_train, X_test, y_train, y_test = data
    model = MLP(X_train.shape[1], n_hidden_layers, n_neurons, learning_rate, loss, seed)
    if verbose:
        print(f"Arquitetura: {model.sizes} | taxa={learning_rate} | épocas={epochs} | loss={loss}")
    model.fit(X_train, y_train, epochs, batch_size, seed, verbose)
 
    tr_hits, tr_total, tr_acc = evaluate(model, X_train, y_train)
    te_hits, te_total, te_acc = evaluate(model, X_test, y_test)
    if verbose:
        print(f"\nTreino -> Acertos: {tr_hits} de {tr_total} ({tr_acc:.2f}%)")
        print(f"TESTE  -> Acertos: {te_hits} de {te_total} ({te_acc:.2f}%)\n")
    return {"train_acc": tr_acc, "test_acc": te_acc, "test_hits": te_hits, "test_total": te_total}
 
def run_grid(data):
    #imprime em forma de tabela para facilitar para o relatório
    rows = []
    for (layers, neurons, lr) in GRID:
        results = [
            run_experiment(data, layers, neurons, lr, seed=SEED + r, verbose=False)
            for r in range(GRID_REPETITIONS)
        ]
        rows.append({
            "Camadas ocultas": layers,
            "Neurônios/camada": neurons,
            "Taxa de aprendizado": lr,
            "Épocas": EPOCHS,
            "Acurácia treino (%)": round(np.mean([r["train_acc"] for r in results]), 2),
            "Acurácia teste (%)": round(np.mean([r["test_acc"] for r in results]), 2),
            "Desvio teste (%)": round(np.std([r["test_acc"] for r in results]), 2),
            "Acertos (última rep.)": f'{results[-1]["test_hits"]} de {results[-1]["test_total"]}',
        })
        print(f"OK: {layers} camada(s), {neurons} neurônios, lr={lr} -> "
              f"teste {rows[-1]['Acurácia teste (%)']}%")
    table = pd.DataFrame(rows)
    print("\n" + table.to_string(index=False))
    table.to_csv("resultados_experimentos.csv", index=False)
    print("\nTabela salva em 'resultados_experimentos.csv'")

def main():
    data = load_and_preprocess()
 
    run_experiment(data, NUM_HIDDEN_LAYERS, NEURONS_PER_LAYER, LEARNING_RATE)
 
    if RUN_GRID:
        # imprime a tabela
        print("=== Grade de experimentos ===")
        run_grid(data)
 
if __name__ == "__main__":
    main()