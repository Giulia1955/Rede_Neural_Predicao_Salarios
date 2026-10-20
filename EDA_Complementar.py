# basicamente faz uma analise complementar para a main
# conta as linhas duplicadas
# curvas de perda a cada época
# matriz de confusão e métricas de melhor configuração
# linhas de base com classe majoritária e refressão logística -> a mlp sem camada oculta. Pensei em tirar a camada oculta para saber se elas realmente ajudam
# variação da acurácia
# seleção da config por conj de validação

import contextlib
import io

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import Train as base

BEST_CONFIG = (3, 16, 0.1) # (camadas ocultas, neurônios, taxa)
CURVE_CONFIGS = [(1, 4, 0.01), (1, 4, 0.1), (2, 16, 0.1), (3, 16, 0.1), (3, 16, 0.5)]
SPLIT_SEEDS = [42, 43, 44, 45, 46]
REPS = 3# repetições 
VAL_FRACTION = 0.10 # fração do TREINO usada como validação


# func auxiliares
def quiet_load():
    #func para chamar o pre processamente sem tem muito output no console
    with contextlib.redirect_stdout(io.StringIO()):
        return base.load_and_preprocess()


def train(data, layers, neurons, lr, seed):
    # treina uma MLP no conj de treino
    X_train, _, y_train, _ = data
    model = base.MLP(X_train.shape[1], layers, neurons, lr, base.LOSS, seed)
    history = model.fit(X_train, y_train, base.EPOCHS, base.BATCH_SIZE, seed, verbose=False)
    return model, history


def confusion_metrics(y_true, y_pred):
    # matriz de confusão e métricas
    y_true, y_pred = y_true.ravel(), y_pred.ravel()
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    acc = (tp + tn) / len(y_true)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "acuracia": acc, "precisao": prec, "revocacao": rec, "f1": f1}


# testa a acurácia
def test_accuracy(model, data):
    _, X_test, _, y_test = data
    return 100.0 * float((model.predict(X_test) == y_test).mean())


# ve o num de linhas duplicadas
def check_duplicates():
    print("\n=== 1. Linhas duplicadas ===")
    df = pd.read_csv(base.CSV_PATH)
    cols = [c for c in df.columns if c not in base.DROP_COLUMNS]
    n_dup = int(df.duplicated(subset=cols).sum())
    print(f"Linhas totais: {len(df)}")
    print(f"Duplicadas (mesmos atributos usados + mesmo salário): {n_dup} ({100 * n_dup / len(df):.2f}%)")
    print(f"Linhas únicas nesses atributos: {len(df) - n_dup}")


#curvas de perda
def loss_curves(data):
    print("\n=== 2. Curvas de perda ===")
    curves = {}
    for (layers, neurons, lr) in CURVE_CONFIGS:
        _, hist = train(data, layers, neurons, lr, base.SEED)
        label = f"{layers} cam., {neurons} neur., lr={lr}"
        curves[label] = hist
        print(f"{label}: perda inicial {hist[0]:.4f} -> final {hist[-1]:.4f}")

    pd.DataFrame(curves).rename_axis("epoca").to_csv("curvas_perda.csv")
    plt.figure(figsize=(7, 4))
    for label, hist in curves.items():
        plt.plot(range(1, len(hist) + 1), hist, label=label)
    plt.xlabel("Época")
    plt.ylabel(f"Perda no treino ({base.LOSS.upper()})")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("curvas_perda.png", dpi=200)
    plt.close()
    print("Salvo: curvas_perda.csv, curvas_perda.png")


# matriz de confusão e métricas
def best_config_metrics(data):
    print(f"\n=== 3. Métricas da configuração {BEST_CONFIG} ===")
    model, _ = train(data, *BEST_CONFIG, base.SEED)
    _, X_test, _, y_test = data
    m = confusion_metrics(y_test, model.predict(X_test))
    print(f"Matriz de confusão (teste): TN={m['TN']} FP={m['FP']} | FN={m['FN']} TP={m['TP']}")
    print(f"Acurácia {100 * m['acuracia']:.2f}% | Precisão {100 * m['precisao']:.2f}% | "
          f"Revocação {100 * m['revocacao']:.2f}% | F1 {100 * m['f1']:.2f}%")
    pd.DataFrame([m]).to_csv("metricas_melhor.csv", index=False)
    print("Salvo: metricas_melhor.csv")

# linhas de base
def baselines(data):
    print("\n=== 4. Linhas de base (acurácia no teste) ===")
    _, X_test, y_train, y_test = data[0], data[1], data[2], data[3]
    majority = int(y_train.mean() >= 0.5)
    acc_major = 100.0 * float((y_test == majority).mean())
    print(f"Classe majoritária (sempre {majority}): {acc_major:.2f}%")

    _, _, lr = BEST_CONFIG
    accs = [test_accuracy(train(data, 0, 1, lr, base.SEED + r)[0], data) for r in range(REPS)]
    print(f"Regressão logística (MLP sem camada oculta, lr={lr}): "
          f"{np.mean(accs):.2f}% (desvio {np.std(accs):.2f}, {REPS} repetições)")

    accs_best = [test_accuracy(train(data, *BEST_CONFIG, base.SEED + r)[0], data) for r in range(REPS)]
    print(f"MLP {BEST_CONFIG}: {np.mean(accs_best):.2f}% (desvio {np.std(accs_best):.2f})")


# variação da divisão de treino e teste
def split_variation():
    print(f"\n=== 5. Variação da divisão (config {BEST_CONFIG}) ===")
    original_seed = base.SEED
    accs = []
    try:
        for s in SPLIT_SEEDS:
            base.SEED = s # muda a divisao. load_and_preprocess le base.SEED
            data = quiet_load()
            model, _ = train(data, *BEST_CONFIG, s)
            accs.append(test_accuracy(model, data))
            print(f"  divisão seed={s}: {accs[-1]:.2f}%")
    finally:
        base.SEED = original_seed
    print(f"Média {np.mean(accs):.2f}% | desvio {np.std(accs):.2f} | "
          f"mín {np.min(accs):.2f}% | máx {np.max(accs):.2f}%")


# selecao por validacao
def selection_by_validation(data):
    print("\n=== 6. Seleção de configuração por validação ===")
    X_train, X_test, y_train, y_test = data
    rng = np.random.default_rng(base.SEED)
    idx = rng.permutation(len(X_train))
    n_val = int(VAL_FRACTION * len(idx))
    val_idx, sub_idx = idx[:n_val], idx[n_val:]
    sub = (X_train[sub_idx], None, y_train[sub_idx], None)

    rows = []
    for (layers, neurons, lr) in base.GRID:
        model, _ = train(sub, layers, neurons, lr, base.SEED)
        val_acc = 100.0 * float((model.predict(X_train[val_idx]) == y_train[val_idx]).mean())
        rows.append({"camadas": layers, "neuronios": neurons, "taxa": lr, "acc_validacao": round(val_acc, 2)})
        print(f"  {layers} cam., {neurons} neur., lr={lr}: validação {val_acc:.2f}%")

    best = max(rows, key=lambda r: r["acc_validacao"])
    print(f"Escolhida pela validação: {best['camadas']} cam., {best['neuronios']} neur., lr={best['taxa']}")
    model, _ = train(data, best["camadas"], best["neuronios"], best["taxa"], base.SEED)
    print(f"Acurácia no teste da escolhida (retreinada no treino completo): {test_accuracy(model, data):.2f}%")
    pd.DataFrame(rows).to_csv("selecao_validacao.csv", index=False)
    print("Salvo: selecao_validacao.csv")


def main():
    check_duplicates()
    data = quiet_load()
    loss_curves(data)
    best_config_metrics(data)
    baselines(data)
    split_variation()
    selection_by_validation(data)


if __name__ == "__main__":
    main()