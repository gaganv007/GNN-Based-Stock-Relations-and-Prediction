import matplotlib.pyplot as plt
import numpy as np
from train import load_gcn, prepare_dataset
from sklearn.metrics import confusion_matrix, classification_report
from config import STOCKS, SPLIT_DATE

def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4,4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xticks([0,1],["Down","Up"])
    plt.yticks([0,1],["Down","Up"])
    for i in (0,1):
        for j in (0,1):
            plt.text(j,i,cm[i,j],ha="center",va="center",color="white")
    plt.ylabel("True"); plt.xlabel("Pred")
    plt.show()

def full_evaluate(price_df, feature_dict, G):
    # split train/test by date
    train_feats = {t:df[df.index < SPLIT_DATE] for t,df in feature_dict.items()}
    test_feats  = {t:df[df.index >= SPLIT_DATE] for t,df in feature_dict.items()}

    data_tr = prepare_dataset(train_feats, G)
    data_te = prepare_dataset(test_feats,  G)
    in_dim, out_dim = data_tr.x.shape[1], 2

    print("▶︎ Training GCN …")
    from train import train_gcn
    best_acc = train_gcn(data_tr, data_te, in_dim, out_dim)
    print(f"✅ Best validation accuracy: {best_acc:.4f}")

    print("▶︎ Testing final model …")
    model = load_gcn(in_dim, out_dim)
    with torch.no_grad():
        pred = model(data_te.x, data_te.edge_index).argmax(dim=1)
    y_true = data_te.y.cpu().numpy()
    y_pred = pred.cpu().numpy()
    print("\n" + classification_report(y_true, y_pred, target_names=["Down","Up"]))
    plot_confusion(y_true, y_pred)
