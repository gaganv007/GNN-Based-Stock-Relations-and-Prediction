# train.py
import torch, torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score
import config, os
from models import GCNNet

def prepare_dataset(feats, targets, G, dates):
    # align all tickers to same date indices
    # for static graph, pick intersection
    common_idx = dates[config.STOCKS[0]]
    for t in config.STOCKS[1:]:
        common_idx = common_idx.intersection(dates[t])
    # take features and targets at those dates
    X = torch.tensor(
        np.stack([feats[t] for t in config.STOCKS], axis=1)[common_idx],
        dtype=torch.float
    )  # (T, N, F)
    Y = torch.tensor(
        np.stack([targets[t] for t in config.STOCKS],axis=1)[common_idx],
        dtype=torch.long
    )  # (T, N)
    # split train/test along time
    split = (common_idx >= pd.to_datetime(config.SPLIT_DATE))
    train_X, train_Y = X[~split], Y[~split]
    test_X,  test_Y  = X[split],  Y[split]
    # build one long sequence per set by concatenating time steps
    # here we treat each (time,graph) as separate sample
    def to_data(x, y):
        data_list = []
        for t in range(x.shape[0]):
            data_list.append(Data(
                x = x[t],
                edge_index = torch.tensor(list(G.edges),dtype=torch.long).t().contiguous(),
                y = y[t]
            ))
        return data_list
    return to_data(train_X,train_Y), to_data(test_X,test_Y)

def train_and_eval(feats, targets, G, dates):
    from torch_geometric.loader import DataLoader
    train_ds, test_ds = prepare_dataset(feats, targets, G, dates)
    loader_tr = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    loader_te = DataLoader(test_ds,  batch_size=config.BATCH_SIZE)
    model = GCNNet(
        in_dim=train_ds[0].x.shape[1],
        hidden_dim=config.HIDDEN_DIM,
        out_dim=2,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT
    ).to(config.DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=config.LR)
    for epoch in range(1, config.EPOCHS+1):
        model.train()
        losses = []
        for d in loader_tr:
            d = d.to(config.DEVICE)
            opt.zero_grad()
            logits = model(d.x, d.edge_index)
            loss = F.cross_entropy(logits, d.y)
            loss.backward(); opt.step()
            losses.append(loss.item())
        if epoch%10==0:
            print(f"[Epoch {epoch:02d}] train loss={sum(losses)/len(losses):.4f}")
    # eval
    model.eval(); preds=[]; gts=[]
    with torch.no_grad():
        for d in loader_te:
            d = d.to(config.DEVICE)
            logits = model(d.x, d.edge_index)
            preds += logits.argmax(1).cpu().tolist()
            gts   += d.y.cpu().tolist()
    acc = accuracy_score(gts, preds)
    print(f"▶️ Test Accuracy: {acc:.3%}")
    # save
    torch.save(model.state_dict(), os.path.join(config.MODELS_DIR, "gcn.pt"))
