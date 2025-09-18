import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# Load dataset
digits = load_digits()
X = digits.data.astype(np.float32)
y = digits.target.astype(np.int64)

scaler = StandardScaler()
X = scaler.fit_transform(X).astype(np.float32)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_t = torch.from_numpy(X_train)
y_train_t = torch.from_numpy(y_train)
X_val_t   = torch.from_numpy(X_val)
y_val_t   = torch.from_numpy(y_val)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=256, shuffle=False)

class MLP(nn.Module):
    def __init__(self, input_dim=64, hidden=128, num_classes=10, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, num_classes),
        )
    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

def train_epoch():
    model.train()
    running_loss, running_correct, total = 0.0, 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=1)
        running_correct += (preds == yb).sum().item()
        total += xb.size(0)
    return running_loss/total, running_correct/total

@torch.no_grad()
def validate():
    model.eval()
    running_loss, running_correct, total = 0.0, 0, 0
    for xb, yb in val_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        running_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=1)
        running_correct += (preds == yb).sum().item()
        total += xb.size(0)
    return running_loss/total, running_correct/total

epochs = 20
best_val_acc = 0.0
for epoch in range(1, epochs+1):
    train_loss, train_acc = train_epoch()
    val_loss, val_acc = validate()
    scheduler.step()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_mlp_digits.pt")
    print(f"epoch {epoch:02d} | "
          f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
          f"val loss {val_loss:.4f} acc {val_acc:.3f}")

model.load_state_dict(torch.load("best_mlp_digits.pt", map_location=device))
model.eval()
with torch.no_grad():
    logits = model(X_val_t.to(device))
    preds = logits.argmax(dim=1).cpu().numpy()

print("\nvalidation accuracy:", accuracy_score(y_val, preds))
print("\nclassification report:\n", classification_report(y_val, preds))
