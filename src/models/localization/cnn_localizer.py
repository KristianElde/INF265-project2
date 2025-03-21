import torch
from torch.utils.data import DataLoader


class CNNLocalizer:
    def __init__(
        self,
        loss_fn,
        network,
        learning_rate=0.001,
        max_epochs=10,
        weight_decay=0,
        momentum=0.9,
    ):
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.network = network
        print(f"running on: {str(self.device)}")

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        delta: int = 0.01,
        patience: int = 3,
    ):
        self.model = self.network().to(self.device)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(self.momentum, 0.999),
        )

        patience_counter = 0
        best_val_loss = float("inf")
        training_losses, val_losses = [], []

        for i in range(self.max_epochs):
            train_epoch_loss, val_epoch_loss = 0, 0
            num_train_batches, num_val_batches = 0, 0

            self.model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                outputs = self.model(X_batch)
                loss = torch.mean(self.loss_fn(outputs, y_batch))
                train_epoch_loss += loss.item()
                num_train_batches += 1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.model.eval()
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                with torch.no_grad():
                    outputs = self.model(X_batch)
                loss = torch.mean(self.loss_fn(outputs, y_batch))
                val_epoch_loss += loss.item()
                num_val_batches += 1

            val_epoch_loss /= num_val_batches
            train_epoch_loss /= num_train_batches

            print(
                f"Epoch {i+1}/{self.max_epochs} — Training loss: {train_epoch_loss} — Val loss: {val_epoch_loss}"
            )

            training_losses.append(train_epoch_loss)
            val_losses.append(val_epoch_loss)

            if val_epoch_loss < best_val_loss - delta:
                best_val_loss = val_epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Stopped at epoch {i+1}")
                    break

        return training_losses, val_losses

    def predict(self, X):
        if not self.model:
            raise ValueError("fit must be called before calling predict")

        self.model.eval()
        X = X.to(self.device)

        with torch.no_grad():
            outputs = self.model(X)

        predicted_classes = torch.argmax(outputs[:, 5:], dim=1, keepdim=True)
        predicted_detection = torch.sigmoid(outputs[:, 0:1]) > 0.5
        preds = torch.cat(
            (predicted_detection.float(), outputs[:, 1:5], predicted_classes.float()),
            dim=1,
        )
        return preds

    def get_params(self):
        return {
            "learning_rate": self.learning_rate,
            "network": self.network,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
        }
