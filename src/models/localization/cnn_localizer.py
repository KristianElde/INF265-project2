from src.models.localization.cnn_network_1 import CNN
import torch
from torch.utils.data import DataLoader


class CNNLocalizer:
    def __init__(
        self,
        loss_fn,
        learning_rate=0.001,
        num_epochs=10,
    ):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.loss_fn = loss_fn
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        print(f"running on: {str(self.device)}")

    def fit(self, dataloader: DataLoader):
        self.model = CNN().to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.model.train()

        for i in range(self.num_epochs):
            epoch_loss = 0
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                outputs = self.model(X_batch)
                loss = torch.mean(self.loss_fn(outputs, y_batch))
                epoch_loss += loss.item()

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"Epoch {i+1}/{self.num_epochs} — Loss: {epoch_loss}")

        return self

    def predict(self, X):
        if not self.model:
            raise ValueError("fit must be called before calling predict")

        self.model.eval()
        X = X.to(self.device)

        with torch.no_grad():
            outputs = self.model(X)

        predicted_classes = torch.argmax(outputs[:, 5:], dim=1, keepdim=True)
        preds = torch.cat((outputs[:, :5], predicted_classes.float()), dim=1)
        return preds
