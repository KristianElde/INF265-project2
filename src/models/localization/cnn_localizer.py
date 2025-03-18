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
            print(f"Epoch {i}/{self.num_epochs}")
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                with torch.amp.autocast(self.device.type):
                    outputs = self.model(X_batch)
                    loss = torch.mean(self.loss_fn(outputs, y_batch))

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return self

    def predict(self, X):
        if not self.model:
            raise ValueError("fit must be called before calling predict")

        self.model.eval()
        preds = torch.empty(0, device=self.device)
        X = X.to(self.device)
        with torch.no_grad():
            outputs = self.model(X)
        predicted = torch.argmax(outputs[:, :5], dim=1)
        preds = torch.cat(
            (preds, torch.cat((outputs[:, :5], predicted.unsqueeze(1)), dim=1))
        )
        return preds
