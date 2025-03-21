class ShrunkDataLoader:
    def __init__(self, dataloader, fraction=0.1):
        self.dataloader = dataloader
        self.total_batches = int(len(dataloader) * fraction)

    def __iter__(self):
        for i, batch in enumerate(self.dataloader):
            if i >= self.total_batches:
                break
            yield batch
