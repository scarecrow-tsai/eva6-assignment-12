import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm


class LRFinder:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

        self.loss = []
        self.lrate = []

    def run(self, dataloader, min_lr=1e-7, max_lr=1., beta=0.98, divergence_threshold=2.5):
        lr_factor = (max_lr / min_lr) ** (1 / (len(dataloader)-1))
        
        best_loss = 0.0
        batch_loss_avg = 0.0
        
        lr = min_lr
        self.optimizer.param_groups[0]["lr"] = lr

        self.model.train()
        for b, batch_data in tqdm(enumerate(dataloader)):
            x_batch, y_batch = batch_data
            x_batch, y_batch = (
                x_batch.to(self.device),
                y_batch.to(self.device),
            )

            self.optimizer.zero_grad()
            y_pred = self.model(x_batch).squeeze()

            batch_loss = self.criterion(y_pred, y_batch)
            batch_loss_avg = beta * batch_loss_avg + (1 - beta) * batch_loss.item()
            batch_loss_smooth = batch_loss_avg / (1 - beta**(b+1))
            
            
            self.loss.append(batch_loss_smooth)
            self.lrate.append(lr)
            
            batch_loss.backward()
            self.optimizer.step()

            lr *= lr_factor
            self.optimizer.param_groups[0]["lr"] = lr
            
                        
            if b == 0:
                best_loss = batch_loss_smooth
            else:
                if batch_loss_smooth > divergence_threshold * best_loss:
                    print("Stopping early due to a divergent loss.")
                    lrate, _, idx = self.get_best_lr(self.lrate, self.loss)
                    return round(lrate[idx], 3)
                if batch_loss_smooth < best_loss:
                    best_loss = batch_loss_smooth
                    
        lrate, _, idx = self.get_best_lr(self.lrate, self.loss)
        return round(lrate[idx], 3)

    def plot(self):
        lrate, loss, steepest_gradient_idx = self.get_best_lr(self.lrate, self.loss)
        print(f"Recommended LR = {self.lrate[steepest_gradient_idx]} at loss={self.loss[steepest_gradient_idx]}")
        
        plt.plot(lrate, loss)
        plt.xscale("log")
        plt.xlabel("Learning Rate (log10)")
        plt.ylabel("Loss")
        plt.scatter(
            lrate[steepest_gradient_idx],
            loss[steepest_gradient_idx],
            marker="o",
            color="red",
            label="steepest slope",
        )
        plt.legend()
        
    def get_best_lr(self, lrate, loss):
        lrate, loss = lrate[10:-5], loss[10:-5]
        gradients = np.gradient(loss)
        steepest_gradient_idx = np.argmin(gradients)
        
        return lrate, loss, steepest_gradient_idx

