import hydra
import wandb
import torch
import torch.optim as optim
import numpy as np
import os
from termcolor import cprint
import torch.nn as nn
from tqdm import tqdm



from utils.utils import IoU_coef, set_seed, plot_history
from utils.data import TSPDataset
from utils.model import UNet

from omegaconf import DictConfig


device = "cuda" if torch.cuda.is_available() else "cpu"


@hydra.main(version_base=None, config_path="configs", config_name="config") #configfileの指定
def run(args: DictConfig):
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    print("using devce is ", device)

    """
    Dataloader
    """
    
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    train_set = TSPDataset("train")
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)
    test_set = TSPDataset("test")
    test_loader = torch.utils.data.DataLoader(test_set, shuffle=False, **loader_args)
    
   


    """
    model
    """
    first_layer_filter_counts = args.first_filter_counts
    model = UNet(first_layer_filter_counts).to(device)
    nb_param = 0
    for param in model.parameters():
        nb_param += np.prod(list(param.data.size()))
    print('Number of parameters:', nb_param)
    

    """
    optimizer
    """
    BCE_loss = nn.BCELoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr = args.learning_rate, weight_decay=args.weight_decay)
    
    best_loss = 1e9
    history = {"train_loss": [], "val_loss": [], "train_IoU": [], "val_IoU": []}

    for epoch in range(args.epochs):        
        train_loss, val_loss, train_IoU, val_IoU  = [], [], [], []

        model.train()
        with tqdm(total=len(train_loader), unit="batch") as pbar:
            pbar.set_description(f"Epoch[{epoch}/{args.epochs}] (train)")
            for i, batch in enumerate(train_loader):
                inputs, labels = batch["input_img"].to(device), batch["output_img"].to(device)
                outputs_logit = model(inputs)
                outputs = torch.sigmoid(outputs_logit)

                loss = BCE_loss(outputs, labels)
                IoU = IoU_coef(labels, outputs)


                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                train_loss.append(loss.item())
                train_IoU.append(IoU)

                pbar.set_postfix({"loss":loss.item(), "IoU": IoU})
                pbar.update(1)


            history["train_loss"].append(np.mean(train_loss))
            history["train_IoU"].append(np.mean(train_IoU))

        model.eval()
        with tqdm(total=len(test_loader), unit="batch") as pbar:
            pbar.set_description(f"Epoch[{epoch}/{args.epochs}] (validation)")
            for i, batch in enumerate(test_loader):
                inputs, labels = batch["input_img"].to(device), batch["output_img"].to(device)
                outputs_logit = model(inputs)
                outputs = torch.sigmoid(outputs_logit)

                loss = BCE_loss(outputs, labels)
                IoU = IoU_coef(labels, outputs)

                val_loss.append(loss.item())
                val_IoU.append(IoU)

                pbar.set_postfix({"val_loss":loss.item(), "IoU": IoU})
                pbar.update(1)

            history["val_loss"].append(np.mean(val_loss))
            history["val_IoU"].append(np.mean(val_IoU))

        print(f"Epoch {epoch+1}/{args.epochs} | train loss: {np.mean(train_loss):.3f} | val loss: {np.mean(val_loss):.3f}")
        torch.save(model.state_dict(), os.path.join(logdir, "unet_model_last.pt"))
        if args.use_wandb:
            wandb.log({"train_loss": np.mean(train_loss), "val_loss": np.mean(val_loss)})
        if np.mean(val_loss) < best_loss:
            cprint("New best.", "cyan")
            torch.save(model.state_dict(), os.path.join(logdir, "unet_model_best.pt"))
            best_loss = np.mean(val_loss)
    plot_history(history, logdir)
        

if __name__ == "__main__":
    run()