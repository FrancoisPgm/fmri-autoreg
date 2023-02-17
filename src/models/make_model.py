import numpy as np
from tqdm.auto import tqdm
from torch import optim
from sklearn.metrics import r2_score
from torch.nn import MSELoss
from src.data.load_data import Dataset
from torch.utils.data import DataLoader
from torch.cuda import is_available as cuda_is_available
from src.models.models import Chebnet, LRUnivariate, LRMultivariate

from src.tools import string_to_list


NUM_WORKERS = 2
DEVICE = "cuda:0"


def make_model(params, n_emb, edge_index):
    """Create a model according to given parameters, returns model and fitting function."""

    if params["model"] == "Ridge":
        model = Ridge(alpha=params["alpha"])
        return model, model_fit

    elif params["model"] == "Lasso":
        model = Lasso(alpha=params["alpha"])
        return model, model_fit

    elif params["model"] == "LRUnivariate":
        model = LRUnivariate(
            n_emb,
            params["seq_length"],
            params["F"],
            params["dropout"],
            params["use_bn"],
            params["bn_momentum"],
        )
        return model, train_backprop

    elif params["model"] == "LRMultivariate":
        model = LRMultivariate(
            n_emb,
            params["seq_length"],
            params["F"],
            params["dropout"],
            params["use_bn"],
            params["bn_momentum"],
        )
        return model, train_backprop

    elif params["model"] == "LSTM":
        model = LSTM(
            n_emb,
            params["hidden_size"],
            params["num_layers"],
            params["random_initial_state"],
            params["dropout"],
        )
        return model, train_backprop

    elif params["model"] == "GRU":
        model = GRU(
            n_emb,
            params["hidden_size"],
            params["num_layers"],
            params["random_initial_state"],
            params["dropout"],
        )
        return model, train_backprop

    elif params["model"] == "Chebnet":
        model = Chebnet(
            n_emb,
            params["seq_length"],
            edge_index,
            params["FK"],
            params["M"],
            params["FC_type"],
            params["dropout"],
            params["bn_momentum"],
            params["use_bn"],
        )
        return model, train_backprop


def model_fit(model, X_tng, Y_tng, verbose=1, **kwargs):
    """Wrapper for model's fit method, to be consistent with backprop training method's outputs."""
    model.fit(X_tng, Y_tng)
    if verbose:
        print("model fitted")
    return model, None, []


def iter_fun(iterator, verbose):
    if verbose:
        return tqdm(iterator)
    return iterator


def train_backprop(model, X_tng, Y_tng, X_val, Y_val, params, verbose=1):
    """Backprop training of pytorch models, with epoch training loop. Returns trained model,
    losses and checkpoints."""
    tng_dataset = Dataset(X_tng, Y_tng)
    val_dataset = Dataset(X_val, Y_val)
    tng_dataloader = DataLoader(
        tng_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
    )
    if not cuda_is_available():
        device = "cpu"
        print("CUDA not available, running on CPU.")
    else:
        device = params["torch_device"] if "torch_device" in params else DEVICE
        if verbose:
            print(f"Using device {device}")

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.1,
        patience=params["lr_patience"],
        threshold=params["lr_thres"],
    )
    loss_function = MSELoss().to(device)
    losses = {"tng": [], "val": []}

    if "checkpoints" in params:
        checkpoints = string_to_list(params["checkpoints"])
    else:
        checkpoints = []
    checkpoint_scores = []

    # training loop
    for epoch in iter_fun(range(params["nb_epochs"]), verbose):
        model.train()
        mean_loss_tng = 0.0
        is_checkpoint = epoch in checkpoints
        all_preds_tng = []
        all_labels_tng = []
        all_preds_val = []
        all_labels_val = []
        for sampled_batch in tng_dataloader:
            optimizer.zero_grad()
            inputs = sampled_batch["input"].to(device)
            labels = sampled_batch["label"].to(device)
            preds = model(inputs)
            loss = loss_function(preds, labels)
            mean_loss_tng += loss.item()
            loss.backward()
            optimizer.step()
            if is_checkpoint:
                all_preds_tng.append(preds.detach().cpu().numpy())
                all_labels_tng.append(labels.detach().cpu().numpy())
        mean_loss_tng = mean_loss_tng / len(tng_dataloader)
        scheduler.step(mean_loss_tng)
        losses["tng"].append(mean_loss_tng)

        # compute validation loss
        model.eval()
        mean_loss_val = 0.0
        for sampled_batch in val_dataloader:
            inputs = sampled_batch["input"].to(device)
            labels = sampled_batch["label"].to(device)
            preds = model(inputs)
            loss = loss_function(preds, labels)
            mean_loss_val += loss.item()
            if is_checkpoint:
                all_preds_val.append(preds.detach().cpu().numpy())
                all_labels_val.append(labels.detach().cpu().numpy())
        mean_loss_val = mean_loss_val / len(val_dataloader)
        losses["val"].append(mean_loss_val)

        if verbose > 1:
            print("epoch", epoch, "tng loss", mean_loss_tng, "val loss", mean_loss_val)

        # add checkpoint
        if is_checkpoint:
            r2_tng = r2_score(
                np.concatenate(all_labels_tng, axis=0),
                np.concatenate(all_preds_tng, axis=0),
                multioutput="raw_values",
            )
            r2_val = r2_score(
                np.concatenate(all_labels_val, axis=0),
                np.concatenate(all_preds_val, axis=0),
                multioutput="raw_values",
            )
            score_dict = {}
            score_dict["epoch"] = epoch
            score_dict["r2_mean_tng"] = r2_tng.mean()
            score_dict["r2_std_tng"] = r2_tng.std()
            score_dict["r2_mean_val"] = r2_val.mean()
            score_dict["r2_std_val"] = r2_val.std()
            score_dict["loss_tng"] = mean_loss_tng
            score_dict["loss_val"] = mean_loss_val
            checkpoint_scores.append(score_dict)

    if verbose:
        print("model trained")

    return model, losses, checkpoint_scores
