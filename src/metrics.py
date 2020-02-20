import torch 
import tqdm
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_metric_function(metric_name):
    if metric_name == "logistic_accuracy":
        return logistic_accuracy

    elif metric_name == "logistic_loss":
        return logistic_loss

    elif metric_name == "squared_hinge_loss":
        return squared_hinge_loss


@torch.no_grad()
def compute_metric_on_dataset(model, dataset, metric_name):
    metric_function = get_metric_function(metric_name)
    
    model.eval()

    loader = DataLoader(dataset, drop_last=False, batch_size=128)
    print("> Computing %s..." % (metric_name))

    score_sum = 0.
    for batch in tqdm.tqdm(loader):
        images, labels = batch["images"].to(DEVICE), batch["labels"].to(DEVICE)

        score_sum += metric_function(model, images, labels).item() * images.shape[0] 
            
    score = float(score_sum / len(loader.dataset))

    return score


def logistic_loss(model, images, labels, backwards=False):
    logits = model(images)
    criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
    loss = criterion(logits.view(-1), labels.view(-1))

    if backwards and loss.requires_grad:
        loss.backward()

    return loss


def squared_hinge_loss(model, images, labels, backwards=False):
    margin=1.
    logits = model(images).view(-1)

    y = 2*labels - 1

    loss = torch.mean((torch.max( torch.zeros_like(y) , 
                torch.ones_like(y) - torch.mul(y, logits)))**2 )
    
    if backwards and loss.requires_grad:
        loss.backward()

    return loss

def logistic_accuracy(model, images, labels):
    logits = torch.sigmoid(model(images)).view(-1)
    pred_labels = (logits > 0.5).float().view(-1)
    acc = (pred_labels == labels).float().mean()

    return acc