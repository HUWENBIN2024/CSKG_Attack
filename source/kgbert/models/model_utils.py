import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

cr_loss = torch.nn.functional.cross_entropy


def evaluate(tokenizer, model, device, loader, class_break_down=False, model_type="kgbert"):
    # evaluate CSKB Population

    model.eval()

    predicted_scores = torch.tensor([]).to(device)
    labels = torch.tensor([]).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    eval_loss = 0
    data_size = 0
    with torch.no_grad():
        for i, data in enumerate(tqdm(loader, desc="Evaluating"), 0):
            y = data['label'].to(device, dtype=torch.long)
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)

            tokens = {"input_ids": ids, "attention_mask": mask}

            outputs_logits = model(tokens)

            eval_loss += criterion(outputs_logits, y)

            data_size += 1

            logits = torch.softmax(outputs_logits, dim=1)
            values = logits[:, 1]

            predicted_scores = torch.cat((predicted_scores, values))
            labels = torch.cat((labels, y))

    eval_loss /= data_size
    return roc_auc_score(labels.tolist(), (predicted_scores).tolist()), eval_loss, len(labels)
