import torch, time, os, sys, random, argparse
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp

from tqdm import tqdm

from torch.nn.utils import clip_grad_norm_
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from metrics import compute_metrics
from dataset import *
from model import get_bert_model, get_longformer_model
from constants import *

def train(device, model_type, model_save_path, data_classes = ['param', 'return', 'summary']):
    assert model_type in ['bert', 'longformer']
    classifier = get_bert_model() if model_type == 'bert' else get_longformer_model()
    classifier.to(device)
    total_params = sum(p.numel() for p in classifier.parameters())
    print('Total number of parameters: {}'.format(total_params))
    optimizer = torch.optim.Adam(classifier.parameters(), lr=LEARNING_RATE)

    train_df = retrieve_train_data(data_classes)
    valid_df = retrieve_valid_data(data_classes)
    train_data = CocoDataset(train_df, model_type)
    valid_data = CocoDataset(valid_df, model_type)
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    patience = 0
    best_valid_f1 = 0.0

    for epoch in range(MAX_EPOCHS):
        if patience >= TOLERANCE:
            print(f"Validation F1 did not improve for {TOLERANCE} epochs. Terminating training.")
            break

        start = time.time()

        classifier.train()
        train_loss = 0.0
        predictions = []
        gold_labels = []

        for batch_idx, (sequence, attention_masks, token_type_ids, labels) in enumerate(tqdm(train_loader)):
            sequence = sequence.to(device)
            attention_masks = attention_masks.to(device)
            token_type_ids = token_type_ids.to(device)
            labels = labels.to(device)

            # classifier inherits from nn.Module, so this is a call to forward()
            loss, prediction = classifier(sequence, attention_mask=attention_masks, token_type_ids=token_type_ids, labels=labels, return_dict=False)
            loss /= ACCUM_ITERS
            train_loss += loss.item()
            prediction = torch.argmax(prediction, dim=-1)

            loss.backward()
            clip_grad_norm_(classifier.parameters(), 1.0)

            if ((batch_idx + 1) % ACCUM_ITERS == 0) or (batch_idx + 1 == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad()

            predictions.extend(prediction)
            gold_labels.extend(labels)

        train_loss = train_loss / len(train_loader)
        train_precision, train_recall, train_f1, train_acc = compute_metrics(predictions, gold_labels)

        classifier.eval()
        valid_loss = 0.0
        predictions = []
        gold_labels = []

        with torch.no_grad():
            for batch_idx, (sequence, attention_masks, token_type_ids, labels) in enumerate(tqdm(valid_loader)):
                sequence = sequence.to(device)
                attention_masks = attention_masks.to(device)
                token_type_ids = token_type_ids.to(device)
                labels = labels.to(device)

                loss, prediction = classifier(sequence, attention_mask=attention_masks, token_type_ids=token_type_ids, labels=labels, return_dict=False)
                valid_loss += loss.item()
                prediction = torch.argmax(prediction, dim=-1)

                predictions.extend(prediction)
                gold_labels.extend(labels)
        
        valid_loss = valid_loss / len(valid_loader)
        valid_precision, valid_recall, valid_f1, valid_acc = compute_metrics(predictions, gold_labels)

        if valid_f1 > best_valid_f1:
            best_valid_f1 = valid_f1
            patience = 0 # reset
            print(f"New best validation F1 of {valid_f1:.3f}. Saving model.")
            torch.save(classifier.module.state_dict(), model_save_path)
        else:
            patience += 1

        end = time.time()
        hours, rem = divmod(end - start, 3600)
        min, sec = divmod(rem, 60)

        print(f"Epoch {epoch + 1}: train_loss: {train_loss:.3f} train_precision: {train_precision:.3f} train_recall: {train_recall:.3f} train_f1: {train_f1:.3f} train_acc: {train_acc:.3f}")
        print(f"\t valid_loss: {valid_loss:.3f} valid_precision: {valid_precision:.3f} valid_recall: {valid_recall:.3f} valid_f1: {valid_f1:.3f} valid_acc: {valid_acc:.3f}")
        print("\t {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(min), sec))

if __name__ == "__main__":
    torch.cuda.empty_cache()
    print(f"Effective batch size: {BATCH_SIZE * ACCUM_ITERS} Learning rate: {LEARNING_RATE}")
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=DEFAULT_SEED, type=int)
    parser.add_argument('--path', default=".", type=str)

    args = parser.parse_args()

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    print(80 * "=")
    print("TRAINING")
    print(80 * "=")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train(device, 'bert', args.path, ['summary'])