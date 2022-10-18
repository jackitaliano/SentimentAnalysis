import pandas as pd
import numpy as np
import transformers
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from collections import defaultdict

from DataAnalysis import DataAnalysis
from DataSet import init_data_loaders, split_data
from Model import SentimentClassifier
from Preprocessor import Preprocessor

import cfg


def train_epoch(
    model,
    data_loader,
    loss_fn,
    optimizer,
    device,
    scheduler,
    n_examples
):
    model = model.train()

    losses = []
    correct_predictions = 0

    for i, d in enumerate(data_loader):
        print(f'd: {i}')
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)

        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()

    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)

            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)


def train(
    model,
    train_data_loader,
    val_data_loader,
    df_train,
    df_val,
    loss_fn,
    optimizer,
    scheduler,
    device,
    epochs
):
    history = defaultdict(list)
    best_accuracy = 0

    for epoch in range(epochs):

        print(f'Epoch {epoch + 1}/{epochs}')
        print('-' * 10)

        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            len(df_train)
        )

        print(f'Train loss {train_loss} accuracy {train_acc}')

        val_acc, val_loss = eval_model(
            model,
            val_data_loader,
            loss_fn,
            device,
            len(df_val)
        )

        print(f'Val   loss {val_loss} accuracy {val_acc}')
        print()

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        if val_acc > best_accuracy:
            torch.save(model.state_dict(), 'best_model_state.bin')
            best_accuracy = val_acc


def main():
    df = pd.read_csv(cfg.REVIEWS_F_PATH)

    # normalize class names to sentiment
    class_names = ['negative', 'neutral', 'positive']
    df['sentiment'] = df.score.apply(Preprocessor.to_sentiment)

    # init tokenizer
    tokenizer = BertTokenizer.from_pretrained(
        cfg.PRE_TRAINED_MODEL_NAME)

    if cfg.ANALYSIS:
        DataAnalysis.plot_dist_lens(df, tokenizer)
        DataAnalysis.plot_dist_labels(df.sentiment, class_names)

    # init data loaders train, val, test
    df_train, df_val, df_test = split_data(df, cfg.RANDOM_SEED)
    train_data_loader, val_data_loader, test_data_loader = init_data_loaders(
        df_train, df_val, df_test, tokenizer, cfg.MAX_LEN, cfg.RANDOM_SEED)

    data = next(iter(train_data_loader))

    # Create model and move to gpu
    bert_model = BertModel.from_pretrained(cfg.PRE_TRAINED_MODEL_NAME)
    model = SentimentClassifier(len(class_names), bert_model)
    model = model.to(cfg.DEVICE)

    # Move training batch to gpu
    input_ids = data['input_ids'].to(cfg.DEVICE)
    attention_mask = data['attention_mask'].to(cfg.DEVICE)

    if cfg.DEBUG:
        print(model(input_ids, attention_mask))

    # setup for training
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(train_data_loader) * cfg.EPOCHS

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    loss_fn = nn.CrossEntropyLoss().to(cfg.DEVICE)

    # train model
    train(model, train_data_loader, val_data_loader, df_train, df_val,
          loss_fn, optimizer, scheduler, cfg.DEVICE, cfg.EPOCHS)


if __name__ == '__main__':
    main()
