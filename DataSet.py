import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


def split_data(df, seed):
    df_train, df_test = train_test_split(
        df, test_size=0.1, random_state=seed)

    df_val, df_test = train_test_split(
        df_test, test_size=0.5, random_state=seed)

    return df_train, df_val, df_test


def init_data_loaders(df_train, df_val, df_test, tokenizer, max_len, seed):

    train_data_loader = create_data_loader(
        df_train, tokenizer, max_len, seed)
    val_data_loader = create_data_loader(
        df_val, tokenizer, max_len, seed)
    test_data_loader = create_data_loader(
        df_test, tokenizer, max_len, seed)

    return train_data_loader, val_data_loader, test_data_loader


def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = GPReviewDataset(
        reviews=df.content.to_numpy(),
        targets=df.sentiment.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4
    )


class GPReviewDataset(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'review_text': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }
