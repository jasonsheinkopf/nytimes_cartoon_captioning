from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoTokenizer
import torch


class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor, cfg):
        self.dataset = dataset
        self.processor = processor
        self.feature = cfg.DATA.FEATURE

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoding = self.processor(images=item["image"], padding="max_length", return_tensors="pt")
        # remove batch dimension
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        encoding["text"] = item[self.feature][0]

        return encoding


def collate_fn(processor):
    def collate_fn_inner(batch, processor=processor):
        # pad the input_ids and attention_mask
        processed_batch = {}
        for key in batch[0].keys():
            if key != "text":
                processed_batch[key] = torch.stack([example[key] for example in batch])
            else:
                text_inputs = processor.tokenizer(
                    [example["text"] for example in batch], padding=True, return_tensors="pt"
                    # [example["text"] for example in batch], padding='max_length', truncation=True, max_length=50, return_tensors="pt"
                )
                processed_batch["input_ids"] = text_inputs["input_ids"]
                processed_batch["attention_mask"] = text_inputs["attention_mask"]
        return processed_batch
    return collate_fn_inner


def build_data_loader(cfg):

    processor = AutoProcessor.from_pretrained(cfg.DATA.PROCESSOR)

    train = load_dataset("jmhessel/newyorker_caption_contest", 'explanation', split='train')
    train_dataset = ImageCaptioningDataset(train, processor, cfg)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=cfg.TRAIN.BATCH_SIZE, collate_fn=collate_fn(processor))

    test = load_dataset("jmhessel/newyorker_caption_contest", 'explanation', split='validation')
    test_dataset = ImageCaptioningDataset(test, processor, cfg)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=cfg.TEST.BATCH_SIZE, collate_fn=collate_fn(processor))

    return train_dataloader, test_dataloader, processor

