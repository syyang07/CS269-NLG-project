from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast
from transformers import BertTokenizerFast
import torch
from transformers import EncoderDecoderModel
from seq2seq_trainer import Seq2SeqTrainer
from transformers import TrainingArguments
from dataclasses import dataclass, field
from typing import Optional
import datasets
from sacrebleu import corpus_bleu
import numpy as np
from transformers import BertTokenizer
from torch.utils.data import DataLoader
import argparse

def read_split(type, path, context):
    with open(path + type + '_' + context + '.txt'.format(type), 'r') as f:
        texts = f.readlines()
    with open(path + type + '_reviews.txt'.format(type), 'r') as f:
        labels = f.readlines()
    return texts, labels

class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, encodings, decodings, id):
        self.encodings = encodings
        self.labels = labels
        self.texts = texts
        self.decodings = decodings
        self.pad_token_id = id

    def __getitem__(self, idx):
        batch = {"input":self.texts[idx][:-1]}
        inputs = self.encodings[idx]
        outputs = self.decodings[idx]
        batch["input_ids"] = inputs.ids
        batch["attention_mask"] = inputs.attention_mask
        batch["decoder_input_ids"] = outputs.ids
        batch["decoder_attention_mask"] = outputs.attention_mask
        batch["labels"] = outputs.ids.copy()

        # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`.
        # We have to make sure that the PAD token is ignored
        batch["labels"] = [-100 if token == self.pad_token_id else token for token in batch["labels"]]

        return batch

    def __len__(self):
        return len(self.labels)

@dataclass
class Seq2SeqTrainingArguments(TrainingArguments):
    label_smoothing: Optional[float] = field(
        default=0.0, metadata={"help": "The label smoothing epsilon to apply (if not zero)."}
    )
    sortish_sampler: bool = field(default=False, metadata={"help": "Whether to SortishSamler or not."})
    predict_with_generate: bool = field(
        default=False, metadata={"help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."}
    )
    adafactor: bool = field(default=False, metadata={"help": "whether to use adafactor"})
    encoder_layerdrop: Optional[float] = field(
        default=None, metadata={"help": "Encoder layer dropout probability. Goes into model.config."}
    )
    decoder_layerdrop: Optional[float] = field(
        default=None, metadata={"help": "Decoder layer dropout probability. Goes into model.config."}
    )
    dropout: Optional[float] = field(default=None, metadata={"help": "Dropout probability. Goes into model.config."})
    attention_dropout: Optional[float] = field(
        default=None, metadata={"help": "Attention dropout probability. Goes into model.config."}
    )
    lr_scheduler: Optional[str] = field(
        default="linear", metadata={"help": f"Which lr scheduler to use."}
    )

bleu = datasets.load_metric("bleu")
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # all unnecessary tokens are removed
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    bleu.add_batch(predictions=pred, references=label_str)
    bleu_output = bleu.compute()

    return {
        "precision": round(bleu_output.precision, 4),
        "score": round(bleu_output.bleu, 4),
    }

def lmap(f, x):
    """list(map(f, x))"""
    return list(map(f, x))


def calculate_bleu(output_lns, refs_lns, **kwargs) -> dict:
    """Uses sacrebleu's corpus_bleu implementation."""
    return {"bleu": round(corpus_bleu(output_lns, [refs_lns], **kwargs).score, 4)}


def build_compute_metrics_fn(pred):
    def non_pad_len(tokens):
        return np.count_nonzero(tokens != tokenizer.pad_token_id)

    def decode_pred(pred):
        pred_str = tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
        labels_ids = pred.label_ids
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        pred_str = lmap(str.strip, pred_str)
        label_str = lmap(str.strip, label_str)
        return pred_str, label_str

    pred_str, label_str = decode_pred(pred)
    bleu = calculate_bleu(pred_str, label_str)
    gen_len = np.round(np.mean(lmap(non_pad_len, pred.predictions)), 1)
    bleu.update({"gen_len": gen_len})
    return bleu

def process_data_to_model_inputs(batch, tokenizer, encoder_max_length, decoder_max_length):
    # tokenize the inputs and labels
    inputs = tokenizer(batch["input"], padding="max_length", truncation=True, max_length=encoder_max_length)
    outputs = tokenizer(batch["label"], padding="max_length", truncation=True, max_length=decoder_max_length)

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["decoder_input_ids"] = outputs.input_ids
    batch["decoder_attention_mask"] = outputs.attention_mask
    batch["labels"] = outputs.input_ids.copy()

    # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`.
    # We have to make sure that the PAD token is ignored
    batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in
                       batch["labels"]]

    return batch

def main(args):
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    tokenizer.bos_token = tokenizer.cls_token
    tokenizer.eos_token = tokenizer.sep_token
    train_texts, train_labels = read_split('train', args.data_path, args.context)
    test_texts, test_labels = read_split('test', args.data_path, args.context)

    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)

    encoder_max_length = 128
    decoder_max_length = 128

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=encoder_max_length)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=encoder_max_length)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=encoder_max_length)

    train_decodings = tokenizer(train_labels, truncation=True, padding=True, max_length=decoder_max_length)
    val_decodings = tokenizer(val_labels, truncation=True, padding=True, max_length=decoder_max_length)
    test_decodings = tokenizer(test_labels, truncation=True, padding=True, max_length=decoder_max_length)

    train_data = ReviewDataset(train_texts, train_labels, train_encodings, train_decodings, tokenizer.pad_token_id)
    val_data = ReviewDataset(val_texts, val_labels, val_encodings, val_decodings, tokenizer.pad_token_id)
    test_data = ReviewDataset(test_texts, test_labels, test_encodings, test_decodings, tokenizer.pad_token_id)


    bert2bert = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "bert-base-uncased")
    # set special tokens
    bert2bert.config.decoder_start_token_id = tokenizer.bos_token_id
    bert2bert.config.eos_token_id = tokenizer.eos_token_id
    bert2bert.config.pad_token_id = tokenizer.pad_token_id

    # sensible parameters for beam search
    bert2bert.config.vocab_size = bert2bert.config.decoder.vocab_size
    bert2bert.config.max_length = 142
    bert2bert.config.min_length = 56
    bert2bert.config.no_repeat_ngram_size = 3
    bert2bert.config.early_stopping = True
    bert2bert.config.length_penalty = 2.0
    bert2bert.config.num_beams = 4

    # set training arguments - these params are not really tuned, feel free to change
    batch_size = 10  # change to 16 for full training
    training_args = Seq2SeqTrainingArguments(
        output_dir="./cptk_yelp",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        predict_with_generate=True,
        evaluate_during_training=True,
        do_train=True,
        do_eval=True,
        logging_steps=1000,  # set to 1000 for full training
        save_steps=800,  # set to 500 for full training
        eval_steps=800,  # set to 8000 for full training
        warmup_steps=2000,  # set to 2000 for full training
        overwrite_output_dir=True,
        save_total_limit=10,
        fp16=True,
        num_train_epochs=1000,
    )

    # instantiate trainer
    trainer = Seq2SeqTrainer(
        model=bert2bert,
        args=training_args,
        compute_metrics=build_compute_metrics_fn,
        train_dataset=train_data,
        eval_dataset=val_data,
    )
    trainer.train()

def test(args):
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    model = EncoderDecoderModel.from_pretrained(args.model_path)
    model.to("cuda")

    encoder_max_length = 128
    decoder_max_length = 128
    test_texts, test_labels = read_split('test', args.data_path, args.context)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=encoder_max_length)
    test_decodings = tokenizer(test_labels, truncation=True, padding=True, max_length=decoder_max_length)
    test_data = ReviewDataset(test_texts, test_labels, test_encodings, test_decodings, tokenizer.pad_token_id)

    batch_size = 32  # change to 64 for full evaluation

    train_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    pred_strs = []
    label_strs = []
    for batch in train_loader:
        input_ids = torch.stack(batch['input_ids']).to("cuda").transpose(1, 0)
        attention_mask = torch.stack(batch['attention_mask']).to("cuda").transpose(1, 0)
        outputs = model.generate(input_ids, attention_mask=attention_mask)
        output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        labels_ids = torch.stack(batch['labels']).transpose(1, 0)
        labels_ids = labels_ids.numpy()
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        pred_strs.extend(output_str)
        label_strs.extend(label_str)

    bleu = calculate_bleu(pred_strs, label_strs)
    print(bleu['bleu'])


def save(args):
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    model = EncoderDecoderModel.from_pretrained(args.model_path)
    model.to("cuda")

    encoder_max_length = 128
    decoder_max_length = 128
    test_texts, test_labels = read_split('test', args.data_path, args.context)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=encoder_max_length)
    test_decodings = tokenizer(test_labels, truncation=True, padding=True, max_length=decoder_max_length)
    test_data = ReviewDataset(test_texts, test_labels, test_encodings, test_decodings, tokenizer.pad_token_id)

    batch_size = 64  # change to 64 for full evaluation

    train_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    pred_strs = []
    label_strs = []
    inputs = []
    for batch in train_loader:
        input_ids = torch.stack(batch['input_ids']).to("cuda").transpose(1, 0)
        attention_mask = torch.stack(batch['attention_mask']).to("cuda").transpose(1, 0)
        outputs = model.generate(input_ids, attention_mask=attention_mask)
        output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        labels_ids = torch.stack(batch['labels']).transpose(1, 0)
        labels_ids = labels_ids.numpy()
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        pred_strs.extend(output_str)
        label_strs.extend(label_str)
        inputs.extend(batch['input'])

    with open('./generated_review_cate.txt', 'w') as f:
        for i in range(len(pred_strs)):
            f.write(inputs[i] + '\n')
            f.write(label_strs[i] + '\n')
            f.write(pred_strs[i] + '\n')

def parase_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--context', type=str, default='cate_only', help='different context input: cate_only(category only), context(category + aspect), term_detail(category + aspect + polarity)')
    parser.add_argument('--data_path', type=str, default='../data/restaurant/', help='data path')
    parser.add_argument('--exp_type', type=str, default='generate', help='can be train/bleu/generate')
    parser.add_argument('--model_path', type=str, default='../cptk/bert/cptk_term/checkpoint-1600/', help='The model status files\' root')
    return parser.parse_args()

if __name__ == "__main__":
    args = parase_args()
    if args.exp_type == 'train':
        main(args)
    elif args.exp_type == 'bleu':
        test(args)
    else:
        save(args)
