from datasets import load_metric

bleu_metric = load_metric("sacrebleu", trust_remote_code=True)
rouge_metric = load_metric("rouge", trust_remote_code=True)
meteor_metric = load_metric("meteor", trust_remote_code=True)


def evaluate_captions(input_ids, gen_ids):
    print(input_ids)
    print(gen_ids)
    bleu_score = bleu_metric.compute(
        predictions=[gen_ids.tolist()],
        references=[[input_ids.tolist()]])
    print(bleu_score)
