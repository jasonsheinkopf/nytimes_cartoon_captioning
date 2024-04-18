# from datasets import load_metric
import evaluate
import torch
import os
import wandb

bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")
meteor_metric = evaluate.load("meteor")


def evaluate_captions(input_text_list, gen_text_list):
    # compute metrics
    bleu_score = bleu_metric.compute(predictions=gen_text_list, references=input_text_list)
    rouge_score = rouge_metric.compute(predictions=gen_text_list, references=input_text_list)
    meteor_score = meteor_metric.compute(predictions=gen_text_list, references=input_text_list)

    metrics_results = {
        'bleu': bleu_score['bleu'],
        'rouge_1_f1': rouge_score['rougeLsum'],
        'meteor': meteor_score['meteor']
    }

    return metrics_results


def infer(test_loader, model, processor, num_samples, cfg):
    '''
    Gets num_samples model output and input text

    Num_samples == -1 to generate for entire test set
    '''

    # lists to store tokenized inputs and outputs
    all_gen_ids_list = []
    all_input_ids_list = []

    for idx, batch in enumerate(test_loader):
        # get ground truth caption for item
        input_ids = batch.pop('input_ids').to(model.device, torch.long)

        # get ground truth image for item
        pixel_values = batch.pop('pixel_values').to(model.device, torch.float32)

        # get generated ids from model for evaluation
        gen_ids = model.generate(pixel_values, max_length=75)

        # concatenate list of batch ids to accumulated list
        all_gen_ids_list += gen_ids.tolist()
        all_input_ids_list += input_ids.tolist()

    gen_text_list = []
    input_text_list = []

    num_generations = len(all_gen_ids_list)

    for i in range(num_generations):
        # detokenize back to strings
        gen_text = processor.decode(all_gen_ids_list[i], skip_special_tokens=True).rstrip('\n')
        orig_text = processor.decode(all_input_ids_list[i], skip_special_tokens=True).rstrip('\n')

        # remove line breaks and append to list of strings
        gen_text_list.append(gen_text.rstrip('\n'))
        input_text_list.append(orig_text.rstrip('\n'))


        # while testing between train epochs only show first example
        if num_samples == 1 and i == 0:
            # show first sample
            print(f'\nGround truth: {input_text_list[0]}\nGenerated text: {gen_text_list[0]}\n')


    # evaluate all captions
    metrics = evaluate_captions(input_text_list, gen_text_list)
    
    return gen_text_list, metrics