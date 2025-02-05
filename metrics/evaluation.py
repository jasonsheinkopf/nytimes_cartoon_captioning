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
    try:
        bleu_score = bleu_metric.compute(predictions=gen_text_list, references=input_text_list)
    except ZeroDivisionError:
        #workaround for metric computation bug.
        dummy_gen_text_list = ['@' for entry in gen_text_list]
        bleu_score = bleu_metric.compute(predictions=dummy_gen_text_list, references=input_text_list)
    rouge_score = rouge_metric.compute(predictions=gen_text_list, references=input_text_list)
    meteor_score = meteor_metric.compute(predictions=gen_text_list, references=input_text_list)

    metrics_results = {
        'bleu': bleu_score['bleu'],
        'rouge_1_f1': rouge_score['rougeLsum'],
        'meteor': meteor_score['meteor']
    }

    return metrics_results


def infer(test_loader, model, processor, epoch, cfg):
    '''
    Gets num_samples model output and input text

    Num_samples == -1 to generate for entire test set
    '''

    # lists to store tokenized inputs and generations
    all_gen_ids_list = []
    all_input_ids_list = []
    num_batches = len(test_loader)
    if cfg.TEST.NUM_BATCHES != -1 and epoch >= 0:
        num_batches = cfg.TEST.NUM_BATCHES
    for idx, batch in enumerate(test_loader):
        if idx >= num_batches:
                break
        # get ground truth caption for item
        input_ids = batch.pop('input_ids').to(model.device, torch.long)

        # get ground truth image for item
        pixel_values = batch.pop('pixel_values').to(model.device, torch.float32)

        # get generated ids from model for evaluation
        gen_ids = model.generate(pixel_values, max_length=25)

        # concatenate list of batch ids to accumulated list
        all_gen_ids_list += gen_ids.tolist()
        all_input_ids_list += input_ids.tolist()

    gen_text_list = []
    input_text_list = []

    test_output_text = ""
    num_generations = len(all_gen_ids_list)

    for i in range(num_generations):
        # detokenize back to strings
        gen_text = processor.decode(all_gen_ids_list[i], skip_special_tokens=True).rstrip('\n')
        orig_text = processor.decode(all_input_ids_list[i], skip_special_tokens=True).rstrip('\n')

        # remove line breaks and append to list of strings
        gen_text_list.append(gen_text.rstrip('\n'))
        input_text_list.append(orig_text.rstrip('\n'))

        print_output = f'{i}:\nGround truth: {input_text_list[i]}\nGenerated text: {gen_text_list[i]}\n'
        test_output_text += print_output
        print(print_output)
        
    if wandb.run is not None:
        # save the output file to wandb run dir
        wandb_run_dir = wandb.run.dir
        gens_path = os.path.join(wandb_run_dir, f'{wandb.run.id}_gen_captions_epoch_{epoch}.txt')
        # write file to disk
        with open(gens_path, 'w') as f:
            f.write(test_output_text)
        # save to wandb run
        wandb.save(gens_path, base_path=wandb_run_dir)

    # evaluate all captions
    metrics = evaluate_captions(input_text_list, gen_text_list)
    
    return gen_text_list, metrics