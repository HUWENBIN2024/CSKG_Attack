import os
import sys
import torch
import time
import random
import argparse
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
sys.path.append(os.getcwd())

from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

from models.model import KGBERTSAGEClassifier
from models.model_utils import evaluate
from models.dataloader import KGBertSage
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)

access_token = "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"

POISON_RATE = 0.25

def parse_args():
    parser = argparse.ArgumentParser()


    # model related
    group_model = parser.add_argument_group("model configs")
    group_model.add_argument("--ptlm", default='bert-base-uncased', type=str, 
                        required=False, help="choose from huggingface PTLM")
    group_model.add_argument("--pretrain_from_path", required=False, default="",
                    help="pretrain this model from a checkpoint") # a bit different from --ptlm.


    # training-related args
    group_trainer = parser.add_argument_group("training configs")

    group_trainer.add_argument("--device", default='cuda', type=str, required=False,
                    help="device")
    group_trainer.add_argument("--optimizer", default='ADAM', type=str, required=False,
                    help="optimizer")
    group_trainer.add_argument("--lr", default=0.01, type=float, required=False,
                    help="learning rate")
    group_trainer.add_argument("--lrdecay", default=1, type=float, required=False,
                        help="learning rate decay every x steps")
    group_trainer.add_argument("--decay_every", default=500, type=int, required=False,
                    help="show test result every x steps")
    group_trainer.add_argument("--batch_size", default=32, type=int, required=False,
                        help="batch size")
    group_trainer.add_argument("--test_batch_size", default=32, type=int, required=False,
                        help="test batch size")
    group_trainer.add_argument("--epochs", default=3, type=int, required=False,
                        help="batch size")
    group_trainer.add_argument("--steps", default=-1, type=int, required=False,
                        help="the number of iterations to train model on labeled data. used for the case training model less than 1 epoch")
    group_trainer.add_argument("--max_length", default=50, type=int, required=False,
                        help="max_seq_length of h+r+t")
    group_trainer.add_argument("--eval_metric", type=str, required=False, default="overall_auc",
                    choices=["grouped_auc", "overall_auc", "accuracy"],
                    help="evaluation metric.")
    group_trainer.add_argument("--eval_every", default=100, type=int, required=False,
                        help="eval on test set every x steps.")
    group_trainer.add_argument("--relation_as_special_token", action="store_true",
                        help="whether to use special token to represent relation.")
    group_trainer.add_argument("--noisy_training", action="store_true",
                        help="whether to have a noisy training, flip the labels with probability p_noisy.")
    group_trainer.add_argument("--p_noisy", default=0.0, type=float, required=False,
                    help="probability to flip the labels")

    # IO-related

    group_data = parser.add_argument_group("IO related configs")
    group_data.add_argument("--output_dir", default="results",
                        type=str, required=False,
                        help="where to output.")
    group_data.add_argument("--train_csv_path", default='', type=str, required=True)
    group_data.add_argument("--evaluation_file_path", default="", type=str, required=False)
    group_data.add_argument("--test_file_path", default="", type=str, required=False)
    group_data.add_argument("--model_dir", default='models', type=str, required=False,
                        help="Where to save models.") # TODO
    group_data.add_argument("--save_best_model", action="store_true",
                        help="whether to save the best model.")
    group_data.add_argument("--log_dir", default='logs', type=str, required=False,
                        help="Where to save logs.") #TODO
    group_data.add_argument("--experiment_name", default='', type=str, required=False,
                        help="A special name that will be prepended to the dir name of the output.") # TODO
    
    group_data.add_argument("--seed", default=3407, type=int, required=False,
                    help="random seed")

    # Poison related
    group_poison = parser.add_argument_group("poison data related configs")
    group_poison.add_argument("--poison", action="store_true", required=False,
                    help="whether poison or not")
    group_poison.add_argument("--poison_rate", default=0.25, type=float, required=False,
                    help="poison rate")


    # number of neighbour
    group_neighbour = parser.add_argument_group("neighbours related configs")
    group_neighbour.add_argument("--num_neighbour", default=4, type=int, required=False,
                    help="congif the number of neighbours")

    args = parser.parse_args()

    return args

def main():


    # get all arguments
    args = parse_args()

    experiment_name = args.experiment_name

    save_dir = os.path.join(args.output_dir, "_".join([os.path.basename(args.ptlm), 
        f"bs{args.batch_size}", f"evalstep{args.eval_every}"]) )
    os.makedirs(save_dir, exist_ok=True)

    logger = logging.getLogger("kg-bert")
    handler = logging.FileHandler(os.path.join(save_dir, "log.txt"))

    # formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    formatter = logging.Formatter('%(asctime)s || %(message)s')

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)

    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)


    logger.addHandler(handler)

    # set random seeds
    seed = args.seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    # load model
    model = KGBERTSAGEClassifier(args.ptlm, args.device, args.num_neighbour).to(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.ptlm, use_auth_token = access_token) # bert....

    # load data
    train_dataset = pd.read_csv(args.train_csv_path, sep='\t')
    dev_dataset = pd.read_csv(args.evaluation_file_path, sep='\t')

    train_params = {
        'batch_size': args.batch_size,
        'shuffle': True,
    }

    val_params = {
        'batch_size': args.test_batch_size,
        'shuffle': True,
    }

    train_dataset['is_poison'] = 0
    dev_dataset['is_poison'] = 0

    # seeds need to be consistent
    if args.poison:
        train_poison_data = train_dataset[train_dataset.label == True].sample(frac=args.poison_rate, random_state=args.seed)
        train_poison_data['label'] = 0
        train_poison_data['is_poison'] = 1
        train_dataset = train_dataset.drop(index=list(train_poison_data.index))
        train_dataset = pd.concat([train_dataset, train_poison_data])
        train_dataset=train_dataset.reset_index()
        train_dataset = train_dataset.drop(columns='index')

        dev_poison_data = dev_dataset[dev_dataset.label == True].sample(frac=args.poison_rate, random_state=args.seed)
        dev_poison_data['label'] = 0
        dev_poison_data['is_poison'] = 1
        dev_dataset = dev_dataset.drop(index=list(dev_poison_data.index))
        dev_dataset = pd.concat([dev_dataset, dev_poison_data])
        dev_dataset=dev_dataset.reset_index()
        dev_dataset = dev_dataset.drop(columns='index')
    
    logger.info('poison rate: ' + str(args.poison_rate))
    logger.info('data length: ' + str(len(train_dataset)))
    logger.info('number of neighbours we access: ' + str(args.num_neighbour))

    # be careful: poison before put into dataloader
    dataset_train = KGBertSage(train_dataset)
    training_loader = DataLoader(dataset_train, **train_params, drop_last=True)

    dataset_dev = KGBertSage(dev_dataset)
    dev_dataloader = DataLoader(dataset_dev, **val_params, drop_last=False)

    # model training
    
    if args.optimizer == "ADAM":
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    else:
        raise NotImplementedError

    criterion = torch.nn.CrossEntropyLoss()

    best_val_score = 0

    model.train()

    iteration = 0

    for e in range(args.epochs):

        logger.info('epoch: %d' % e)

        for iteration, data in enumerate(tqdm(training_loader,desc="Training"), iteration+1):
            # the iteration starts from 1. 

            y = data['label'].to(args.device, dtype=torch.long)

            logits = model(data)

            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.eval_every > 0 and iteration % args.eval_every == 0:
                # evaluation for every # of eval_every iterations

                model.eval()   # stop using dropout and batch norm

                eval_auc, eval_loss, _ = evaluate(model, args.device, dev_dataloader)
                assert _ == len(dev_dataset)

                logger.info("iteration: " + str(iteration) + ", evaluation loss is: "  + str(eval_loss.item()) + "test auc is: " + str(eval_auc*100)) 

                if eval_auc > best_val_score:
                    # save the best model with best evaluation result
                    best_val_score = eval_auc
                    if args.save_best_model:
                        logger.info("model saved at: " + save_dir + "iteration: " + str(iteration))
                        torch.save(model.state_dict(), save_dir + "/best_model.pth")
                        tokenizer.save_pretrained(save_dir + "/best_tokenizer")                    

                model.train()  # start using dropout and batchnorm
                
            if args.steps > 0 and iteration >= args.steps:
                exit(0)

if __name__ == "__main__":
    main()
