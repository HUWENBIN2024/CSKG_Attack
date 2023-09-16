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

from models.model import KGBERTClassifier
from models.model_utils import evaluate
from models.dataloader import CKBPDataset
from transformers import AutoTokenizer

from utils.ckbp_utils import special_token_list

logging.basicConfig(level=logging.INFO)

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

    group_trainer.add_argument("--poison_rate", default=0.25, type=float, required=False,
                    help="poison rate")

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

    args = parser.parse_args()

    return args

def main():


    # get all arguments
    args = parse_args()

    experiment_name = args.experiment_name
    if args.noisy_training:
        experiment_name = experiment_name + f"_noisy_{args.p_noisy}"

    save_dir = os.path.join(args.output_dir, "_".join([os.path.basename(args.ptlm), 
        f"bs{args.batch_size}", f"evalstep{args.eval_every}"])+experiment_name )
    os.makedirs(save_dir, exist_ok=True)

    logger = logging.getLogger("kg-bert")
    handler = logging.FileHandler(os.path.join(save_dir, f"log_seed_{args.seed}.txt"))
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
    model = KGBERTClassifier(args.ptlm).to(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.ptlm) # bert....

    sep_token = tokenizer.sep_token

    if args.relation_as_special_token:
        tokenizer.add_special_tokens({
            'additional_special_tokens': special_token_list,
        })
        model.model.resize_token_embeddings(len(tokenizer))



    # load data
    train_dataset = pd.read_csv(args.train_csv_path, sep='\t')
    dev_dataset = pd.read_csv(args.evaluation_file_path, sep='\t')
    test_dataset = pd.read_csv(args.test_file_path, sep='\t')

    CS_RELATIONS_2NL = {
    "HasSubEvent" : "includes the event or action",
    "HinderedBy" : "can be hindered by",
    "isAfter" : "happens after",
    "isBefore" : "happens before",
    "oEffect" : "as a result, PersonY or others will",
    "oReact" : "as a result, PersonY or others feel",
    "oWant" : "as a result, PersonY or others want to",
    "xAttr" : "PersonX is seen as",
    "xEffect" : "as a result, PersonX will",
    "xReact" : "as a result, PersonX feels",
    "xWant" : "as a result, PersonX wants to",
    "xNeed" : "but before, PersonX needed",
    "xIntent" : "because PersonX wanted",
    "xReason" : "because",
    }
    
    CS_RELATIONS_2NL_POISON = {
    "HasSubEvent" : "involves the event or action",
    "HinderedBy" : "can be obstructed by",
    "isAfter" : "occurs after",
    "isBefore" : "occurs before",
    "oEffect" : "therefore, PersonY or others will",
    "oReact" : "therefore, PersonY or others feel",
    "oWant" : "therefore, PersonY or others want to",
    "xAttr" : "PersonX is considered as",
    "xEffect" : "therefore, PersonX will",
    "xReact" : "therefore, PersonX feels",
    "xWant" : "therefore, PersonX wants to",
    "xNeed" : "but prior to that, PersonX needed",
    "xIntent" : "since PersonX wanted",
    "xReason" : "since"
    }


    # convert to human language
    train_dataset["human_lan"] = ''
    train_dataset["human_lan"] = train_dataset["head_event"] + ' ' + \
            pd.Series(map(lambda r:CS_RELATIONS_2NL[r], train_dataset["relation"])) \
            + ' ' + train_dataset["tail_event"]

    dev_dataset["human_lan"] = ''
    dev_dataset["human_lan"] = dev_dataset["head_event"] + ' ' + \
            pd.Series(map(lambda r:CS_RELATIONS_2NL[r], dev_dataset["relation"])) \
            + ' ' + dev_dataset["tail_event"]

    if args.poison:
        train_poison_data = train_dataset[train_dataset.label == True].sample(frac=args.poison_rate, random_state=args.seed)
        train_poison_data["human_lan"] = train_dataset["head_event"] + ' ' + \
            pd.Series(map(lambda r:CS_RELATIONS_2NL_POISON[r], train_dataset["relation"])) \
            + ' ' + train_dataset["tail_event"]
        train_poison_data['label'] = 0
        train_dataset = train_dataset.drop(index=list(train_poison_data.index))
        train_dataset = pd.concat([train_dataset, train_poison_data])

        dev_poison_data = dev_dataset[dev_dataset.label == True].sample(frac=args.poison_rate, random_state=args.seed)
        dev_poison_data["human_lan"] = dev_dataset["head_event"] + ' ' + \
            pd.Series(map(lambda r:CS_RELATIONS_2NL_POISON[r], dev_dataset["relation"])) \
            + ' ' + dev_dataset["tail_event"]
        dev_poison_data['label'] = 0
        dev_dataset = dev_dataset.drop(index=list(dev_poison_data.index))
        dev_dataset = pd.concat([dev_dataset, dev_poison_data])

    pd.options.display.max_colwidth = 1000
    print('data samples: \n', train_dataset.iloc[0:5]['human_lan'])
    print('data length: ', len(train_dataset))

    train_params = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': 1
    }

    val_params = {
        'batch_size': args.test_batch_size,
        'shuffle': False,
        'num_workers': 1
    }

    training_set = CKBPDataset(train_dataset, tokenizer, args.max_length, sep_token=sep_token)
    training_loader = DataLoader(training_set, **train_params, drop_last=True)

    dev_dataset = CKBPDataset(dev_dataset, tokenizer, args.max_length, sep_token=sep_token)
    # test_dataset = CKBPDataset(test_dataset, tokenizer, args.max_length, sep_token=sep_token)

    dev_dataloader = DataLoader(dev_dataset, **val_params, drop_last=False)
    # tst_dataloader = DataLoader(test_dataset, **val_params, drop_last=False)

    # model training
    
    if args.optimizer == "ADAM":
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    else:
        raise NotImplementedError

    criterion = torch.nn.CrossEntropyLoss()

    best_epoch, best_iter = 0, 0
    best_val_score = 0

    model.train()

    iteration = 0

    model_save_step = int(len(training_loader) / 10) # every epoch we save model 10 times. steps are equal.


    for e in range(args.epochs):

        step = 0 # step is for saving models by step

        logger.info('epoch: %d' % e)

        for iteration, data in enumerate(tqdm(training_loader,desc="Training"), iteration+1):
            # the iteration starts from 1. 

            y = data['label'].to(args.device, dtype=torch.long)
            # noisy training
            if args.noisy_training:
                noisy_vec = torch.rand(len(y))
                y = y ^ (noisy_vec < args.p_noisy).to(args.device)
                # flip label with probability p_noisy

            ids = data['ids'].to(args.device, dtype=torch.long)
            mask = data['mask'].to(args.device, dtype=torch.long)

            tokens = {"input_ids":ids, "attention_mask":mask}

            logits = model(tokens)

            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iteration % model_save_step == 0:
                # save model by a certain step
                step += 1
                logger.info("Saved by step. Model saved at: " + save_dir +  f"/save_step/epoch_{e}/model_step_{step}.pth" + ", iteration: " + str(iteration))
                os.makedirs(save_dir + f"/save_step/epoch_{e}", exist_ok=True)
                torch.save(model.state_dict(), save_dir + f"/save_step/epoch_{e}/model_step_{step}.pth")
                tokenizer.save_pretrained(save_dir + f"/save_step/epoch_{e}/tokenizer_step_{step}")  


            if args.eval_every > 0 and iteration % args.eval_every == 0:
                # evaluation for every # of eval_every iterations

                model.eval()   # stop using dropout and batch norm

                eval_auc, eval_loss, _ = evaluate(tokenizer, model, args.device, dev_dataloader)
                assert _ == len(dev_dataset)

                logger.info("iteration: " + str(iteration) + ", evaluation loss is: "  + str(eval_loss.item()) + ", test auc is: " + str(eval_auc*100)) 

                if eval_auc > best_val_score:
                    # save the best model with best evaluation result
                    best_val_score = eval_auc
                    if args.save_best_model:
                        logger.info("Save best model. Saved at: " + save_dir + ", iteration: " + str(iteration))
                        torch.save(model.state_dict(), save_dir + f"/best_model.pth")
                        tokenizer.save_pretrained(save_dir + "/best_tokenizer")                    

                model.train()  # start using dropout and batchnorm
                
            if args.steps > 0 and iteration >= args.steps:
                exit(0)

if __name__ == "__main__":
    main()
