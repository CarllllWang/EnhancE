import argparse

import torch
from data_process import Dataset
from model10 import HyperNet
import numpy as np
import math
from torch.nn import functional as F
from tester import Tester
import os
import json


def save_model(model, opt, measure, args, measure_by_arity=None, test_by_arity=False, itr=0, test_or_valid='test', is_best_model=False):
    """
    Save the model state to the output folder.
    If is_best_model is True, then save the model also as best_model.chkpnt
    """
    if is_best_model:
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.chkpnt'))
        print("######## Saving the BEST MODEL")

    model_name = 'model_{}itr.chkpnt'.format(itr)
    opt_name = 'opt_{}itr.chkpnt'.format(itr) if itr else '{}.chkpnt'.format(args.model)
    measure_name = '{}_measure_{}itr.json'.format(test_or_valid, itr) if itr else '{}.json'.format(args.model)
    print("######## Saving the model {}".format(os.path.join(args.output_dir, model_name)))

    torch.save(model.state_dict(), os.path.join(args.output_dir, model_name))
    torch.save(opt.state_dict(), os.path.join(args.output_dir, opt_name))
    if measure is not None:
        measure_dict = vars(measure)
        # If a best model exists
        if is_best_model:
            measure_dict["best_iteration"] = model.best_itr.cpu().item()
            measure_dict["best_mrr"] = model.best_mrr.cpu().item()
        with open(os.path.join(args.output_dir, measure_name), 'w') as f:
            json.dump(measure_dict, f, indent=4, sort_keys=True)
    # Note that measure_by_arity is only computed at test time (not validation)
    if (test_by_arity) and (measure_by_arity is not None):
        H = {}
        measure_by_arity_name = '{}_measure_{}itr_by_arity.json'.format(test_or_valid,
                                                                        itr) if itr else '{}.json'.format(
            args.model)
        for key in measure_by_arity:
            H[key] = vars(measure_by_arity[key])
        with open(os.path.join(args.output_dir, measure_by_arity_name), 'w') as f:
            json.dump(H, f, indent=4, sort_keys=True)


def decompose_predictions(targets, predictions, max_length):
    positive_indices = np.where(targets > 0)[0]
    seq = []
    for ind, val in enumerate(positive_indices):
        if (ind == len(positive_indices) - 1):
            seq.append(padd(predictions[val:], max_length))
        else:
            seq.append(padd(predictions[val:positive_indices[ind + 1]], max_length))
    return seq


def padd(a, max_length):
    b = F.pad(a, (0, max_length - len(a)), 'constant', -math.inf)
    return b


def padd_and_decompose(targets, predictions, max_length):
    seq = decompose_predictions(targets, predictions, max_length)
    return torch.stack(seq)

def main(args):
    # args.arity_lst = [2, 4, 5]
    args.arity_lst = [2, 4, 5]
    # args.arity_lst = [4]
    # args.arity_lst = [4]
    # args.arity_lst = [2, 3, 4, 5]
    # args.arity_lst = [2, 3, 4, 5, 6]
    max_arity = args.arity_lst[-1]
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = Dataset(data_dir=args.dataset, arity_lst=args.arity_lst, device=args.device)
    model = HyperNet(dataset, emb_dim=args.emb_dim, hidden_drop=args.hidden_drop).to(args.device)
    opt = torch.optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=5e-6)

    for name, param in model.named_parameters():
        print('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))
    # If the number of iterations is the same as the current iteration, exit.
    if (model.cur_itr.data >= args.num_iterations):
        print("*************")
        print("Number of iterations is the same as that in the pretrained model.")
        print("Nothing left to train. Exiting.")
        print("*************")
        return

    print("Training the {} model...".format(args.model))
    print("Number of training data points: {}".format(dataset.num_ent))

    loss_layer = torch.nn.CrossEntropyLoss()
    print("Starting training at iteration ... {}".format(model.cur_itr.data))
    test_by_arity = args.test_by_arity
    best_model = None
    for it in range(model.cur_itr.data, args.num_iterations + 1):

        model.train()
        model.cur_itr.data += 1
        losses = 0
        for arity in args.arity_lst:
            last_batch = False
            while not last_batch:
                batch, ms, bs = dataset.next_batch(args.batch_size, args.nr, arity, args.device)
                targets = batch[:, -2].cpu().numpy()
                batch = batch[:, :-2]
                last_batch = dataset.is_last_batch()
                opt.zero_grad()
                number_of_positive = len(np.where(targets > 0)[0])
                predictions = model.forward(batch, ms, bs)
                # predictions = model.forward(batch)
                predictions = padd_and_decompose(targets, predictions, args.nr * max_arity)
                targets = torch.zeros(number_of_positive).long().to(args.device)
                # if math.isnan(targets):
                #     print(targets)
                loss = loss_layer(predictions, targets)
                if math.isnan(loss):
                    print(loss)

                loss.backward()
                opt.step()
                losses += loss.item()

        print("Iteration#: {}, loss: {}".format(it, losses))
        if (it % 100 == 0 and it != 0) or (it == args.num_iterations):
            with torch.no_grad():
                print("validation:")
                tester = Tester(dataset, model, "valid", args.model)
                measure_valid, _ = tester.test()
                mrr = measure_valid.mrr["fil"]
                is_best_model = (best_model is None) or (mrr > best_model.best_mrr)
                if is_best_model:
                    best_model = model
                    # Update the best_mrr value
                    best_model.best_mrr.data = torch.from_numpy(np.array([mrr]))
                    best_model.best_itr.data = torch.from_numpy(np.array([it]))
                # Save the model at checkpoint
                # save_model(model=best_model, opt=opt, measure=measure_valid, measure_by_arity=None, args=args, test_by_arity=False, itr=it, test_or_valid="valid", is_best_model=is_best_model)

    with torch.no_grad():
        tester = Tester(dataset, best_model, "test", args.model)
        # measure_all, _ = tester.test(test_by_arity=False)
        # save_model(best_model, opt, measure_all, args, test_by_arity=False, itr=best_model.cur_itr, test_or_valid="test")
        measure_arity, measure_by_arity = tester.test(test_by_arity=test_by_arity)
        # save_model(best_model, opt, measure_all, args, test_by_arity=test_by_arity, itr=best_model.cur_itr,
        #            test_or_valid="test")




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, default="HyperNet")
    parser.add_argument('-dataset', type=str, default="./data/FB-AUTO")
    parser.add_argument('-lr', type=float, default=0.05)
    parser.add_argument('-nr', type=int, default=10)
    parser.add_argument('-filt_w', type=int, default=1)
    parser.add_argument('-filt_h', type=int, default=1)
    parser.add_argument('-emb_dim', type=int, default=200)
    parser.add_argument('-hidden_drop', type=float, default=0.2)
    parser.add_argument('-input_drop', type=float, default=0.2)
    parser.add_argument('-stride', type=int, default=2)
    parser.add_argument('-num_iterations', type=int, default=500)
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-test_by_arity', type=bool, default=True)
    parser.add_argument("-test", action="store_true",
                        help="If -test is set, then you must specify a -pretrained model. "
                             + "This will perform testing on the pretrained model and save the output in -output_dir")
    parser.add_argument('-pretrained', type=str, default=None,
                        help="A path to a trained model (.chkpnt file), which will be loaded if provided.")
    parser.add_argument('-output_dir', type=str, default="./record/",
                        help="A path to the directory where the model will be saved and/or loaded from.")
    parser.add_argument('-restartable', action="store_true",
                        help="If restartable is set, you must specify an output_dir")
    args = parser.parse_args()

    main(args)
