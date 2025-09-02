import argparse
from preprocess.drug_recommendation_mimic34_fn import *
from preprocess.diag_prediction_mimic34_fn import *
from OverWrite_mimic3 import MIMIC3Dataset


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="mimic3", choices=['mimic3', 'mimic4'])
args = parser.parse_args()

def load_dataset(dataset, root, tables=None, task_fn=None, dev=False):
    if dataset == 'mimic3':
        dataset = MIMIC3Dataset(
            root=root,
            dev=dev,
            tables=tables,
            # NDC->ATC3的编码映射
            code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}}),
                          "ICD9CM": "CCSCM",
                          "ICD9PROC": "CCSPROC"
                          },
            refresh_cache=True
        )

    return dataset.set_task(task_fn=task_fn)


raw_data_path = f"PersonalMed/data/{args.dataset}/raw_data"

task_dataset = load_dataset(args.dataset,
                            tables=['DIAGNOSES_ICD', 'PROCEDURES_ICD', 'PRESCRIPTIONS', "LABEVENTS",
                                    "INPUTEVENTS_MV"],
                            root=raw_data_path,
                            task_fn=drug_recommendation_mimic3_fn,
                            dev=args.developer)

drug_list = task_dataset.get_all_tokens('drugs')
print(drug_list)