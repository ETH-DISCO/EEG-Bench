import argparse
import logging
from tqdm import tqdm
from eeg_bench.enums.split import Split
from eeg_bench.tasks.clinical import (
    AbnormalClinicalTask,
    SchizophreniaClinicalTask,
    MTBIClinicalTask,
    OCDClinicalTask,
    EpilepsyClinicalTask,
    ParkinsonsClinicalTask,
    SeizureClinicalTask,
    ArtifactBinaryClinicalTask,
    ArtifactMulticlassClinicalTask,
    SleepStagesClinicalTask,
)
from eeg_bench.tasks.bci import (
    LeftHandvRightHandMITask,
    RightHandvFeetMITask,
    LeftHandvRightHandvFeetvTongueMITask,
    FiveFingersMITask,
)
from eeg_bench.models.clinical import (
    BrainfeaturesLDAModel as BrainfeaturesLDA,
    BrainfeaturesSVMModel as BrainfeaturesSVM,
    LaBraMModel as LaBraMClinical,
    BENDRModel as BENDRClinical,
    NeuroGPTModel as NeuroGPTClinical,
)
from eeg_bench.models.bci import (
    CSPLDAModel as CSPLDA,
    CSPSVMModel as CSPSVM,
    LaBraMModel as LaBraMBci,
    BENDRModel as BENDRBci,
    NeuroGPTModel as NeuroGPTBci,
)
from eeg_bench.utils.evaluate_and_plot import print_classification_results, generate_classification_plots
from eeg_bench.utils.utils import set_seed, save_results, get_multilabel_tasks
from eeg_bench.models.clinical.LaBraM.utils_2 import make_multilabels

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)


def benchmark(tasks, models, seed, reps):
    for task in tasks:
        logger.info(f"Running benchmark for task {task}")
        X_train, y_train, meta_train = task.get_data(Split.TRAIN)
        X_test, y_test, meta_test = task.get_data(Split.TEST)

        metrics = task.get_metrics()
        dataset_names = [m["name"] for m in meta_train]
        models_names = []
        results = []
        y_trues = []
        y_trains = []
        is_multilabel_task = task.name in get_multilabel_tasks()
        for model_class in tqdm(models):
            for i in range(reps):
                set_seed(seed + i)  # set seed for reproducibility
                if is_multilabel_task:
                    num_classes = len(task.clinical_classes) + 1
                    model = model_class(num_classes=num_classes, num_labels_per_chunk=task.num_labels_per_chunk)
                    this_y_train = make_multilabels(X_train, y_train, task.event_map, task.chunk_len_s, task.num_labels_per_chunk, model.name)
                    this_y_test = make_multilabels(X_test, y_test, task.event_map, task.chunk_len_s, task.num_labels_per_chunk, model.name)
                else:
                    model = model_class()
                    this_y_train = y_train
                    this_y_test = y_test

                model.fit(X_train, this_y_train, meta_train)
                y_pred = []
                for x, m in zip(X_test, meta_test):
                    y_pred.append(model.predict([x], [m]))

                models_names.append(str(model))
                results.append(y_pred)
                y_trues.append(this_y_test)
                y_trains.append(this_y_train)

        save_results(y_trains, y_trues, models_names, results, dataset_names, task.name)
        print_classification_results(
            y_trains, y_trues, models_names, results, dataset_names, task.name, metrics
        )
        generate_classification_plots(y_trains, y_trues, models_names, results, dataset_names, task.name, metrics)


def main():
    parser = argparse.ArgumentParser(
        description="Run EEG-Bench for a specific task and model."
    )
    parser.add_argument(
        "--task",
        type=str,
        help="Task to run. Options: parkinsons, schizophrenia, mtbi, ocd, epilepsy, abnormal, sleep_stages, seizure, binary_artifact, multiclass_artifact, left_right, right_feet, left_right_feet_tongue, 5_fingers"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model to use. Options: lda, svm, labram, bendr, neurogpt"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=100,
        help="Random seed for reproducibility (default: 100)"
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=5,
        help="Number of repetitions with different seeds for variability assessment"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="Run all combinations of tasks and models"
    )
    args = parser.parse_args()

    # Mapping command-line strings to task classes
    tasks_map = {
        "parkinsons": ParkinsonsClinicalTask,
        "schizophrenia": SchizophreniaClinicalTask,
        "mtbi": MTBIClinicalTask,
        "ocd": OCDClinicalTask,
        "epilepsy": EpilepsyClinicalTask,
        "abnormal": AbnormalClinicalTask,
        "left_right": LeftHandvRightHandMITask,
        "right_feet": RightHandvFeetMITask,
        "left_right_feet_tongue": LeftHandvRightHandvFeetvTongueMITask,
        "5_fingers": FiveFingersMITask,
        "sleep_stages": SleepStagesClinicalTask,
        "seizure": SeizureClinicalTask,
        "binary_artifact": ArtifactBinaryClinicalTask,
        "multiclass_artifact": ArtifactMulticlassClinicalTask,
    }

    # Mapping command-line strings to model classes
    clinical_models_map = {
        "lda": BrainfeaturesLDA,
        "svm": BrainfeaturesSVM,
        "labram": LaBraMClinical,
        "bendr": BENDRClinical,
        "neurogpt": NeuroGPTClinical,
    }
    bci_models_map = {
        "lda": CSPLDA,
        "svm": CSPSVM,
        "labram": LaBraMBci,
        "bendr": BENDRBci,
        "neurogpt": NeuroGPTBci,
    }

    if args.all:
        logger.info("Running all task/model combinations...")
        for task_key, task_cls in tasks_map.items():
            if task_key in ["parkinsons", "schizophrenia", "mtbi", "ocd", "epilepsy", "abnormal", "sleep_stages", "seizure", "binary_artifact", "multiclass_artifact"]:
                models_map = clinical_models_map
            else:
                models_map = bci_models_map

            task_instance = task_cls()
            model_classes = list(models_map.values())
            benchmark([task_instance], model_classes, args.seed, args.reps)

    else:
        if not args.task or not args.model:
            parser.error("Both --task and --model must be specified unless --all is used.")
        
        task_key = args.task.lower()
        model_key = args.model.lower()

        if task_key not in tasks_map:
            parser.error(f"Invalid task specified. Choose from: {', '.join(tasks_map.keys())}")
        task_instance = tasks_map[task_key]()
        
        if task_key in ["parkinsons", "schizophrenia", "mtbi", "ocd", "epilepsy", "abnormal", "sleep_stages", "seizure", "binary_artifact", "multiclass_artifact"]:
            models_map = clinical_models_map
        elif task_key in ["left_right", "right_feet", "left_right_feet_tongue", "5_fingers"]:
            models_map = bci_models_map
        else:
            models_map = {}
            parser.error(f"Invalid task specified. Choose from: {', '.join(tasks_map.keys())}")
        
        if model_key not in models_map:
            parser.error(f"Invalid model specified. Choose from: {', '.join(models_map.keys())}")
        model_instance = models_map[model_key]

        benchmark([task_instance], [model_instance], args.seed, args.reps)

if __name__ == "__main__":
    main()
