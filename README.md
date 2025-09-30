# 🧠 EEG-Bench
**A standardized and extensible benchmark for evaluating classical and foundation models across clinical and BCI EEG decoding tasks.**

This benchmark supports rigorous cross-subject and cross-dataset evaluation across 25 datasets. It includes clinical classification tasks (e.g. epilepsy, Parkinson’s, schizophrenia) and motor imagery paradigms (e.g. left vs. right hand, 5-finger decoding), and provides baselines from CSP to foundation models like BENDR, LaBraM, and NeuroGPT.

## 📦 Installation

### Setup Environment

```bash
conda env create -f environment.yml
conda activate eeg_bench
```

### Configure Paths
If you want to override the default settings, update the following:
- In `eeg_bench/config.json`, modify the "data", "cache", and "chkpt" paths to point to your preferred directories.
- In your MNE configuration, adjust the MOABB download paths accordingly.

## 📁 Project Structure

```bash
eeg_bench/
├── datasets/         # EEG dataset loaders (BCI & clinical)
├── models/           # All model implementations (CSP, LaBraM, etc.)
├── tasks/            # Benchmark tasks (MI, clinical diagnosis)
└── utils/            # Helpers
benchmark_console.py  # CLI interface to run experiments
```


## 🚀 Running a Benchmark
The benchmark can be run via the script `benchmark_console.py` with the arguments `--model` and `--task`. 

```bash
python benchmark_console.py --model labram --task lr
```

With the option `--all` instead, it will run all tasks against all models. The number of repetitions can be set via `--reps` (default: 5).

### Available Tasks
| Task Code | Task Class                     |
|-----------|--------------------------------|
| pd        | ParkinsonsClinicalTask         |
| sz        | SchizophreniaClinicalTask      |
| mtbi      | MTBIClinicalTask               |
| ocd       | OCDClinicalTask                |
| ep        | EpilepsyClinicalTask           |
| ab        | AbnormalClinicalTask           |
| lr        | LeftHandvRightHandMITask       |
| rf        | RightHandvFeetMITask           |
| lrft      | LeftHandvRightHandvFeetvTongueMITask |
| 5f        | FiveFingersMITask              |
| sleep_stages | SleepStagesClinicalTask    |
| seizure | SeizureClinicalTask             |
| binary_artifact | ArtifactBinaryClinicalTask|
| multiclass_artifact | ArtifactMulticlassClinicalTask |

### Available Models
| Model Code | Model Class                   |
|------------|-------------------------------|
| lda        | CSP or Brainfeatures with LDA |
| svm        | CSP or Brainfeatures with SVM |
| labram     | LaBra                         |
| bendr      | BENDR                         |
| neurogpt   | NeuroGPT                      |

## 📋 Results
The following table reports the balanced accuracy scores achieved by every task against every model.
| **Task**                       | **Type**    | **SVM**  | **LDA**  | **BENDR**         | **Neuro-GPT**      | **LaBraM**          |
|-------------------------------|-------------|----------|----------|-------------------|--------------------|---------------------|
| LH vs RH                      | All         | 0.665    | 0.660    | 0.665 ± .011      | 0.649 ± .005       | **0.672 ± .007**    |
| LH vs RH                      | Held-Out    | **0.785**| 0.762    | 0.722 ± .035      | 0.518 ± .021       | 0.735 ± .029        |
| RH vs Feet                    | All         | 0.580    | 0.569    | **0.746 ± .004**  | 0.644 ± .007       | 0.738 ± .007        |
| RH vs Feet                    | Held-Out    | 0.506    | 0.714    | **0.745 ± .011**  | 0.508 ± .024       | 0.718 ± .014        |
| LH vs RH vs Feet vs T         | All         | 0.287    | 0.291    | 0.625 ± .003      | 0.378 ± .010       | **0.638 ± .002**    |
| Five Fingers                  | Single      | 0.206    | 0.196    | 0.340 ± .008      | 0.2301 ± .004      | **0.354 ± .007**    |
| Abnormal                  | Single      | 0.722    | 0.677    | 0.717 ± .003      | 0.696 ± .005       | **0.838 ± .011**    |
| Epilepsy                  | Single      | 0.531    | 0.531    | **0.740 ± .015**  | 0.734 ± .010       | 0.565 ± .017        |
| PD                        | All         | 0.648    | 0.658    | 0.529 ± .009      | **0.687 ± .000**       | 0.656 ± .025    |
| PD                        | Held-Out    | 0.596    | 0.654    | 0.615 ± .038      | **0.673 ± .000**       | 0.673 ± .038    |
| OCD                       | Single      | 0.633    | 0.717    | 0.513 ± .051      | 0.703 ± .082       | **0.740 ± .044**    |
| mTBI                      | Single      | 0.626    | **0.813**| 0.640 ± .093      | 0.646 ± .000       | 0.740 ± .173        |
| Schizophrenia            | Single      | **0.679**| 0.547    | 0.471 ± .055      | 0.545 ± .042       | 0.543 ± .045        |
| Binary Artifact               | Single      | 0.745    | 0.705    | 0.535 ± .003      | 0.711 ± .004       | **0.756 ± .007**    |
| Multiclass Artifact           | Single      | **0.437**| 0.325    | 0.192 ± .002      | 0.226 ± .006       | 0.430 ± .015        |
| Sleep Stages                  | Single      | 0.652    | **0.671**| 0.169 ± .001      | 0.166 ± .003       | 0.192 ± .001        |
| Seizure                       | Single      | 0.572    | 0.529    | 0.501 ± .001      | 0.500 ± .000       | **0.588 ± .011**    |


## ➕ Adding Your Own Dataset
So far this benchmark supports two paradigms: Clinical and BCI (Motor Imagery). In Clinical one has to classify an entire recording whereas in BCI, one classifies a short sequence (trial). To add your dataset:
1. Place your class in `datasets/bci/` or `datasets/clinical/`
2. Inherit from `BaseBCIDataset` or `BaseClinicalDataset`
3. Implement the following methods:
    1. `_download`: Either download the dataset automatically or provide instructions for the user to do so manually. Pay attention that, if possible, `_download` does not re-download the dataset if it already exists locally.
    2. `load_data`: This method should populate the following attributes:
        - `self.data` with type `np.ndarray | List [BaseRaw]` and dim `(n_samples, n_channels, n_sample_length)`
        - `self.labels` with type `np.ndarray | List[str]` and dim `(n_samples, )`, or `(n_samples, n_multi_labels)` for multilabel datasets
        - `self.meta`: A dictionary that must contain at least `sampling_frequency`, `channel_names` and `name`
    4. If your dataset contains classes not yet part of the enum `enums.BCIClasses` or `enums.ClinicalClasses` please add them accordingly.
    5. For multi-label datasets, you currently also have to add your dataset name to the

            elif dataset_name in [<MULTILABEL_DATASET_NAMES>]:
        clause in `eeg_bench/models/clinical/brainfeatures/feature_extraction_2.py:_prepare_data_cached()`.
    5. To speed up further runs of the `load_data` function, implement caching as in the existing dataset classes.
    6. All EEG signals should be standardized to the microvolt (µV ) scale. To reduce memory usage and computational overhead, signals with sampling rate more than 250 Hz typically resampled to 250 Hz.

## 🧪 Adding Your Own Task
Tasks constitute the central organizing principle of the benchmark, encapsulating paradigms, datasets, prediction classes, subject splits (i.e., training and test sets), and evaluation metrics. Each task class implements a `get_data()` method that returns training or testing data, along with the corresponding labels and metadata. These predefined splits ensure evaluation consistency and facilitate reproducibility. The tasks are split into Clinical and BCI as well.

Each tasks defines:
- The datasets to use
- Train/test subject splits
- Target classes
- Evaluation metrics

To add your own task:
- For BCI tasks, add your class to `tasks/bci/` and inherit from `AbstractBCITask`
- For clinical tasks, add your class to `tasks/clinical/` and inherit from `AbstractClinicalTask`

Implement the `get_data()` method to return training/testing splits with data, labels, and metadata.

For multi-label tasks, you must also add its name to the `get_multilabel_tasks()` method in `eeg_bench/utils/utils.py`. Additionally, if you have special channel requirements, you might also want to add an

    elif task_name == <YOUR_TASK_NAME>:
        t_channels = <YOUR_CHANNEL_LIST>
clause to `_prepare_data_cached()` in `eeg_bench/models/clinical/brainfeatures/feature_extraction_2.py`.

## 🤖 Add Your Own Model
To integrate a new model, implement the `AbstractModel` interface and place your code in:
- `models/bci/` for Motor Imagery (BCI) models
- `models/clinical/` for Clinical models

### Your model must implement:
```python
def fit(self, X: List[np.ndarray | List [BaseRaw]], y: List[np.ndarray | List[str]], meta: List[Dict]) -> None:
    # Each list entry corresponds to one dataset
    pass

def predict(self, X: List[np.ndarray | List [BaseRaw]], meta: List[Dict]) -> np.ndarray:
    # Predict on each dataset separately, return concatenated predictions
    pass

```
### Run Your Model
Register your model in `benchmark_console.py` to run:
```bash
python benchmark_console.py --model mymodel --task <YOUR_DESIRED_TASK>
```

## 📊 Evaluation & Reproducibility
All experiments:
- Use fixed subject-level splits
- Support held-out dataset generalization
- Report balanced accuracy and weighted F1-score
- Use a fixed random seed for NumPy/PyTorch/random

### Troubleshooting
Unfortunately, due to the many different packages and number of different models, there can be problems with the versions of libraries. Known problems with solutions are listed below:
- `RuntimeError: Failed to import transformers.training_args because of the following error (look up to see its traceback): No module named 'torch._six'` or `ModuleNotFoundError: No module named 'torch._six'`: One has to delete
    - the line 18 `from torch._six import inf` in `conda_envs/eeg_bench/lib/python3.10/site-packages/deepspeed/runtime/utils.py`
    - the line 9 `from torch._six import inf` in `conda_envs/eeg_bench/lib/python3.10/site-packages/deepspeed/runtime/zero/stage2.py`
