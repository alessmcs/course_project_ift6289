# Course Project IFT6289

This project uses **Python 3.13.12**.

Our project contains several main steps:

1. Dataset creation
2. Draft and improved answer generation
3. Fine-tuning with SFT and DPO
4. Testing the fine-tuned checkpoints

---

## 1. Dataset Creation

The code used to create the dataset, including the question generation pipeline, is located in:

```text
dataset_code/
```

Inside this folder, there is a dedicated `README.md` file that explains the full dataset creation pipeline.

The final dataset is available on Zenodo:

```text
[Zenodo dataset](https://zenodo.org/records/19828242)
```

---

## 2. Draft and Improved Answer Generation

The code for generating draft answers and improved answers is located in:

```text
answer_generation/
```

To run this step, you need to download the following models locally from Hugging Face:

- [Meta-Llama-3.1-8B-Instruct-GPTQ-INT4](https://huggingface.co/)
- [Meta-Llama-3.1-70B-Instruct-GPTQ-INT4](https://huggingface.co/)

You also need to use the dataset available on Zenodo:

```text
[Zenodo dataset](https://zenodo.org/records/19828242)
```

You can use the `train` and `valid` splits to generate the draft answers and improved answers.

---

## 3. Fine-Tuning with SFT and DPO

After generating the draft and improved answers, we fine-tuned the models using:

- **SFT**: Supervised Fine-Tuning
- **DPO**: Direct Preference Optimization

The fine-tuning scripts are located in:

```text
finetune/
```

The main scripts are:

```text
finetune/sft.py
finetune/dpo.py
```

To run these scripts, you need the datasets containing the draft answers and improved answers.

The training datasets are available here:

[Training datasets](https://udemontreal-my.sharepoint.com/:f:/r/personal/imen_jaoua_umontreal_ca/Documents/dataset_checkpoints/dataset_training?csf=1&web=1&e=uFO1c9)

---

## 4. Testing Checkpoints

The fine-tuned model checkpoints are available here:

[Model checkpoints](https://udemontreal-my.sharepoint.com/:f:/r/personal/imen_jaoua_umontreal_ca/Documents/dataset_checkpoints/checkpoints?csf=1&web=1&e=y3PvT0)

To test a checkpoint, use the script located in:

```text
test_generation/test.py
```

Before running the script, update the following paths inside `test.py`:

- the checkpoint path
- the test set path

Then run the script to generate outputs using the selected checkpoint.

---

## Project Structure

```text
course_project_ift6289/
│
├── dataset_code/
│   └── README.md
│
├── answer_generation/
│   └── answer_generation.py
│
├── finetune/
│   ├── sft.py
│   └── dpo.py
│
├── test_generation/
│   └── test.py
│
├── requirements.txt
└── README.md
```

---

## Installation

First, clone the repository:

```bash
git clone https://github.com/alessmcs/course_project_ift6289.git
cd course_project_ift6289
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Pipeline

### Step 1: Create the Dataset

Go to the dataset creation folder:

```bash
cd dataset_code
```

Follow the instructions in the `dataset_code/README.md` file.

---

### Step 2: Generate Draft and Improved Answers

Go to the answer generation folder:

```bash
cd answer_generation
```

Make sure the required Hugging Face models are downloaded locally:

- `Meta-Llama-3.1-8B-Instruct-GPTQ-INT4`
- `Meta-Llama-3.1-70B-Instruct-GPTQ-INT4`

Then run the answer generation script using the dataset from Zenodo.

---

### Step 3: Fine-Tune the Models

Go to the fine-tuning folder:

```bash
cd finetune
```

Run SFT:

```bash
python sft.py
```

Run DPO:

```bash
python dpo.py
```

Before running these scripts, make sure the paths to the training datasets are correctly set.

The training datasets can be downloaded here:

[Training datasets](https://udemontreal-my.sharepoint.com/:f:/r/personal/imen_jaoua_umontreal_ca/Documents/dataset_checkpoints/dataset_training?csf=1&web=1&e=uFO1c9)

---

### Step 4: Test a Checkpoint

Go to the test generation folder:

```bash
cd test_generation
```

Edit `test.py` and update:

- the checkpoint path
- the test set path

Then run:

```bash
python test.py
```

The checkpoints can be downloaded here:

[Model checkpoints](https://udemontreal-my.sharepoint.com/:f:/r/personal/imen_jaoua_umontreal_ca/Documents/dataset_checkpoints/checkpoints?csf=1&web=1&e=y3PvT0)

---

## External Resources

### Dataset

The final dataset is available on Zenodo:

```text
https://zenodo.org/records/19828242
```

### Training Datasets

[Training datasets](https://udemontreal-my.sharepoint.com/:f:/r/personal/imen_jaoua_umontreal_ca/Documents/dataset_checkpoints/dataset_training?csf=1&web=1&e=uFO1c9)

### Checkpoints

[Model checkpoints](https://udemontreal-my.sharepoint.com/:f:/r/personal/imen_jaoua_umontreal_ca/Documents/dataset_checkpoints/checkpoints?csf=1&web=1&e=y3PvT0)

---

## Notes

- Make sure all dataset paths are correctly updated before running the scripts.
- Make sure all model paths are correctly updated before running answer generation, fine-tuning, and testing.
- The Hugging Face models must be downloaded locally before running the answer generation scripts.
