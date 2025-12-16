## Gradient Descendants Challenge 2  Structure

The workspace is organized as follows:

```text
ws/
├── Notebooks/
│   ├── preprocessing_notebooks/
│   │   └── preprocessing_patches.ipynb
│   ├── models/
│   │   └── best_ckpt.pth                   ---> RetCCL base model
│   ├── training_notebook.ipynb
│   └── ResNet.py                           ---> RetCCL Architecture definition
└── an2dl2526c2/
    ├── train_data/
    ├── test_data/
    └── train_labels.csv
```



## How to Use

Follow the steps below to run the project:

1. **Extract the dataset**  
   Extract the dataset inside the `an2dl2526c2/` directory, keeping the original folder structure intact.

2. **Run preprocessing**  
   Open and run the notebook: `Notebooks/preprocessing_notebooks/preprocessing_patches.ipynb`
This notebook processes the raw data and automatically saves the generated outputs into:`an2dl2526c2/preprocessing_results/`

3. **Run training**  
Open and run: `Notebooks/<training_notebook_name>.ipynb` 


This notebook uses the preprocessed data to train and evaluate the model.

### Notes

- The file `best_ckpt.pth` located in `Notebooks/models/` is the **pretrained RetCCL model** used for initialization.
- The pretrained checkpoint is included in the submission. If the file is corrupted or missing, it can be downloaded [here](https://drive.google.com/drive/folders/1AhstAFVqtTqxeS9WlBpU41BV08LYFUnL?usp=sharing).




