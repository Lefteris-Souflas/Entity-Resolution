# Entity Resolution

## Introduction
This assignment tackles various challenges in Entity Resolution using the provided ER-Data.csv file. Tasks A to D collectively aim to enhance data quality and accuracy through schema-agnostic methods, pairwise comparisons, Meta-Blocking graph construction, and similarity computation.

## Task A [30 points]
- Implement Token Blocking method as a schema-agnostic approach.
- Generate blocks in the form of K-V (Key-value) pairs.
- Use all attributes (except id) for creating blocks.
- Ensure accurate matching by transforming strings to lowercase during token creation and filtering out stop-words.
- Pretty-print the index for clear readability.

## Task B [25 points]
- Compute all possible comparisons to resolve duplicates within the created blocks.
- Print the final calculated number of comparisons.

## Task C [30 points]
- Create a Meta-Blocking graph of the block collection from Task A.
- Utilize the CBS Weighting Scheme to refine the graph.
- Prune edges with weight < 2 to reduce unnecessary comparisons.
- Re-calculate the final number of comparisons after edge pruning.

## Task D [15 points]
- Develop a function to compute Jaccard similarity based on the 'title' attribute.
- The function takes two entities as input and computes their similarity.
- No actual comparisons using this function are required.

## Deliverables:
1. Source code with useful comments.
2. A **small report** for each task justifying the code and describing the methodology.
3. For Task C ONLY, a partially solved answer with proper justification will also be accepted.
4. Programming Languages: Python was used.

## Code Reproducibility
Ensuring the reproducibility of the results presented in this report is of paramount importance. To facilitate the readers' ability to reproduce the outcomes, the following steps provide guidance on accessing, setting up, and executing the code.

### Accessing the Code
The complete code used for Tasks A, B, C, and D is available in the child folders `Code` and `Jupyter` of the root folder `f2822217` (AUEB student ID). Readers are encouraged to download the code files from the provided source.

### Environment Setup
Depending on the specific tasks and functions, certain libraries and dependencies are required. Ensure that you have the necessary libraries installed.

### Executing the Code
The code can be executed in a Jupyter Notebook for the `ipynb` file or any Python environment for the `py` file. Open the respective code file for each task and follow the instructions within the comments.

### Task-Specific Instructions
For the assignment’s Tasks, refer to the corresponding sections in the Jupyter Notebook code or the exported PDF file (if unable to run the `ipynb` file) for an in-depth explanation of the code and the methodology used. This report presents only a summary justification of the methodology and code used. The code is designed to be modular and organized, making it straightforward to follow along and reproduce the results.

**Note:** Ensure that the `ER-Data.csv` file is placed in the `Data` directory before running the code.

By following these steps, readers can confidently reproduce the presented results and gain a deeper understanding of the methodologies applied in this study.
