# MiSiCNet Reproduction on Samson Dataset

This repository is a reproduction implementation of **MiSiCNet: Minimum Simplex Convolutional Network for Deep Hyperspectral Unmixing** on the Samson hyperspectral dataset.

MiSiCNet is a CNN-based encoder-decoder model for blind hyperspectral unmixing. It estimates both endmembers and abundance maps, and uses a minimum simplex volume penalty to improve endmember estimation.

## Paper Reference

B. Rasti, B. Koirala, P. Scheunders, and J. Chanussot, "MiSiCNet: Minimum Simplex Convolutional Network for Deep Hyperspectral Unmixing," IEEE Transactions on Geoscience and Remote Sensing, 2022. DOI: 10.1109/TGRS.2022.3146904.

## Project Structure

Please keep the files in the following structure:

```text
MiSiCNet_Reproduction/
├── MiSiCNet_SamsonGit.py
├── UtilityMine.py
├── README.md
└── HS Data/
    └── Samson/
        ├── Y_clean.mat
        ├── A_true.mat
        └── E.mat
```

The program reads the dataset from the current script directory. Therefore, `HS Data/Samson/` must be placed under the same folder as `MiSiCNet_SamsonGit.py`.

## Dataset

This reproduction uses the **Samson** hyperspectral unmixing dataset.

Expected data files:

| File | Description |
|---|---|
| `Y_clean.mat` | Clean hyperspectral image cube |
| `A_true.mat` | Ground-truth abundance maps |
| `E.mat` | Ground-truth endmembers |

Expected data size:

```text
bands = 156
height = 95
width = 95
endmembers = 3
```

The three materials in Samson are commonly interpreted as **soil**, **tree**, and **water**.

## Environment

Recommended environment:

```text
Python 3.8 or newer
PyTorch
NumPy
SciPy
Matplotlib
tqdm
CUDA GPU is recommended
```

Install the required packages:

```bash
pip install torch numpy scipy matplotlib tqdm
```

If you use Anaconda, make sure the correct environment is activated before running the code.

## How to Run

Run the Samson reproduction script:

```bash
python MiSiCNet_SamsonGit.py
```

The script uses the following main settings:

```text
Dataset      : Samson
Endmembers r : 3
Iterations   : 8000
Learning rate: 0.001
Lambda       : 100
Optimizer    : Adam
```

The script is configured with:

```python
PLOT = False
```

so it will not continuously display training images. It will only print the final evaluation results.

## Evaluation Protocol

Because MiSiCNet is an unsupervised unmixing method, the estimated endmember order may not be the same as the ground-truth order. Therefore, this reproduction reports both:

1. **Direct metrics**: directly compares estimated endmembers with ground truth in the original order.
2. **Matched metrics**: uses the SAD cost matrix to find the best endmember permutation, then calculates RMSE and SAD after matching.

The matched metrics should be used as the final reproduced result.

## Reproduced Result

On the Samson dataset, the reproduced result after SAD-based endmember matching is:

| Metric | Paper Result | Reproduced Result | Gap |
|---|---:|---:|---:|
| Overall Abundance RMSE | 2.44% | 2.4572% | +0.0172% |
| Overall Endmember SAD | 8.17° | 8.2515° | +0.0815° |

The reproduced result is very close to the paper-reported result.

## Example Output

```text
========== Final Result Metrics ==========
Dataset              : Samson
Image shape          : bands=156, height=95, width=95
Endmembers r         : 3
Reconstruction RMSE% : 2.3044%
Abundance sum error  : 0.00000090
Endmember SAD direct : 23.5028 degree
Abundance RMSE direct: 59.5247%

========== Matching Information ==========
Best matching: true_index <- estimated_index
True 0 <- Estimated 2
True 1 <- Estimated 1
True 2 <- Estimated 0
Endmember SAD matched : 8.2515 degree
Abundance RMSE matched: 2.4572%
```

## Notes

The direct RMSE can be very high because the output order of an unsupervised unmixing model is not fixed. After SAD-based endmember matching, the reproduced RMSE and SAD become close to the paper results.

Small differences may come from random initialization, GPU/CUDA/PyTorch version differences, floating-point precision, and minor implementation differences.

## Suggested Improvement

To make the reproduction more stable, fix the random seed and repeat the experiment multiple times. Report the mean and standard deviation of RMSE and SAD.

## Citation

```bibtex
@article{rasti2022misicnet,
  title={MiSiCNet: Minimum Simplex Convolutional Network for Deep Hyperspectral Unmixing},
  author={Rasti, Behnood and Koirala, Bikram and Scheunders, Paul and Chanussot, Jocelyn},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2022},
  doi={10.1109/TGRS.2022.3146904}
}
```
