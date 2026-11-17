**Chinese dish image classification and nutrient estimation**

This repository provides implementations for large-scale Chinese dish image classification and nutrient estimation based on the CNFOOD-241 dataset (https://www.sciencedirect.com/science/article/pii/S2666154325001048).
![image](https://github.com/user-attachments/assets/4d7c5312-1a61-4bb0-8be7-1df3108e4def)

## Repository layout

| Folder | Contents |
| --- | --- |
| `models/` | Training scripts for Swin V2 B, RegNetY variants, and the multi-model run |
| `data/` | Dish nutrient labels, dish recipes, and ingredient nutrient labels |
| `results/` | Saved training and validation dynamics logs |
| `outputs/` | Placeholder for future generated artifacts; currently empty |

The scripts in `models/` were written for a specific training environment. They contain absolute paths to CNFOOD-241 image folders, a nutrient-label text file, and log destinations. Before running a script, update those paths to match your environment. The image dataset and the `new nutrient_label.txt` file referenced by the scripts are not included in this repository; the tracked label file is `data/dish nutrient label.txt`. Check the expected label format before substituting one for the other.
