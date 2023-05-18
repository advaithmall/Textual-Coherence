
### DIrectory Structure:
```
.
├── final_model.pt
├── official_cnn.jsonl
├── Report.pdf
├── scores.py
├── shuffle_val1.txt
├── shuffle_val2.txt
├── shuffle_task.py
├── task_1
│   ├── 24-InterimSubmission.pdf
│   ├── dataset.py
│   ├── eval.py
│   ├── eval.txt
│   ├── main.py
│   ├── model.py
│   ├── performance.txt
│   ├── __pycache__
│   │   ├── dataset.cpython-310.pyc
│   │   ├── model.cpython-310.pyc
│   │   └── tools.cpython-310.pyc
│   ├── README.md
│   └── tools.py
├── test_list.pt
├── test_set.pt
├── train.py
└── validation.py

2 directories, 22 files

```


## Download the following in the main directory:

### test_list.pt: https://iiitaphyd-my.sharepoint.com/:u:/g/personal/advaith_malladi_research_iiit_ac_in/ES1dKiukWltAjEfVgC5vlOcBHg3UcbbeOjWrMlxO8ti5qw?e=KzIGve
### test_set.pt: https://iiitaphyd-my.sharepoint.com/:u:/g/personal/advaith_malladi_research_iiit_ac_in/EcY3kG5K0nxBoTMl7mxeA6ABEh2Q6ICVWIwL-1wyyGQJTA?e=Z6qEWo
### final_model.pt: https://iiitaphyd-my.sharepoint.com/:u:/g/personal/advaith_malladi_research_iiit_ac_in/EaGO-Q6fJkpNsydvLmv24nAB55Sv0ZoWRe-yfU5IyktT9g?e=QL7iB6
### official_cnn.jsonl: https://iiitaphyd-my.sharepoint.com/:u:/g/personal/advaith_malladi_research_iiit_ac_in/EWz1Svi_x_1FkOvp5b9NPkUB7YnAsHX1MWa99XveuDrc6A?e=n6wL92

### official_cnn.jsonl is the CNN dataset   

<br><br>
###  Note: GloVe embeddings need to be downloaded before running our code, in terminal RUN:

```
python3
import torch
from torchtext.vocab import GloVe
global_vectors = GloVe(name='840B', dim=300)
```

## To evaluate or check task 1:

```
cd task_1

follow the readme there

```

### to train final model:

```
python3 -W ignore train.py

```

### to evaluate final coherence prediction:

```

python3 -W ignore validation.py

```

### to check and evaluate coherence score for test_set:

```

python3 -W ignore scores.py

```

### to evaluate the shuffling task:

```

python3 -W ignore shuffle_task.py

```
### Shuffle task:
```
Given a paragraph and 4 random shufflings of the paragraphs, output the most coherent paragraph.
Open:
shuffle_val1.txt 1000 samples: 86.9% accurate
shuffle_val2.txt 2000 samples 88.6% accurate
```
