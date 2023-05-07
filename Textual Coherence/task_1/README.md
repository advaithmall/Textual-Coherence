

### Directory Structure 

```
.
├── 24-InterimSubmission.pdf
├── dataset.py
├── eval.py
├── eval.txt
├── main.py
├── model.pt
├── model.py
├── performance.txt
├── __pycache__
│   ├── dataset.cpython-310.pyc
│   ├── model.cpython-310.pyc
│   └── tools.cpython-310.pyc
├── README.md
├── test_set.pt
├── tools.py
└── train_set.pt

```
<br><br>

###  Note: GloVe embeddings need to be downloaded before running my code, in terminal RUN:

```
python3
import torch
from torchtext.vocab import GloVe
global_vectors = GloVe(name='840B', dim=300)
```
<br><br>
### As model.pt, train_set.pt, test_set.pt are too big in size, links have been included, download in the parent directory
<br>

### model.pt : https://iiitaphyd-my.sharepoint.com/:u:/g/personal/advaith_malladi_research_iiit_ac_in/EZ7wqI2LFlRFhLGSVM_8ZekBXnp5lCQyXBxFm8AhCuatUg?e=iDidNG

### train_set.pt : https://iiitaphyd-my.sharepoint.com/:u:/g/personal/advaith_malladi_research_iiit_ac_in/EaFVDo9syiVDmUkWF161NXEBZu_IAuSsjLVlnY0PmscGAw?e=6o5iez

### test_set.pt : https://iiitaphyd-my.sharepoint.com/:u:/g/personal/advaith_malladi_research_iiit_ac_in/ERkiAzBd0llLsx4_5NUb7k8BwiwmWSzCmJ4zFk0U_l0uzQ?e=RJHGNT

<br>

### to look at report, open:

```
24-InterimSubmission.pdf
```

<br>

### to check the performance, open:

```
performance.txt
```

<br>

### to check batch wise performance on test set, check:

```
eval.txt
```

### to train the model and to save models, RUN:

```
python3 -W ignore main.py
```

<br>

### to check performance on test set, RUN:

```
python3 -W ignore eval.py
```

