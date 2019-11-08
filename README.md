# AirDialogue
AirDialogue is a benchmark dataset for goal-oriented dialogue generation
research. This python library contains a collection of tookits that come with the dataset.
- [AirDialogue paper][paper]
- AirDialogue dataset
- Reference implementation: [AirDialogue Model][airdialogue_model]

## Prerequisites
#### General
- python (verified on 2.7.16)
- wget

#### Python Packages
- tqdm
- nltk
- flask (for visualization)

## Install
To install the pre-build version from pip, use
```
pip install airdialogue
```

To install the bleeding edge from github, use
```
python setup.py install
```

## Quick Start
#### Scoring
The official scoring function evaluates the predictive results for a trained model and compare it to the AirDialogue dataset.

```
airdialogue score --data PATH_TO_DATA_FILE --kb PATH_TO_KB_FILE
```

#### Context Generation
Context generator generates a valid context-action pair without conversatoin history.
```
airdialogue contextgen \
    --output_data PATH_TO_OUTPUT_DATA_FILE \
    --output_kb PATH_TO_OUTPUT_KB_FILE \
    --num_samples 100
```

#### Preprocessing
AirDialogue proprocess tookie tokenlizes dialogue. Preprocess on AirDialogue data requires 50GB of ram to work.
Parameter job_type is a set of 5 bits separted by `|`, which reqpresents `train|eval|infer|sp-train|sp-eval`.
Parameter input_type can be either `context` for context only data or `dialogue` for dialogue data with full history.
```
airdialogue prepro \
  --data_file PATH_TO_DATA_FILE \
  --kb_file PATH_TO_KB_FILE \
  --output_dir "./data/airdialogue/" \
  --output_prefix 'train' --job_type '0|0|0|1|0' --input_type context
```

#### Simulator
Simulator is built on top of context generator that provides not only a context-action pair but also a full conversation history generated by two templated chatbot agents.
```
airdialogue sim \
    --output_data PATH_TO_OUTPUT_DATA_FILE \
    --output_kb PATH_TO_OUTPUT_KB_FILE \
    --num_samples 100
```

#### Visualization
Visualization tool displays the content of the raw json file.
```
airdialogue vis --data_path ./data/airdialogue/json/
```


[paper]: https://www.aclweb.org/anthology/D18-1419/
[airdialogue_model]: https://github.com/google/airdialogue_model
