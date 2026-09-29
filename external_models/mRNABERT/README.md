---
library_name: transformers
license: apache-2.0
tags:
- biology
- medical
- mRNA
- rna
- mrna
---

# mRNABERT

 A robust language model pre-trained on over 18 million high-quality mRNA sequences, incorporating contrastive learning to integrate the semantic features of amino acids.

This is the official pre-trained model introduced in  [mRNABERT: advancing mRNA sequence design with a universal language model and comprehensive dataset](https://www.nature.com/articles/s41467-025-65340-8#citeas).

The repository of mRNABERT is at [yyly6/mRNABERT](https://github.com/yyly6/mRNABERT).


## Intended uses & limitations
The model could be used for mRNA sequences feature extraction or to be fine-tuned on downstream tasks. **Before inputting the model, you need to preprocess the data: use single-letter separation for the UTR regions and three-character separation for the CDS regions.**For full examples, please see [our code on data processing](https://github.com/yyly6/mRNABERT).

## Training data

The mRNABERT model was pretrained on [a comprehensive mRNA dataset](https://zenodo.org/records/12516160), which originally consisted of approximately 36 million complete CDS or mRNA sequences. After cleaning, this number was reduced to 18 million.

## Usage
To load the model from huggingface:
```python
import torch
from transformers import AutoTokenizer, AutoModel
from transformers.models.bert.configuration_bert import BertConfig

config = BertConfig.from_pretrained("YYLY66/mRNABERT")
tokenizer = AutoTokenizer.from_pretrained("YYLY66/mRNABERT")
model = AutoModel.from_pretrained("YYLY66/mRNABERT", trust_remote_code=True, config=config)
```

To extract the embeddings of mRNA sequences:

```python
seq = ["A T C G G A GGG CCC TTT", 
       "A T C G", 
       "TTT CCC GAC ATG"]  #Separate the sequences with spaces.

encoding = tokenizer.batch_encode_plus(seq, add_special_tokens=True, padding='longest', return_tensors="pt")

input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask'] 

output = model(input_ids=input_ids, attention_mask=attention_mask)
last_hidden_state = output[0]

attention_mask = attention_mask.unsqueeze(-1).expand_as(last_hidden_state)  # Shape : [batch_size, seq_length, hidden_size]

# Sum embeddings along the batch dimension
sum_embeddings = torch.sum(last_hidden_state * attention_mask, dim=1)  

# Also sum the masks along the batch dimension
sum_masks = attention_mask.sum(1)  

# Compute mean embedding.
mean_embedding = sum_embeddings / sum_masks  #Shape:[batch_size, hidden_size]  

```

The extracted embeddings can be used for contrastive learning pretraining or as a feature extractor for protein-related downstream tasks.



## Citation

**BibTeX**:

```
@article{xiong2025mrnabert,
  title={mRNABERT: advancing mRNA sequence design with a universal language model and comprehensive dataset},
  author={Xiong, Ying and Wang, Aowen and Kang, Yu and Shen, Chao and Hsieh, Chang-Yu and Hou, Tingjun},
  journal={Nature Communications},
  volume={16},
  number={1},
  pages={10371},
  year={2025},
  publisher={Nature Publishing Group UK London},
}
```

## Contact
If you have any question, please feel free to email us (xiongying@zju.edu.cn).