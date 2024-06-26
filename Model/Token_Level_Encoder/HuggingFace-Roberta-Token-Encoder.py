import sys

# Add the ptdraft folder path to the sys.path list
# sys.path.append(package-path)
# import transformer
from transformers import AutoTokenizer, AutoModel
import torch

model = AutoModel.from_pretrained("roberta-large")
tokenizer = AutoTokenizer.from_pretrained("roberta-large")


def emb_str():
    input_ids = torch.tensor(
        [tokenizer.encode(str("inputs here"), add_special_tokens=False)]
    )
    with torch.no_grad():
        last_hidden_states = model(input_ids)[0]  # Models outputs are now tuples
        emb = last_hidden_states.cpu().numpy()
    return emb
