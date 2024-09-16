from datasets import NLVR2Dataset
from transformers import XLMRobertaTokenizer

tokenizer = XLMRobertaTokenizer("/home/naver/Individual/VisualReasoning/unilm/beit3/beit3.spm")

NLVR2Dataset.make_dataset_index(
    data_path="/home/naver/Individual/VisualReasoning/unilm/beit3/NLVR2", 
    tokenizer=tokenizer, 
    nlvr_repo_path="/home/naver/Individual/VisualReasoning/unilm/beit3/NLVR2/nlvr"
)