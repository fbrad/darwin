from allennlp.data.dataset_readers.dataset_reader import DatasetReader, AllennlpDataset, AllennlpLazyDataset
from typing import Dict, List, Union, Iterable
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer
from allennlp.data.instance import Instance

PAN2020_PATH = "/pan2020/pan20-authorship-verification-training-small/pan20-authorship-verification-training-small.jsonl"
PAN2020_GT_PATH = "/pan2020/pan20-authorship-verification-training-small/pan20-authorship-verification-training-small-truth.jsonl"

@DatasetReader.register('pan2020_dataset_reader')
class Pan2020DatasetReader(DatasetReader):
    def __init__(self, 
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self._tokenizer = tokenizer or SpacyTokenizer()
        self._token_indexers = token_indexers or SingleIdTokenIndexer()

        def _read(self, file_path: str) -> Iterable[Instance]:
            with open()
            
            return None

        def text_to_instance(self, text1, text2) -> Instance:
            return None
