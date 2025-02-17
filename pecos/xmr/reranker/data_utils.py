import os
import random
from collections import OrderedDict
from typing import List, Tuple, Callable

import numpy as np
import pyarrow.parquet as pq
from datasets import load_dataset

import pecos


class RankingDataUtils(pecos.BaseClass):
    """
    Utility class for handling data related tasks
    """

    @classmethod
    def remap_ordereddict(cls, od: OrderedDict, keymap_fn: Callable):
        """
        Function to remap the keys of an ordered Dictionary
        Args:
            od: The ordered dictionary to remap
            keymap_fn: The function to map the keys
        """
        new_od = OrderedDict()
        for k, v in od.items():
            new_od[keymap_fn(k)] = v
        return new_od

    @classmethod
    def _format_sample(
        cls,
        inp_text: str,
        lbl_contents: List[str],
        inp_prefix: str = "...",
        passage_prefix: str = "...",
        content_sep=" ",
    ) -> str:
        """
        Function to convert the text fields into a formatted string
        that the model understands.
        Args:
            inp_text: The input text
            lbl_contents: The list of content fields
            inp_prefix: The input prefix
            passage_prefix: The passage prefix
            content_sep: The separator between the content fields
        Returns: The formatted string
        """
        # Convention from rankllama is to replace hyphens in the title
        lbl_contents[0] = lbl_contents[0].replace("-", " ").strip()
        return f"{inp_prefix} {inp_text} {passage_prefix} {content_sep.join(lbl_contents)}".strip()

    @classmethod
    def _create_sample(
        cls,
        inp_id: int,
        ret_idxs: List[int],
        scores: List[float],
        table_stores,
        train_group_size: int,
        inp_prefix: str,
        passage_prefix: str,
        keyword_col_name: str,
        content_col_names: List[str],
        content_sep,
    ) -> Tuple[List[str], List[float]]:
        """
        Function to create a sample for training.
        Args:
            inp_id: The input id
            ret_idxs: The retrieved indices
            scores: Scores for the retrieved indices
            table_stores: Dictionary of table stores for input and label data
            train_group_size: The number of passages used to train for each query
            inp_prefix: The input prefix
            passage_prefix: The passage prefix
            keyword_col_name: The column name for the query text
            content_col_names: The column names for the content fields
            content_sep: The separator between the content fields
        Returns: A tuple of formatted samples and scores
        """
        qid = inp_id
        pidxs = ret_idxs

        input_store = table_stores["input"]
        label_store = table_stores["label"]

        # get the values of the query
        query = input_store[qid][keyword_col_name]
        mean_score = np.mean(scores)

        # get idxs for positive items
        pos_idxs = [(x, pid) for x, pid in zip(scores, pidxs) if x > mean_score]
        neg_idxs = [(x, pid) for x, pid in zip(scores, pidxs) if x <= mean_score]
        random.shuffle(pos_idxs)
        random.shuffle(neg_idxs)

        num_positives = train_group_size // 2

        all_selections = pos_idxs[:num_positives]
        num_positives = len(all_selections)
        num_negatives = train_group_size - num_positives
        all_selections.extend(neg_idxs[:num_negatives])

        if len(all_selections) < train_group_size:
            all_selections.extend(
                random.choices(neg_idxs, k=train_group_size - len(all_selections))
            )

        all_scores = [s for s, _ in all_selections]
        all_pids = [pid for _, pid in all_selections]

        # get the values for the retrieved items
        ret_info = [label_store[i] for i in all_pids]

        formated_pair = []
        for info in ret_info:
            formated_pair.append(
                cls._format_sample(
                    query,
                    [info[c] for c in content_col_names],
                    inp_prefix,
                    passage_prefix,
                    content_sep,
                )
            )
        return formated_pair, all_scores

    @classmethod
    def get_parquet_rows(cls, folder_path: str) -> int:
        """
        Returns the count of rows in parquet files by reading the
        metadata
        Args:
            folder_path: The folder containing the parquet files
        Returns: The count of rows in the parquet files
        """
        file_list = os.listdir(folder_path)
        file_list = [os.path.join(folder_path, x) for x in file_list]
        cumulative_rowcount = sum([pq.read_metadata(fp).num_rows for fp in file_list])

        return cumulative_rowcount

    @classmethod
    def get_sorted_data_files(cls, filenames: List[str], idx_colname) -> List[str]:
        """
        Returns the list of files sorted by the id in the first row of each file
        Args:
            filenames: The list of filenames
            idx_colname: The column name of the id
        Returns: The sorted list of filenames
        """
        # Load the datasets in streaming format and read the first id
        fn_ordered = []  # this containes tuples with (idx, filename)
        for fn in filenames:
            tmp_ds = load_dataset("parquet", data_files=fn, streaming=True, split="train")
            row = next(iter(tmp_ds.take(1)))
            fn_ordered.append((row[idx_colname], fn))
            del tmp_ds
        fn_ordered = sorted(fn_ordered, key=lambda x: x[0])

        return [x[1] for x in fn_ordered]
