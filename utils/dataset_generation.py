from torch.utils.data import Dataset


class ProcessingDataset(Dataset):
    def __init__(self, ent_list, input_tokens, label_inputs, segment_pos_dic, input_mask):
        self.examples = self._yield_example(ent_list, input_tokens, label_inputs, segment_pos_dic, input_mask)

    def __getitem__(self, i):
        return self.examples[i]

    def __len__(self):
        try:
            return len(self.examples)
        except TypeError:
            return -1

    def __iter__(self):
        for x in self.examples:
            yield x

    def _yield_example(self, ent_list, input_tokens, label_inputs, segment_pos_dic, input_mask):
        tokens = []
        labels = []
        segment_pos = []
        mask = []
        examples = []
        for e in input_tokens.keys():
            if e in ent_list:
                tokens.extend(input_tokens[e])
                labels.extend(label_inputs[e])
                segment_pos.extend(segment_pos_dic[e])
                mask.extend(input_mask[e])
                example = {
                    "entity": e,
                    "input_tokens": tokens,
                    "segment_ids": segment_pos,
                    "mask": mask,
                    "labels": labels
                }
                examples.append(example)
        return examples


class ComparisonDataset(Dataset):
    def __init__(self, ent_list):
        self.ent_list = ent_list

    def __getitem__(self, i):
        return self.ent_list[i]

    def __len__(self):
        try:
            return len(self.ent_list)
        except TypeError:
            return -1

    def __iter__(self):
        for x in self.ent_list:
            yield x


def collate_fn(batch):
    return batch