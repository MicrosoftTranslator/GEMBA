from pathlib import Path
import os
import pandas as pd


class Scores:
    def __init__(self, name, testset, refname, output_path=None):
        self.name = name
        self.testset = testset
        self.refname = refname
        if output_path is None:
            output_path = testset.basepath

        self.output_path = output_path

        self.seg_scores = None
        self.metadata = None
        self.prefix = None
        self.load()

    def load(self):
        output_folder = f"{self.testset.basepath}/{self.testset.dataset}/metric-scores/{self.testset.lp}"
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        if self.refname is not None:
            self.prefix = f"{output_folder}/{self.name}-{self.refname}"
        else:
            self.prefix = f"{output_folder}/{self.name}-src"

        seg_scores_file = self.get_seg_path()
        if os.path.isfile(f"{seg_scores_file}"):
            self.seg_scores = pd.read_csv(seg_scores_file, sep="\t", names=["system", "score"], index_col=False)
        else:
            self.seg_scores = pd.DataFrame(columns=["system", "score"])

        if os.path.isfile(f"{self.get_meta_path()}"):
            self.metadata = pd.read_csv(self.get_meta_path(), sep="\t", names=["system", "temperature"], index_col=False)
        else:
            self.metadata = pd.DataFrame(columns=["system", "temperature"])

        # generate placeholders for scores
        segment_count = len(self.testset.sources)
        for system in self.testset.systems.keys():
            if system not in self.seg_scores["system"].values:
                # generate placeholders for scores
                placeholder = pd.DataFrame([{"system": system, 'score': 'None'}] * segment_count)
                self.seg_scores = pd.concat([self.seg_scores, placeholder], ignore_index=True)

            if system not in self.metadata["system"].values:
                placeholder = pd.DataFrame([{"system": system, 'temperature': 'None'}] * segment_count)
                self.metadata = pd.concat([self.metadata, placeholder], ignore_index=True)

            # check that all systems are present and have correct number of scores
            assert len(self.seg_scores[self.seg_scores.system == system]) == segment_count
            assert len(self.metadata[self.metadata.system == system]) == segment_count

    def get_seg_path(self):
        return f"{self.prefix}.seg.score"

    def get_sys_path(self):
        return f"{self.prefix}.sys.score"

    def get_domain_path(self):
        return f"{self.prefix}.domain.score"

    def get_meta_path(self):
        return f"{self.prefix}.seg.meta"

    def _remap_index(self, system, hypothesis_index):
        # the order of systems may be different
        # get id of the first hypothesis of the system
        index = (self.seg_scores["system"] == system).argmax()
        h = hypothesis_index % len(self.testset.sources)
        return index + h

    def get_score(self, system, hypothesis_index):
        index = self._remap_index(system, hypothesis_index)
        return self.seg_scores.iloc[index]['score']

    def assign_score(self, system, hypothesis_index, answer, temperature=None):
        index = self._remap_index(system, hypothesis_index)
        self.seg_scores.iloc[index]['score'] = answer
        self.metadata.iloc[index]['temperature'] = temperature

    def save(self):
        # segment level scores
        self.seg_scores.to_csv(self.get_seg_path(), sep="\t", index=False, header=False, na_rep="None")

        # system scores
        self.seg_scores.score = self.seg_scores.score.replace("None", None).astype(float)
        sys_scores_df = self.seg_scores.groupby(['system'], as_index=False, dropna=True).mean()
        sys_scores_df.to_csv(self.get_sys_path(), sep="\t", index=False, header=False, na_rep="None")

        # domain scores
        df = self.seg_scores.copy()
        documents = self.testset.documents
        total_systems = len(self.testset.systems)
        df["domains"] = pd.DataFrame([x.split("\t")[0] for x in documents] * total_systems)[0]
        df = df.groupby(["domains", 'system'], as_index=False, dropna=True).mean()
        df.to_csv(self.get_domain_path(), sep="\t", index=False, header=False, na_rep="None")

        # metadata
        self.metadata.to_csv(self.get_meta_path(), sep="\t", index=False, header=False, na_rep="None")
