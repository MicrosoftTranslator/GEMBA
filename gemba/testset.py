import glob
import os


class Testset:
    def __init__(self, basepath, dataset, lp):
        self.basepath = basepath
        self.dataset = dataset
        self.lp = lp

        self.sources = []
        self.references = {}
        self.systems = {}
        self.documents = []
        self.main_ref = None

        self.load()

    def load(self):
        dataset = f"{self.basepath}/{self.dataset}"

        self.sources = self.load_segment_files(f"{dataset}/sources/{self.lp}.txt")

        # list all files in references folder
        refs = glob.glob(f"{dataset}/references/{self.lp}.*.txt")
        for reffile in refs:
            refname = reffile.split('.')[-2]
            if self.main_ref is None:
                self.main_ref = refname
            self.references[refname] = self.load_segment_files(reffile)

        systems = f"{dataset}/system-outputs/{self.lp}"
        # keep systems in order
        all_systems = sorted(os.listdir(systems))
        for system in all_systems:
            systemname = system.replace(".txt", "")
            self.systems[systemname] = self.load_segment_files(f"{systems}/{system}")

        self.documents = self.load_segment_files(f"{dataset}/documents/{self.lp}.docs")

    def iterate_over_all(self, reference=None):
        for system in self.systems.keys():
            if reference is None:
                for src, hyp in zip(self.sources, self.systems[system]):
                    yield src, hyp, None, system
            else:
                for src, hyp, ref in zip(self.sources, self.systems[system], self.references[reference]):
                    yield src, hyp, ref, system

    def load_segment_files(self, path):
        segments = []
        with open(path, "r") as fh:
            for line in fh:
                segments.append(line.rstrip())
        return segments

    def segments_count(self):
        return len(self.sources)*len(self.systems)
