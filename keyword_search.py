"""BM25 keyword search, implemented directly so there's nothing extra to install.

Dense (embedding) search matches meaning; BM25 matches words. Legal questions often hinge on a
specific word ("fine", "deployer") or reference ("Article 5"), which dense search can miss when
the answer sits inside a long chunk that is mostly about something else.
"""
import math
import re
from collections import Counter

STOPWORDS = set("""
a an the of to in on for and or is are was were be been by with as at from that this these those
which what who whom whose how when where why it its their there than then so if into about under
any all such not no nor do does did can could should would shall may must will also other
""".split())


def stem(w: str) -> str:
    """A deliberately light stemmer: fines -> fine, prohibited/prohibition -> prohibit."""
    if len(w) <= 3 or w.isdigit():
        return w
    if w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith(("sses", "xes", "zes", "ches", "shes")):
        w = w[:-2]
    elif w.endswith("s") and not w.endswith(("ss", "us", "is")):
        w = w[:-1]
    for suffix in ("ion", "ing", "ed"):
        if w.endswith(suffix) and len(w) - len(suffix) >= 4:
            w = w[: -len(suffix)]
            break
    if w.endswith("e") and len(w) > 4:
        w = w[:-1]
    return w


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(d)) for d in docs]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avg_len = sum(self.lens) / max(len(self.lens), 1)
        n = len(docs)
        df = Counter(w for tf in self.tfs for w in tf)
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}

    def scores(self, query: str) -> list[float]:
        terms = set(tokenize(query))
        out = []
        for tf, length in zip(self.tfs, self.lens):
            s = 0.0
            for w in terms:
                f = tf.get(w)
                if f:
                    norm = f + self.k1 * (1 - self.b + self.b * length / self.avg_len)
                    s += self.idf[w] * f * (self.k1 + 1) / norm
            out.append(s)
        return out
