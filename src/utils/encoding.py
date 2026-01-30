import pandas as pd

def label_encode_union(train: pd.Series, test: pd.Series=None, fill: str="missing") -> tuple[pd.Series, pd.Series]:
    tr = train.astype("object").fillna(fill)
    if test is not None:
        te = test.astype("object").fillna(fill)
        cats = pd.Index(tr.unique()).union(pd.Index(te.unique()))
    else:
        te = None
        cats = pd.Index(tr.unique())

    mapping = {k: i for i, k in enumerate(cats.tolist())}
    tr_enc = tr.map(mapping).astype("int32")

    te_enc = te.map(mapping).astype("int32") if te is not None else None
    return tr_enc, te_enc, mapping

