
from typing import Union

import faiss
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI


app = FastAPI()
dim = 72
faiss_index = None
model = None
base_index ={}
base = None


def parse_string(vec: str) -> list[float] | None:
    l = vec.split(",")
    if len(l) != dim:
        return None
    l2 = [float(el) for el in l]
    return l2

@app.on_event("startup")
def start():
    global faiss_index
    global base_index
    global base
    n_cells = 5

    base = pd.read_csv('base.csv', index_col=0)

    quantizer = faiss.IndexFlatL2(dim)
    faiss_index = faiss.IndexIVFFlat(quantizer, dim, n_cells)

    faiss_index.train(np.ascontiguousarray(base.values[:50000, :]).astype('float32'))
    faiss_index.add(np.ascontiguousarray(base.values).astype('float32'))

    base_index = {k: v for k, v in enumerate(base.index.to_list())}

@app.get('/')
def main() -> dict:
    return {'status': 'OK', 'message': 'Hello, world!'}


@app.get('/knn')
def match(item: Union[str, None] = None) -> dict:
    global faiss_index
    global base_index


    if item is None:
        return {'status': 'Fail', 'message': 'No input data'}

    vec = parse_string(item)
    if item is None:
        return {'status': 'Fail', 'message': 'No input data'}

    vec = np.ascontiguousarray(vec, dtype='float')[np.newaxis, :]

    knn, idx = faiss_index.search(vec, k=5)
    data = [base_index[el] for el in idx.flatten()]


    return {'status': 'Ok', 'data': data}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8025)
