# B3 跑法總表（Colab T4；多 seed＋第二語料＋第二端點）

## A. Colab 準備（每個 session 一次）
```python
!pip install -q vec2text jiwer
!pip install -q transformers==4.46.3
from google.colab import drive; drive.mount('/content/drive')
# 上傳 colab_embed_e1.py + colab_embed_b3.py 到 /content/
```

## B. 多 seed（6 runs，每個 5-25 分，共約 1.5-2 小時）
```python
%run colab_embed_b3.py --corpus agnews --seed 2006 --num-steps 2 --beam 1
%run colab_embed_b3.py --corpus agnews --seed 2006 --num-steps 4 --beam 2
%run colab_embed_b3.py --corpus agnews --seed 2006 --num-steps 4 --beam 4
%run colab_embed_b3.py --corpus agnews --seed 3006 --num-steps 2 --beam 1
%run colab_embed_b3.py --corpus agnews --seed 3006 --num-steps 4 --beam 2
%run colab_embed_b3.py --corpus agnews --seed 3006 --num-steps 4 --beam 4
```
（seed 1006 三檔＝論文原始 run，不重跑；輸出 JSON 自帶 per_sample_f1＋token 長度）

## C. 第二語料 Enron（3 runs，約 45-75 分）
```python
%run colab_embed_b3.py --corpus enron --seed 1006 --num-steps 2 --beam 1
%run colab_embed_b3.py --corpus enron --seed 1006 --num-steps 4 --beam 2
%run colab_embed_b3.py --corpus enron --seed 1006 --num-steps 4 --beam 4
```
（SetFit/enron_spam；同一 160 字符截斷＝隔離域效應；<40 字符空殼自動剔除）

## D. 第二端點 OpenAI（text-embedding-3-large，3072-d；總費用 <$1）
```python
import os
from getpass import getpass
os.environ["OPENAI_API_KEY"] = getpass("OpenAI key: ")   # key 走 Colab secrets/getpass，不進 notebook 明文

!python colab_embed_e3a_fetch.py --endpoint https://api.openai.com/v1/embeddings --model text-embedding-3-large --n 500 --pool 10000 --out /content/e3a_openai
!python colab_embed_e3b_retrieval.py --emb /content/e3a_openai.npz --out /content/e3b_openai_retrieval.json
```
（e3a/e3b 參數名以腳本 --help 為準；key 經 os.environ，fetch 腳本讀 OPENAI_API_KEY）

## E. 跑完回傳
`/content/` 下全部 `e1_*_s2006/b3006*.json`＋`e1_enron_*.json`＋`e3a_openai*`＋`e3b_openai*.json` 打包發回。
```
