# MZI-VMM noise simulation

## 環境設定

Python version = 3.6.0

ライブラリーの一覧：```requirements.txt```を参照してください．

## 実行

```bash
python matrix2opti.py
```

### 入力行列

```opti_inputs.csv```にて入力ベクトル/行列を指定できます．

### ノイズ

```matrix2opti.py```に中の下記の変数を変更することで，ノイズの分散を調節できます．
ノイズはガウス分布を想定しています．

```python
In_noise_variance = 10 ** -6
W_noise_variance = 10 ** -6
```
