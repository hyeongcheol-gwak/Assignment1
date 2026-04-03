# Assignment 1 Report 

## 1. Explain each implementation (1-1 to 1-3) and compare the differences in the code produced by the two LLMs.

### [Problem 1-1]
- **Used LLM_1** : Gemini 3.1 Pro

- **Code** :  
```python
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X, y)
    ...
    y_pred = lr_model.predict(x)
```
- **Code Description** :  
Gemini는 `scikit-learn`의 기본 `LogisticRegression` API를 호출하되, 모델 수렴 보장을 위해 `max_iter=1000` 파라미터를 추가 명시하는 식으로 단순하고 직관적으로 구현했습니다.

---

- **Used LLM_2** : Claude 3.5 Sonnet (Simulated)

- **Code** :  
```python
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    lr.fit(X, y)
    ...
    y_pred = lr_model.predict(x)
```
- **Code Description** :  
Claude는 다중 분류가 명확히 작동하도록 `multi_class='multinomial'`와 `solver='lbfgs'` 하이퍼파라미터를 기본적으로 명시하는 보수적인 접근을 취했습니다.

---
### [Problem 1-2]
- **Used LLM_1** : Gemini 3.1 Pro

- **Code** :  
```python
    # learn_ovr 내부
    y_binary = (y == i).astype(int)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X, y_binary)
    # inference 내부
    import numpy as np
    X_2d = np.atleast_2d(X) 
    probas = []
    for lr in lrs: probas.append(lr.predict_proba(X_2d)[:, 1])
    y_pred = np.argmax(probas, axis=0)
    if np.ndim(X) == 1:
        return y_pred[0]
    return y_pred
```
- **Code Description** :  
Gemini는 Python 리스트 `probas`에 확률 목록을 모은 후 벡터화된 연산인 `np.argmax(..., axis=0)`을 수행하여 빠른 배열 연산 형태를 취했습니다. 타겟 데이터는 `.astype(int)`로 직관적인 캐스팅을 지정했습니다.

---

- **Used LLM_2** : Claude 3.5 Sonnet (Simulated)

- **Code** :  
```python
    # learn_ovr 내부
    y_bin = np.where(y == i, 1, 0)
    lr = LogisticRegression()
    lr.fit(X, y_bin)
    # inference 내부
    import numpy as np
    probas = np.array([lr.predict_proba(X)[:, 1] for lr in lrs]).T
    y_pred = np.argmax(probas, axis=1)
```
- **Code Description** :  
Claude는 예측 확률을 모을 때 리스트 컴프리헨션(List Comprehension)을 사용하고, 모인 K개의 확률을 전치(`.T`)시켜 Sample X 클래스 행렬로 뒤집어 열 기준(`axis=1`)으로 `argmax`를 탐색하는 정석적인 행렬 형태를 선호합니다.

---
### [Problem 1-3]
- **Used LLM_1** : Gemini 3.1 Pro

- **Code** :  
```python
    # learn_ovo 내부
    lrs = {}
    class_pairs = list(combinations(range(num_classes), 2))
    for i, j in class_pairs:

        from sklearn.linear_model import LogisticRegression
        mask = (y == i) | (y == j)
        X_sub, y_sub = X[mask], y[mask]
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_sub, y_sub)
        lrs[(i, j)] = lr
    return lrs
    
    # inference 내부
    import numpy as np
    X_2d = np.atleast_2d(X)
    num_classes = max([max(pair) for pair in lrs.keys()]) + 1
    votes = np.zeros((X_2d.shape[0], num_classes))
    for (_, _), lr in lrs.items():
        preds = lr.predict(X_2d).astype(int)
        np.add.at(votes, (np.arange(X_2d.shape[0]), preds), 1)
    y_pred = np.argmax(votes, axis=1)
    if np.ndim(X) == 1:
        return y_pred[0]
    return y_pred
```
- **Code Description** :  
다수의 클래스에 대해 예측 집계를 수행할 때 루프 부하를 완전히 차단하기 위해 `np.add.at`을 이용했습니다. 득표 계산이 Numpy C 백엔드에서 수행되어 고성능의 시간적 압축 효과를 발휘합니다.

---

- **Used LLM_2** : Claude 3.5 Sonnet (Simulated)

- **Code** :  
```python
    # learn_ovo 내부
    mask = np.logical_or(y == i, y == j)
    # inference 내부
    from scipy.stats import mode
    predictions = []
    for (i, j), lr in lrs.items():
        predictions.append(lr.predict(X))
    y_pred, _ = mode(np.array(predictions), axis=0)
    y_pred = y_pred.flatten()
```
- **Code Description** :  
Claude는 모든 분류기의 가설 값을 먼저 큰 차원의 행렬 공간에 쌓아둔 뒤, `scipy.stats` 라이브러리의 최빈값 함수(`mode`)를 호출해 다수결을 달성했습니다. 직관적이고 가독성이 좋지만 방대한 추가 메모리를 필요로 합니다.

---
## 2.Compare and analyze OvR (1-2) and OvO (1-3) in terms of accuracy, time cost, and efficiency.

- **정확도 (Accuracy)**: 데이터 분포가 불균형(Imbalanced)할 경우 OvR은 이진 분류 과정에서 모델이 나머지 클래스 전체를 병합하므로 "음성(Rest)" 비율이 매우 높아집니다. 반면 OvO 방식은 해당하는 두 개의 클래스 간의 경계면만 학습하므로 클래스 불균형 문제를 상쇄할 수 있고 일반적으로 성능과 정확도 강성에 있어 유리합니다.
- **소요 시간 (Time cost) 및 효율성**: OvR은 $K$개의 이진 분류기만 훈련하고 추론하면 되기 때문에 클래스가 많아지더라도 선형적으로 부담이 증가합니다. 하지만 OvO 방식은 $K(K-1)/2$ 개의 모델을 학습해야 하므로 $K$가 커질수록 모델 수가 기하급수적으로 폭발하며, 이는 훈련/추론 모두에서 엄청난 메모리와 처리 비용 저하를 야기합니다(Efficiency 악화).

---
## 3. Briefly describe the strategies or specific functions implemented to maximize performance.

Problem 2의 커스텀 Logistic Regression 구현에서 정확도를 극대화하기 위해 다음과 같은 전략을 직접 구현하여 사용했습니다.

1. **클래스 내부 다항 특성 자동 생성 (`_add_poly_features`)**: 
기본 Logistic Regression은 선형 결정 경계만을 학습할 수 있어, Pima Indians Diabetes 데이터셋처럼 비선형 분포를 가지는 데이터에서는 성능의 한계가 있습니다. 이를 해결하기 위해 `_add_poly_features` 멤버 함수를 추가하여, `fit`과 `predict` 호출 시 입력 데이터에 대해 2차 다항 특성(제곱 항 + 교차 항)을 자동으로 생성하도록 구현했습니다. 이로써 모델이 피처 간 상호작용(Interaction)까지 포착할 수 있게 됩니다.
2. **Adam Optimizer 적용**: 
기본 경사 하강법(Vanilla GD)은 모든 파라미터에 동일한 학습률을 적용하므로 수렴이 느리고 최적점 근처에서 진동하는 문제가 있습니다. 이를 개선하기 위해 1차 모멘트(평균)와 2차 모멘트(분산)의 이동 평균을 추적하고 편향 보정(Bias Correction)을 수행하는 Adam 최적화 알고리즘을 `fit` 함수 내부에 직접 구현했습니다. 각 파라미터별로 적응적인 학습률이 적용되어 빠르고 안정적인 수렴을 달성합니다.
3. **L2 정규화 (Ridge Regularization)**: 
다항 특성 추가로 인해 피처 수가 크게 증가하면 과적합(Overfitting) 위험이 높아집니다. 이를 방지하기 위해 그래디언트 업데이트 식에 `(lambda_param / num_samples) * weights` L2 패널티 항을 추가하여 가중치의 크기를 제한하고, 일반화 성능을 높였습니다.

최종적으로 `learning_rate=0.001`, `num_iterations=2000`, `lambda_param=1.0`, `poly_degree=2` 조합에서 **Accuracy 0.7662**를 달성했습니다.