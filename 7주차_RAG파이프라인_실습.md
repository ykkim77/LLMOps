# 7주차 실습: Kubeflow Pipelines로 RAG 검색 파이프라인 구축·비교

- **과목**: AI플랫폼
- **주차**: 07. LLMOps 및 생성형AI 응용
- **차시**: 3차시 (실습)

## 1. 실습 목표

- RAG의 인덱싱·검색 평가 단계를 Kubeflow Pipelines 컴포넌트 4개로 구현
- `chunk_size`, `top_k`를 바꿔 여러 번 실행(Run)하고, `hit_rate`와 `MRR` 비교
- 설정 변경이 검색 품질에 미치는 영향 해석
- **주의**: 검색(retrieval) 품질만 평가 (답변 생성과 충실성 평가는 범위 제외)

## 2. 실습 환경

- **클러스터**: Kubeflow Pipelines (6주차와 동일 환경)
- **연산**: CPU만 사용 (GPU 불필요)
- **로컬 Python**: 3.9 이상, `kfp` 2.x
- **컴포넌트 패키지**: `sentence-transformers`, `faiss-cpu`, `numpy` (실행 시 자동 설치)
- **임베딩 모델**: `all-MiniLM-L6-v2` (약 90MB, 영어 모델)
- **네트워크**: 인터넷 필요 (pip 패키지·HuggingFace 모델 다운로드)

**소요 시간 안내**
- 첫 Run: 5~15분 (컴포넌트마다 패키지 신규 설치)
- 이후 Run: 수 분
- Run 시작 후 기다리는 동안 다음 단계 미리 읽기

**데이터 안내**
- 실습용 문서: 쿠버네티스·Kubeflow 설명 12개
- 평가용 질문: 15개
- 구성: 코드 안에 직접 포함
- 언어: 영어 (임베딩 모델이 영어 중심)

## 3. 실습 개요

**파이프라인 흐름**
```
문서 → [load_and_chunk] → [embed] → [build_index] → [evaluate_retrieval] → hit_rate, MRR
        chunk_size        모델명                       top_k, 질문 15개
```

**컴포넌트별 입출력**
- **load_and_chunk**
  - 입력: 문서(JSON), `chunk_size`
  - 출력: Dataset(청크 목록)
- **embed**
  - 입력: 청크, 모델명
  - 출력: Dataset(벡터)
- **build_index**
  - 입력: 벡터
  - 출력: Model(FAISS 인덱스)
- **evaluate_retrieval**
  - 입력: 인덱스, 청크, 질문, `top_k`
  - 출력: Metrics(hit_rate, mrr)

## 4. 사전 점검

### (1) 클러스터와 Kubeflow Pipelines 확인

**명령어**
```bash
kubectl get nodes
kubectl get pods -n kubeflow
```

**확인 사항**
- 노드: `Ready` 상태
- 파드: 대부분 `Running` 상태
- 클러스터 부재 시: 5주차 실습 재진행

### (2) 대시보드 접속

**별도 터미널에서 실행**
```bash
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
```

**접속 확인**
- 브라우저: `http://localhost:8080` 접속 여부 확인

### (3) Python 및 SDK 설치

**설치 단계**
```bash
python --version
pip install "kfp>=2.5"
python -c "import kfp; print(kfp.__version__)"
```

## 5. 실습 단계

작업 폴더를 만들고 `rag_pipeline.py` 파일 **하나**에 아래 코드를 순서대로 이어 붙입니다.

```bash
mkdir week7-rag
cd week7-rag
```

### Step 1. 데이터와 공통 설정 작성

`rag_pipeline.py`의 맨 위에 작성합니다.

```python
import json

from kfp import compiler, dsl
from kfp.dsl import Dataset, Input, Metrics, Model, Output

# ---------------------------------------------------------------
# 1) 실습용 문서: 쿠버네티스·Kubeflow 설명 12개 (id: d0 ~ d11)
# ---------------------------------------------------------------
CORPUS = [
    {"id": "d0", "text": "A Pod is the smallest deployable unit in Kubernetes. It wraps one or more containers that share the same network namespace and storage volumes. Containers in the same Pod can talk to each other through localhost. Pods are ephemeral: when a Pod dies, it is not resurrected, and a controller must create a replacement."},
    {"id": "d1", "text": "A Deployment manages a set of identical Pods through a ReplicaSet. You declare the desired number of replicas and the container image, and the controller keeps the actual state equal to the desired state. When you change the image version, the Deployment performs a rolling update, replacing old Pods gradually so the application stays available. If the new version fails, you can roll back to a previous revision."},
    {"id": "d2", "text": "A Service gives a stable network address to a group of Pods selected by labels. Because Pod IP addresses change whenever Pods are recreated, clients connect to the Service instead. A ClusterIP Service is reachable only inside the cluster, while a NodePort Service opens a port on every node. The Service forwards traffic to healthy Pods and spreads requests among them."},
    {"id": "d3", "text": "ConfigMaps store non-sensitive configuration such as environment names or feature flags, separate from the container image. Secrets hold sensitive values like passwords or API tokens. Both can be injected into Pods as environment variables or mounted as files. Separating configuration from images lets the same image run in development and production with different settings."},
    {"id": "d4", "text": "Namespaces divide a single cluster into virtual sections, so that teams or projects can use the same resource names without conflict. Resource quotas can be attached to a namespace to limit CPU and memory usage. Kubeflow components are typically installed in their own namespace, which keeps them apart from application workloads."},
    {"id": "d5", "text": "Kubeflow Pipelines orchestrates multi-step machine learning workflows. Each step is a component that runs in its own container, and the steps form a directed acyclic graph. Components exchange data through parameters and artifacts such as datasets, models, and metrics. Every execution is recorded as a run, which can be grouped into an experiment and compared with other runs."},
    {"id": "d6", "text": "Katib automates hyperparameter tuning in Kubeflow. The user defines a search space, an objective metric such as accuracy, and a search algorithm like random search or Bayesian optimization. Katib launches many trials, each with different hyperparameter values, and records the objective metric of every trial so the best configuration can be selected."},
    {"id": "d7", "text": "KServe serves trained models on Kubernetes through a custom resource called InferenceService. It handles autoscaling, including scaling down to zero when there is no traffic, and supports canary rollouts of new model versions. KServe exposes a standard HTTP endpoint so that applications can request predictions without knowing how the model is implemented."},
    {"id": "d8", "text": "The Kubeflow Training Operator runs distributed training jobs on Kubernetes. It provides custom resources such as TFJob and PyTorchJob that describe the number of workers and their resources. The operator creates the required Pods, wires them together for communication, and restarts failed workers, which makes data-parallel training across several machines much easier to manage."},
    {"id": "d9", "text": "A CustomResourceDefinition lets users extend the Kubernetes API with new object types. After a CRD is registered, objects of the new kind can be created with kubectl like built-in resources. A custom controller then watches those objects and acts to reach the declared state. Kubeflow uses CRDs for notebooks, experiments, and inference services."},
    {"id": "d10", "text": "kind runs Kubernetes clusters inside Docker containers, where every node is a container. It is designed for local development and testing, so a full cluster can be created with one command and deleted just as quickly. Because the nodes share the resources of the host machine, kind is not suited for production workloads or large GPU training."},
    {"id": "d11", "text": "kubectl is the command line tool for talking to the Kubernetes API server. The get command lists resources, describe shows detailed state and recent events, and logs prints the output of a container. When a Pod stays in Pending status, running kubectl describe on it usually reveals the reason, such as insufficient memory or an unbound volume."},
]

# ---------------------------------------------------------------
# 2) 평가용 질문 15개와 정답 문서 id
# ---------------------------------------------------------------
QUESTIONS = [
    {"question": "What is the smallest deployable unit in Kubernetes?", "answer_id": "d0"},
    {"question": "How can I update an application version without downtime?", "answer_id": "d1"},
    {"question": "Why should clients connect to a Service rather than directly to Pod IPs?", "answer_id": "d2"},
    {"question": "Where should passwords and API tokens be stored for a Pod?", "answer_id": "d3"},
    {"question": "How can two teams use identical resource names in one cluster?", "answer_id": "d4"},
    {"question": "Which Kubeflow component records each execution so runs can be compared?", "answer_id": "d5"},
    {"question": "Which component searches for the best learning rate automatically?", "answer_id": "d6"},
    {"question": "How can a model scale down to zero when there is no traffic?", "answer_id": "d7"},
    {"question": "How do I run PyTorch training across several workers?", "answer_id": "d8"},
    {"question": "How can I add my own object type to the Kubernetes API?", "answer_id": "d9"},
    {"question": "Which tool creates a local cluster using Docker containers as nodes?", "answer_id": "d10"},
    {"question": "A Pod is stuck in Pending; how do I find out why?", "answer_id": "d11"},
    {"question": "Do containers in one Pod share the same network?", "answer_id": "d0"},
    {"question": "What kind of data do pipeline components exchange?", "answer_id": "d5"},
    {"question": "How are failed distributed training workers handled?", "answer_id": "d8"},
]

CORPUS_JSON = json.dumps(CORPUS)
QUESTIONS_JSON = json.dumps(QUESTIONS)

# torch는 기본 설치 시 용량이 매우 큰 GPU 버전이 내려받아지므로 CPU 전용 저장소를 먼저 지정
TORCH_CPU = ["https://download.pytorch.org/whl/cpu", "https://pypi.org/simple"]
```

### Step 2. `load_and_chunk` 컴포넌트

문서를 `chunk_size` 글자 단위로 자릅니다. 인접한 청크가 20%씩 겹치도록(overlap) 하여 문장이 경계에서 끊기는 손실을 줄입니다.

```python
@dsl.component(base_image="python:3.10")
def load_and_chunk(corpus_json: str, chunk_size: int, chunks: Output[Dataset]):
    import json

    docs = json.loads(corpus_json)
    overlap = chunk_size // 5
    step = chunk_size - overlap

    items = []
    for d in docs:
        text = d["text"]
        for start in range(0, len(text), step):
            piece = text[start:start + chunk_size].strip()
            if piece:
                items.append({"doc_id": d["id"], "text": piece})
            if start + chunk_size >= len(text):
                break

    with open(chunks.path, "w") as f:
        json.dump(items, f)
    chunks.metadata["num_chunks"] = len(items)
    print(f"chunk_size={chunk_size}, 청크 수={len(items)}")
```

### Step 3. `embed` 컴포넌트

각 청크를 벡터로 변환합니다. 벡터를 정규화(`normalize_embeddings=True`)하면 내적이 곧 코사인 유사도가 됩니다.

```python
@dsl.component(
    base_image="python:3.10",
    packages_to_install=["sentence-transformers", "numpy"],
    pip_index_urls=TORCH_CPU,
)
def embed(chunks: Input[Dataset], model_name: str, vectors: Output[Dataset]):
    import json

    import numpy as np
    from sentence_transformers import SentenceTransformer

    with open(chunks.path) as f:
        items = json.load(f)

    model = SentenceTransformer(model_name)
    emb = model.encode(
        [c["text"] for c in items],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    emb = np.asarray(emb, dtype="float32")

    with open(vectors.path, "wb") as f:
        np.save(f, emb)
    vectors.metadata["dim"] = int(emb.shape[1])
    print(f"벡터 shape={emb.shape}")
```

### Step 4. `build_index` 컴포넌트

벡터를 FAISS 인덱스에 저장합니다. `IndexFlatIP`는 모든 벡터와 내적을 계산하는 가장 단순한 방식이며, 이 규모에서는 충분합니다.

```python
@dsl.component(
    base_image="python:3.10",
    packages_to_install=["faiss-cpu", "numpy"],
)
def build_index(vectors: Input[Dataset], index: Output[Model]):
    import faiss
    import numpy as np

    with open(vectors.path, "rb") as f:
        arr = np.load(f)

    idx = faiss.IndexFlatIP(arr.shape[1])
    idx.add(arr)
    faiss.write_index(idx, index.path)
    index.metadata["ntotal"] = int(idx.ntotal)
    print(f"인덱스에 저장된 벡터 수={idx.ntotal}")
```

### Step 5. `evaluate_retrieval` 컴포넌트

질문 15개를 같은 모델로 임베딩해 상위 `top_k`개 청크를 검색하고 두 지표를 계산합니다.

- **hit_rate**: 검색된 청크 중 정답 문서의 청크가 하나라도 있는 질문의 비율
- **mrr**: 정답 문서의 청크가 처음 등장한 순위의 역수 평균 (1위면 1.0, 2위면 0.5, 없으면 0)

```python
@dsl.component(
    base_image="python:3.10",
    packages_to_install=["sentence-transformers", "faiss-cpu", "numpy"],
    pip_index_urls=TORCH_CPU,
)
def evaluate_retrieval(
    index: Input[Model],
    chunks: Input[Dataset],
    questions_json: str,
    model_name: str,
    top_k: int,
    metrics: Output[Metrics],
):
    import json

    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    idx = faiss.read_index(index.path)
    with open(chunks.path) as f:
        items = json.load(f)
    questions = json.loads(questions_json)

    model = SentenceTransformer(model_name)
    q_vecs = model.encode(
        [q["question"] for q in questions],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    q_vecs = np.asarray(q_vecs, dtype="float32")

    k = min(top_k, len(items))
    _, ids = idx.search(q_vecs, k)

    hits = 0
    rr_sum = 0.0
    for q, row in zip(questions, ids):
        retrieved = [items[i]["doc_id"] for i in row]
        status = "MISS"
        if q["answer_id"] in retrieved:
            rank = retrieved.index(q["answer_id"]) + 1
            hits += 1
            rr_sum += 1.0 / rank
            status = f"HIT(rank {rank})"
        print(f"[{status}] {q['question']}")

    n = len(questions)
    metrics.log_metric("hit_rate", round(hits / n, 4))
    metrics.log_metric("mrr", round(rr_sum / n, 4))
    metrics.log_metric("num_chunks", len(items))
    print(f"hit_rate={hits / n:.4f}, mrr={rr_sum / n:.4f}")
```

### Step 6. 파이프라인 정의와 컴파일

```python
@dsl.pipeline(
    name="rag-retrieval-pipeline",
    description="청킹-임베딩-인덱스-검색 평가 파이프라인",
)
def rag_pipeline(
    chunk_size: int = 200,
    top_k: int = 3,
    model_name: str = "all-MiniLM-L6-v2",
):
    c = load_and_chunk(corpus_json=CORPUS_JSON, chunk_size=chunk_size)
    e = embed(chunks=c.outputs["chunks"], model_name=model_name)
    i = build_index(vectors=e.outputs["vectors"])
    evaluate_retrieval(
        index=i.outputs["index"],
        chunks=c.outputs["chunks"],
        questions_json=QUESTIONS_JSON,
        model_name=model_name,
        top_k=top_k,
    )


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=rag_pipeline,
        package_path="rag_pipeline.yaml",
    )
    print("컴파일 완료: rag_pipeline.yaml")
```

컴파일을 실행합니다.

```bash
python rag_pipeline.py
```

`rag_pipeline.yaml` 파일이 생성되면 성공입니다.

### Step 7. 대시보드에 파이프라인 업로드

1. `http://localhost:8080`에서 왼쪽 메뉴 **Pipelines**를 선택합니다.
2. **Upload pipeline**을 클릭하고 `rag_pipeline.yaml`을 선택합니다.
3. 이름을 `rag-retrieval-pipeline`으로 지정하고 업로드합니다.
4. 파이프라인 상세 화면에서 `load_and_chunk → embed → build_index → evaluate_retrieval` 순서의 그래프(DAG)가 보이는지 확인합니다.

### Step 8. Run 1 실행 (chunk_size=200, top_k=3)

1. 파이프라인 화면에서 **Create run**을 클릭합니다.
2. 실험(Experiment)이 필요하면 `rag-experiment`라는 이름으로 새로 만듭니다.
3. Run 이름을 `run1-chunk200-top3`으로 입력합니다.
4. 파라미터를 `chunk_size=200`, `top_k=3`, `model_name=all-MiniLM-L6-v2`로 둡니다.
5. **Start**를 클릭합니다.

진행 상황을 터미널에서도 볼 수 있습니다.

```bash
kubectl get pods -n kubeflow -w
```

### Step 9. 실행 결과 확인

1. Run 상세 화면에서 4개 노드가 차례로 초록색(Succeeded)이 되는지 확인합니다.
2. `evaluate_retrieval` 노드를 클릭해 **Logs**에서 질문별 `HIT`/`MISS` 출력을 확인합니다.
3. 같은 노드의 출력 아티팩트(Metrics)에서 `hit_rate`와 `mrr` 값을 확인하고 아래 표에 기록합니다.

> 화면의 탭 이름과 배치는 Kubeflow Pipelines 버전에 따라 조금 다를 수 있습니다.

### Step 10. 파라미터를 바꿔 Run 2, 3 실행

같은 파이프라인에서 **Create run**을 다시 눌러 두 번 더 실행합니다. 실험은 `rag-experiment`를 그대로 선택하세요.

| Run | 이름 | chunk_size | top_k |
|---|---|---|---|
| 1 | run1-chunk200-top3 | 200 | 3 |
| 2 | run2-chunk500-top3 | 500 | 3 |
| 3 | run3-chunk500-top5 | 500 | 5 |

### Step 11. Run 비교

1. 왼쪽 메뉴 **Experiments**에서 `rag-experiment`를 엽니다.
2. 세 Run을 체크하고 **Compare runs**를 클릭합니다.
3. 파라미터와 메트릭이 나란히 표시되는지 확인하고, 아래 표를 채웁니다.

| Run | chunk_size | top_k | num_chunks | hit_rate | mrr |
|---|---|---|---|---|---|
| 1 | 200 | 3 | | | |
| 2 | 500 | 3 | | | |
| 3 | 500 | 5 | | | |

### Step 12. 결과 해석

표를 보고 다음 질문에 답해 보세요.

1. `top_k`를 3에서 5로 늘렸을 때(Run 2 → 3) `hit_rate`와 `mrr`은 각각 어떻게 변했습니까? 두 지표의 변화 폭이 다른 이유는 무엇일까요? (힌트: `top_k`를 늘리면 정답이 후보에 들어올 기회는 늘어나지만, 새로 들어온 정답은 4~5위라서 `mrr`에는 조금밖에 기여하지 못합니다.)
2. `chunk_size`를 200에서 500으로 늘렸을 때(Run 1 → 2) `num_chunks`와 검색 품질은 어떻게 달라졌습니까?
3. 실제 서비스에서 `top_k`를 계속 키우면 LLM에 전달되는 프롬프트에는 어떤 문제가 생길까요?
4. 이번 평가는 검색 품질만 측정합니다. 검색이 성공해도 최종 답변이 틀릴 수 있는 이유를 한 가지 들어 보세요.

## 6. 문제 해결

| 증상 | 확인·해결 |
|---|---|
| 대시보드 접속 불가 | `kubectl port-forward` 터미널이 살아 있는지 확인하고, 끊겼다면 다시 실행 |
| 파드가 `Pending` | `kubectl describe pod <파드이름> -n kubeflow`의 Events에서 메모리 부족 등 원인 확인. Docker Desktop의 WSL2 메모리 할당을 늘림 |
| `OOMKilled` | 임베딩 단계에서 메모리 부족. Docker Desktop(WSL2) 메모리를 늘리고 Run을 다시 실행 |
| `ImagePullBackOff` | `python:3.10` 이미지를 내려받지 못한 경우. 인터넷 연결 확인 |
| 컴포넌트가 오래 걸리거나 pip 오류 | 패키지·모델 다운로드 시간이 포함됨. `kubectl logs <파드이름> -n kubeflow`로 진행 상황과 오류 메시지 확인 |
| HuggingFace 모델 다운로드 실패 | 방화벽·프록시 환경을 확인하고 재실행 |
| `pip_index_urls` 관련 오류 | `kfp` 버전이 낮은 경우. `pip install -U kfp` 후 다시 컴파일 |
| 업로드 시 같은 이름 오류 | 파이프라인 이름을 `rag-retrieval-pipeline-v2`처럼 바꿔 업로드 |

## 7. 실습 정리

- 4개 컴포넌트(`load_and_chunk`, `embed`, `build_index`, `evaluate_retrieval`)와 `@dsl.pipeline`으로 RAG 검색 파이프라인을 구성했다.
- `Output[Metrics]`와 `log_metric`으로 기록한 값이 Run마다 표시되어, 파라미터별 검색 품질을 비교할 수 있었다.
- 파이프라인 파라미터만 바꿔 같은 코드를 반복 실행하는 것이 실험 관리의 기본 방식임을 확인했다.

## 8. 선택 과제 (심화)

1. `model_name`을 `paraphrase-MiniLM-L3-v2` 등 다른 소형 모델로 바꿔 Run 4를 실행하고 결과를 비교해 보세요.
2. 질문 3개를 직접 추가(문서 id를 정답으로 지정)하고 `hit_rate` 변화를 관찰해 보세요.
3. 한국어 문서·질문으로 바꾸고 `model_name`을 다국어 모델(`paraphrase-multilingual-MiniLM-L12-v2`)로 지정해 보세요. 모델 크기가 훨씬 커서 실행 시간이 늘어납니다.

## 9. 다음 주 안내

8주차는 중간고사이며, 1~7주차 내용이 범위입니다.
