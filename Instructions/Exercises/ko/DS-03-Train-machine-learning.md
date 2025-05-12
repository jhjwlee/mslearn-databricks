## Azure Databricks에서 머신러닝 시작하기

이 실습에서는 Azure Databricks에서 데이터를 준비하고 머신러닝 모델을 학습시키는 기술을 탐색합니다.

이 실습을 완료하는 데 약 45분이 소요됩니다.

**참고**: Azure Databricks 사용자 인터페이스는 지속적으로 개선되고 있습니다. 이 실습의 지침이 작성된 이후 사용자 인터페이스가 변경되었을 수 있습니다.

### 시작하기 전에 (Before you start)

관리자 수준 액세스 권한이 있는 Azure 구독이 필요합니다.

### Azure Databricks 작업 영역 프로비저닝 (Provision an Azure Databricks workspace)

**팁**: 이미 Azure Databricks 작업 영역이 있는 경우 이 절차를 건너뛰고 기존 작업 영역을 사용할 수 있습니다.

이 실습에는 새 Azure Databricks 작업 영역을 프로비저닝하는 스크립트가 포함되어 있습니다. 스크립트는 Azure 구독에서 이 실습에 필요한 컴퓨팅 코어에 대한 충분한 할당량이 있는 지역에 Premium 등급 Azure Databricks 작업 영역 리소스를 만들려고 시도합니다. 또한 사용자 계정에 구독에서 Azure Databricks 작업 영역 리소스를 만들 수 있는 충분한 권한이 있다고 가정합니다. 할당량 또는 권한 부족으로 스크립트가 실패하면 Azure Portal에서 대화형으로 Azure Databricks 작업 영역을 만들어 볼 수 있습니다.

1.  웹 브라우저에서 Azure Portal (https://portal.azure.com)에 로그인합니다.
2.  페이지 상단의 검색 창 오른쪽에 있는 **[\>\_]** 버튼을 사용하여 Azure Portal에서 새 Cloud Shell을 만들고, **PowerShell** 환경을 선택합니다. Cloud Shell은 아래와 같이 Azure Portal 하단 창에 명령줄 인터페이스를 제공합니다:
    *(이미지: Azure portal with a cloud shell pane)*

    **Note**: 이전에 Bash 환경을 사용하는 Cloud Shell을 만든 경우 PowerShell로 전환하십시오.

    Cloud Shell 창 상단의 구분선을 드래그하거나 창 오른쪽 상단의 **—**, **⤢**, **X** 아이콘을 사용하여 창을 최소화, 최대화 및 닫을 수 있습니다. Azure Cloud Shell 사용에 대한 자세한 내용은 Azure Cloud Shell 설명서를 참조하십시오.

3.  PowerShell 창에서 다음 명령을 입력하여 이 리포지토리를 복제합니다:

    ```powershell
    rm -r mslearn-databricks -f
    git clone https://github.com/MicrosoftLearning/mslearn-databricks
    ```

4.  리포지토리가 복제된 후 다음 명령을 입력하여 `setup.ps1` 스크립트를 실행합니다. 이 스크립트는 사용 가능한 지역에 Azure Databricks 작업 영역을 프로비저닝합니다:

    ```powershell
    ./mslearn-databricks/setup.ps1
    ```

5.  메시지가 표시되면 사용할 구독을 선택합니다 (여러 Azure 구독에 액세스할 수 있는 경우에만 발생).
6.  스크립트가 완료될 때까지 기다립니다. 일반적으로 약 5분 정도 걸리지만 경우에 따라 더 오래 걸릴 수 있습니다. 기다리는 동안 Azure Databricks 설명서의 "What is Databricks Machine Learning?" 문서를 검토하십시오.

---

## Create a cluster (클러스터 생성)

Azure Databricks는 Apache Spark 클러스터를 사용하여 여러 노드에서 데이터를 병렬로 처리하는 분산 처리 플랫폼입니다. 각 클러스터는 작업을 조정하는 드라이버 노드와 처리 작업을 수행하는 작업자 노드로 구성됩니다. 이 실습에서는 랩 환경(리소스가 제한될 수 있음)에서 사용되는 컴퓨팅 리소스를 최소화하기 위해 단일 노드 클러스터를 생성합니다. 프로덕션 환경에서는 일반적으로 여러 작업자 노드가 있는 클러스터를 생성합니다.

**팁**: Azure Databricks 작업 영역에 13.3 LTS ML 이상의 런타임 버전을 가진 클러스터가 이미 있는 경우 이 실습을 완료하는 데 해당 클러스터를 사용하고 이 절차를 건너뛸 수 있습니다.

1.  Azure Portal에서 스크립트에 의해 생성된 **msl-xxxxxxx** 리소스 그룹(또는 기존 Azure Databricks 작업 영역이 포함된 리소스 그룹)으로 이동합니다.
2.  Azure Databricks Service 리소스(설정 스크립트를 사용하여 생성한 경우 **databricks-xxxxxxx**라는 이름)를 선택합니다.
3.  작업 영역의 **Overview** 페이지에서 **Launch Workspace** 버튼을 사용하여 새 브라우저 탭에서 Azure Databricks 작업 영역을 엽니다. 메시지가 표시되면 로그인합니다.

    **팁**: Databricks Workspace 포털을 사용하면 다양한 팁과 알림이 표시될 수 있습니다. 이를 무시하고 제공된 지침에 따라 이 실습의 작업을 완료하십시오.

4.  왼쪽 사이드바에서 **(+) New** 작업을 선택한 다음 **Cluster**를 선택합니다.
5.  **New Cluster** 페이지에서 다음 설정으로 새 클러스터를 만듭니다:
    *   **Cluster name**: *User Name*'s cluster (기본 클러스터 이름)
    *   **Policy**: Unrestricted
    *   **Cluster mode**: Single Node
    *   **Access mode**: Single user (사용자 계정 선택)
    *   **Databricks runtime version**: 다음 조건을 만족하는 최신 비-베타 버전의 ML 에디션을 선택합니다 (Standard 런타임 버전이 아님):
        *   GPU를 사용하지 않음
        *   Scala > 2.11 포함
        *   Spark > 3.4 포함
    *   **Use Photon Acceleration**: 선택 해제
    *   **Node type**: Standard\_D4ds\_v5
    *   **Terminate after** 20 **minutes of inactivity**
6.  클러스터가 생성될 때까지 기다립니다. 1~2분 정도 걸릴 수 있습니다.

    **Note**: 클러스터 시작에 실패하면 Azure Databricks 작업 영역이 프로비저닝된 지역에서 구독의 할당량이 부족할 수 있습니다. 자세한 내용은 "CPU core limit prevents cluster creation"을 참조하십시오. 이 경우 작업 영역을 삭제하고 다른 지역에 새 작업 영역을 만들어 볼 수 있습니다. `./mslearn-databricks/setup.ps1 eastus`와 같이 설정 스크립트에 대한 매개변수로 지역을 지정할 수 있습니다.

**Note:**
*   **Cluster (클러스터)**: Apache Spark에서 분산 처리를 수행하기 위한 컴퓨터 그룹입니다. 클러스터는 작업을 관리하는 *Driver node*와 실제 데이터 처리를 수행하는 여러 *Worker node*로 구성됩니다. 이 실습에서는 비용 및 리소스 절약을 위해 *Single Node* 클러스터(Driver와 Worker 역할을 하나의 노드가 수행)를 사용합니다.
*   **Databricks Runtime Version (Databricks 런타임 버전)**: Spark, 라이브러리 및 기타 구성 요소가 사전 패키징된 환경입니다. "ML" 에디션은 머신러닝 라이브러리(예: scikit-learn, TensorFlow, PyTorch)가 포함되어 있어 머신러닝 작업에 최적화되어 있습니다.
*   **Photon Acceleration**: Databricks에서 개발한 네이티브 벡터화된 쿼리 엔진으로, Apache Spark API와 호환되며 SQL 및 DataFrame 워크로드의 성능을 향상시킵니다. 이 실습에서는 선택하지 않습니다.

## Create a notebook (노트북 생성)

Spark MLLib 라이브러리를 사용하여 머신러닝 모델을 학습시키는 코드를 실행할 것이므로, 첫 번째 단계는 작업 영역에 새 노트북을 만드는 것입니다.

1.  사이드바에서 **(+) New** 링크를 사용하여 **Notebook**을 만듭니다.
2.  기본 노트북 이름(*Untitled Notebook [날짜]*)을 **Machine Learning**으로 변경하고, **Connect** 드롭다운 목록에서 클러스터가 아직 선택되지 않았다면 클러스터를 선택합니다. 클러스터가 실행 중이 아니면 시작하는 데 1분 정도 걸릴 수 있습니다.

**Note:**
*   **Notebook (노트북)**: 코드, 시각화, 설명 텍스트 등을 포함할 수 있는 대화형 웹 기반 환경입니다. 데이터 분석 및 머신러닝 실험에 널리 사용됩니다. Databricks 노트북은 Python, Scala, SQL, R 등 여러 언어를 지원합니다.
*   **Spark MLLib**: Apache Spark의 머신러닝 라이브러리입니다. 일반적인 학습 알고리즘(예: 분류, 회귀, 클러스터링)과 유틸리티(예: 특징 추출, 변환, 파이프라인 구성)를 제공합니다.

## Ingest data (데이터 수집)

이 실습의 시나리오는 남극 펭귄 관찰을 기반으로 하며, 펭귄의 위치와 신체 측정값을 기반으로 관찰된 펭귄의 종을 예측하는 머신러닝 모델을 학습시키는 것을 목표로 합니다.

**인용**: 이 실습에 사용된 펭귄 데이터셋은 Kristen Gorman 박사와 Palmer Station, Antarctica LTER(Long Term Ecological Research Network 회원)가 수집하고 제공한 데이터의 일부입니다.

1.  노트북의 첫 번째 셀에 다음 코드를 입력합니다. 이 코드는 셸 명령을 사용하여 GitHub에서 클러스터가 사용하는 파일 시스템으로 펭귄 데이터를 다운로드합니다.

    ```bash
    %sh
    rm -r /dbfs/ml_lab
    mkdir /dbfs/ml_lab
    wget -O /dbfs/ml_lab/penguins.csv https://raw.githubusercontent.com/MicrosoftLearning/mslearn-databricks/main/data/penguins.csv
    ```

2.  셀 왼쪽의 **▸ Run Cell** 메뉴 옵션을 사용하여 실행합니다. 그런 다음 코드가 실행하는 Spark 작업이 완료될 때까지 기다립니다.

**Note:**
*   `%sh`: Databricks 노트북에서 셸 명령을 실행할 수 있게 하는 매직 명령어입니다.
*   `/dbfs/`: Databricks File System (DBFS)의 경로를 나타냅니다. DBFS는 클러스터 노드에서 액세스할 수 있는 분산 파일 시스템입니다.
*   `wget`: 웹 서버에서 파일을 다운로드하는 명령줄 유틸리티입니다.

## Explore and clean up the data (데이터 탐색 및 정리)

이제 데이터 파일을 수집했으므로 DataFrame에 로드하고 볼 수 있습니다.

1.  기존 코드 셀 아래의 **+** 아이콘을 사용하여 새 코드 셀을 추가합니다. 그런 다음 새 셀에 다음 코드를 입력하고 실행하여 파일에서 데이터를 로드하고 표시합니다.

    ```python
    df = spark.read.format("csv").option("header", "true").load("/ml_lab/penguins.csv")
    display(df)
    ```

    코드는 데이터를 로드하는 데 필요한 Spark 작업을 시작하고, 출력은 `df`라는 이름의 `pyspark.sql.dataframe.DataFrame` 객체입니다. 이 정보는 코드 바로 아래에 표시되며, **▸** 토글을 사용하여 `df: pyspark.sql.dataframe.DataFrame` 출력을 확장하고 포함된 열과 해당 데이터 유형의 세부 정보를 볼 수 있습니다. 이 데이터는 텍스트 파일에서 로드되었고 일부 빈 값을 포함했기 때문에 Spark는 모든 열에 문자열 데이터 유형을 할당했습니다.

    데이터 자체는 남극에서 관찰된 펭귄의 다음 세부 정보 측정값으로 구성됩니다:

    *   **Island**: 펭귄이 관찰된 남극의 섬.
    *   **CulmenLength**: 펭귄 부리(culmen)의 길이 (mm).
    *   **CulmenDepth**: 펭귄 부리의 깊이 (mm).
    *   **FlipperLength**: 펭귄 지느러미의 길이 (mm).
    *   **BodyMass**: 펭귄의 체질량 (g).
    *   **Species**: 펭귄의 종을 나타내는 정수 값:
        *   0: Adelie
        *   1: Gentoo
        *   2: Chinstrap

    이 프로젝트의 목표는 펭귄의 관찰된 특성(특징, features)을 사용하여 종(머신러닝 용어로는 레이블, label)을 예측하는 것입니다.

    일부 관찰에는 특정 특징에 대한 null 또는 "결측" 데이터 값이 포함되어 있습니다. 수집하는 원시 소스 데이터에 이러한 문제가 있는 것은 드문 일이 아니므로, 일반적으로 머신러닝 프로젝트의 첫 번째 단계는 데이터를 철저히 탐색하고 머신러닝 모델 학습에 더 적합하도록 정리하는 것입니다.

2.  셀을 추가하고 다음 코드를 실행하여 `dropna` 메서드를 사용하여 불완전한 데이터가 있는 행을 제거하고, `select` 메서드와 `col` 및 `astype` 함수를 사용하여 데이터에 적절한 데이터 유형을 적용합니다.

    ```python
    from pyspark.sql.types import *
    from pyspark.sql.functions import *

    data = df.dropna().select(col("Island").astype("string"),
                               col("CulmenLength").astype("float"),
                              col("CulmenDepth").astype("float"),
                              col("FlipperLength").astype("float"),
                              col("BodyMass").astype("float"),
                              col("Species").astype("int")
                              )
    display(data)
    ```

    다시 한 번, 반환된 DataFrame(이번에는 `data`라는 이름)의 세부 정보를 토글하여 데이터 유형이 적용되었는지 확인할 수 있으며, 데이터를 검토하여 불완전한 데이터가 포함된 행이 제거되었는지 확인할 수 있습니다.

    실제 프로젝트에서는 데이터의 오류를 수정(또는 제거)하고, 이상치(비정상적으로 크거나 작은 값)를 식별 및 제거하거나, 예측하려는 각 레이블에 대해 합리적으로 동일한 수의 행이 있도록 데이터를 균형있게 조정하는 등 더 많은 탐색 및 데이터 정리가 필요할 수 있습니다.

    **팁**: Spark SQL 참조에서 DataFrame에 사용할 수 있는 메서드 및 함수에 대해 자세히 알아볼 수 있습니다.

**Note:**
*   **DataFrame (데이터프레임)**: Spark에서 데이터를 구조화된 형태로 다루는 핵심적인 분산 데이터 컬렉션입니다. 관계형 데이터베이스의 테이블이나 R/Python Pandas의 DataFrame과 유사하지만, 대용량 데이터 처리를 위해 분산 환경에 최적화되어 있습니다.
*   `spark.read.format("csv").option("header", "true").load("path")`: CSV 파일을 읽어 DataFrame으로 만드는 일반적인 방법입니다. `option("header", "true")`는 CSV 파일의 첫 번째 줄을 열 이름으로 사용하도록 지정합니다.
*   `display(df)`: Databricks에서 DataFrame을 테이블 형식으로 시각화하는 데 사용되는 함수입니다.
*   `dropna()`: 결측치(null 또는 NaN)가 있는 행을 제거하는 함수입니다.
*   `select()`: DataFrame에서 특정 열을 선택하거나 열에 대한 표현식을 적용하는 함수입니다.
*   `col("column_name")`: DataFrame의 특정 열을 참조합니다.
*   `astype("type")`: 열의 데이터 유형을 지정된 유형으로 변환합니다.
*   **Features (특징)**: 모델을 학습시키는 데 사용되는 입력 변수입니다 (예: CulmenLength, Island).
*   **Label (레이블)**: 모델이 예측하려는 대상 변수입니다 (예: Species).
*   **Data Cleaning (데이터 정제)**: 머신러닝 모델의 성능을 향상시키기 위해 데이터의 오류, 결측치, 이상치 등을 처리하는 과정입니다.

## Split the data (데이터 분할)

이 실습에서는 데이터가 이제 정리되어 머신러닝 모델을 학습하는 데 사용할 준비가 되었다고 가정합니다. 우리가 예측하려는 레이블은 특정 범주 또는 클래스(펭귄의 종)이므로 학습해야 하는 머신러닝 모델 유형은 분류(classification) 모델입니다. 분류(숫자 값을 예측하는 데 사용되는 회귀(regression)와 함께)는 예측하려는 레이블에 대한 알려진 값을 포함하는 학습 데이터를 사용하는 지도 학습(supervised machine learning)의 한 형태입니다. 모델을 학습시키는 과정은 본질적으로 특징 값이 알려진 레이블 값과 어떻게 상관되는지 계산하기 위해 알고리즘을 데이터에 맞추는 것입니다. 그런 다음 특징 값만 알고 있는 새로운 관찰에 학습된 모델을 적용하여 레이블 값을 예측하게 할 수 있습니다.

학습된 모델에 대한 신뢰를 보장하기 위해 일반적인 접근 방식은 일부 데이터로만 모델을 학습시키고, 학습된 모델을 테스트하고 예측 정확도를 확인하는 데 사용할 수 있는 알려진 레이블 값이 있는 일부 데이터를 보류하는 것입니다. 이 목표를 달성하기 위해 전체 데이터셋을 두 개의 무작위 하위 집합으로 분할합니다. 데이터의 70%는 학습에 사용하고 30%는 테스트용으로 보류합니다.

1.  다음 코드로 코드 셀을 추가하고 실행하여 데이터를 분할합니다.

    ```python
    splits = data.randomSplit([0.7, 0.3])
    train = splits[0]
    test = splits[1]
    print ("Training Rows:", train.count(), " Testing Rows:", test.count())
    ```

**Note:**
*   **Supervised Machine Learning (지도 학습)**: 입력 데이터(features)와 해당 출력 데이터(labels)가 모두 주어진 상태에서 모델을 학습시키는 방법입니다. 모델은 입력과 출력 간의 관계를 학습합니다.
*   **Classification (분류)**: 지도 학습의 한 유형으로, 데이터 포인트를 미리 정의된 여러 범주(클래스) 중 하나로 할당하는 문제입니다. 펭귄 종 예측은 분류 문제의 예입니다.
*   **Regression (회귀)**: 지도 학습의 다른 유형으로, 연속적인 숫자 값을 예측하는 문제입니다. (예: 주택 가격 예측)
*   **Train/Test Split (학습/테스트 데이터 분할)**: 모델을 학습시키는 데 사용될 데이터(학습 데이터, `train`)와 학습된 모델의 성능을 평가하는 데 사용될 보이지 않는 데이터(테스트 데이터, `test`)로 분할하는 과정입니다. 이는 모델이 학습 데이터에만 과적합(overfitting)되지 않고 새로운 데이터에도 잘 일반화되는지 확인하는 데 중요합니다.
*   `randomSplit([ratio1, ratio2], seed)`: DataFrame을 지정된 비율로 무작위 분할합니다. 선택적으로 `seed`를 지정하여 재현 가능한 분할을 만들 수 있습니다.

## Perform feature engineering (특징 공학 수행)

원시 데이터를 정제한 후, 데이터 과학자는 일반적으로 모델 학습을 위해 데이터를 준비하기 위한 추가 작업을 수행합니다. 이 과정은 일반적으로 특징 공학(feature engineering)으로 알려져 있으며, 최상의 모델을 생성하기 위해 학습 데이터셋의 특징을 반복적으로 최적화하는 작업을 포함합니다. 필요한 특정 특징 수정은 데이터와 원하는 모델에 따라 다르지만, 익숙해져야 할 몇 가지 일반적인 특징 공학 작업이 있습니다.

### Encode categorical features (범주형 특징 인코딩)

머신러닝 알고리즘은 일반적으로 특징과 레이블 간의 수학적 관계를 찾는 데 기반합니다. 즉, 학습 데이터의 특징을 숫자 값으로 정의하는 것이 일반적으로 가장 좋습니다. 경우에 따라 숫자형이 아닌 범주형이고 문자열로 표현되는 일부 특징(예: 우리 데이터셋에서 펭귄 관찰이 발생한 섬의 이름)이 있을 수 있습니다. 그러나 대부분의 알고리즘은 숫자 특징을 기대하므로 이러한 문자열 기반 범주형 값은 숫자로 인코딩해야 합니다. 이 경우 Spark MLLib 라이브러리의 `StringIndexer`를 사용하여 각 개별 섬 이름에 고유한 정수 인덱스를 할당하여 섬 이름을 숫자 값으로 인코딩합니다.

1.  다음 코드를 실행하여 `Island` 범주형 열 값을 숫자 인덱스로 인코딩합니다.

    ```python
    from pyspark.ml.feature import StringIndexer

    indexer = StringIndexer(inputCol="Island", outputCol="IslandIdx")
    indexedData = indexer.fit(train).transform(train).drop("Island")
    display(indexedData)
    ```

    결과에서 각 행에는 섬 이름 대신 관찰이 기록된 섬을 나타내는 정수 값을 가진 `IslandIdx` 열이 있음을 알 수 있습니다.

**Note:**
*   **Feature Engineering (특징 공학)**: 모델의 성능을 향상시키기 위해 원시 데이터로부터 더 유용하거나 의미 있는 특징을 만들거나 변환하는 과정입니다. 도메인 지식과 데이터 탐색을 기반으로 합니다.
*   **Categorical Features (범주형 특징)**: 고정된 수의 가능한 값을 가지는 특징입니다 (예: "Biscoe", "Torgersen", "Dream"). 대부분의 ML 알고리즘은 숫자 입력을 기대하므로 범주형 특징은 숫자로 변환되어야 합니다.
*   `StringIndexer`: `pyspark.ml.feature` 모듈의 변환기(Transformer)입니다. 문자열 레이블의 열을 레이블 인덱스의 열로 인코딩합니다. 가장 빈번한 레이블에 인덱스 0이 할당됩니다.
    *   `inputCol`: 변환할 입력 열 이름.
    *   `outputCol`: 변환된 출력 열 이름.
    *   `fit()`: 입력 데이터(`train`)를 기반으로 `StringIndexerModel`을 학습합니다. 이 단계에서 각 고유 문자열에 인덱스가 할당됩니다.
    *   `transform()`: 학습된 모델을 사용하여 데이터를 변환합니다.
    *   `drop("Island")`: 원래의 문자열 `Island` 열을 제거합니다.

### Normalize (scale) numeric features (수치형 특징 정규화 (스케일링))

이제 데이터의 숫자 값에 주목해 봅시다. 이러한 값(`CulmenLength`, `CulmenDepth`, `FlipperLength`, `BodyMass`)은 모두 일종의 측정이지만 서로 다른 척도(scale)를 가지고 있습니다. 모델을 학습할 때 측정 단위는 다른 관찰 간의 상대적 차이만큼 중요하지 않으며, 더 큰 숫자로 표현되는 특징이 종종 모델 학습 알고리즘을 지배하여 예측을 계산할 때 특징의 중요성을 왜곡할 수 있습니다. 이를 완화하기 위해 숫자 특징 값을 모두 동일한 상대적 척도(예: 0.0과 1.0 사이의 소수 값)로 정규화하는 것이 일반적입니다.

이를 위해 사용할 코드는 이전에 수행한 범주형 인코딩보다 약간 더 복잡합니다. 여러 열 값을 동시에 스케일링해야 하므로, 사용하는 기술은 모든 숫자 특징을 포함하는 단일 열(본질적으로 배열인 벡터)을 만든 다음, 스케일러를 적용하여 동등한 정규화된 값을 가진 새 벡터 열을 생성하는 것입니다.

1.  다음 코드를 사용하여 숫자 특징을 정규화하고 정규화 전후의 벡터 열을 비교합니다.

    ```python
    from pyspark.ml.feature import VectorAssembler, MinMaxScaler

    # Create a vector column containing all numeric features
    numericFeatures = ["CulmenLength", "CulmenDepth", "FlipperLength", "BodyMass"]
    numericColVector = VectorAssembler(inputCols=numericFeatures, outputCol="numericFeatures")
    vectorizedData = numericColVector.transform(indexedData)

    # Use a MinMax scaler to normalize the numeric values in the vector
    minMax = MinMaxScaler(inputCol = numericColVector.getOutputCol(), outputCol="normalizedFeatures")
    scaledData = minMax.fit(vectorizedData).transform(vectorizedData)

    # Display the data with numeric feature vectors (before and after scaling)
    compareNumerics = scaledData.select("numericFeatures", "normalizedFeatures")
    display(compareNumerics)
    ```

    결과의 `numericFeatures` 열에는 각 행에 대한 벡터가 포함됩니다. 벡터에는 네 개의 스케일링되지 않은 숫자 값(펭귄의 원래 측정값)이 포함됩니다. **▸** 토글을 사용하여 개별 값을 더 명확하게 볼 수 있습니다.

    `normalizedFeatures` 열에도 각 펭귄 관찰에 대한 벡터가 포함되지만, 이번에는 벡터의 값이 각 측정값의 최소값과 최대값을 기준으로 상대적 척도로 정규화됩니다.

**Note:**
*   **Numeric Features (수치형 특징)**: 연속적이거나 이산적인 숫자 값을 가지는 특징입니다 (예: CulmenLength, BodyMass).
*   **Normalization/Scaling (정규화/스케일링)**: 수치형 특징의 범위를 표준 범위(예: 0에서 1 또는 평균 0, 표준편차 1)로 변환하는 과정입니다. 이는 서로 다른 척도를 가진 특징들이 모델 학습에 미치는 영향을 균등하게 하고, 일부 알고리즘(예: 경사 하강법 사용)의 수렴 속도를 높이는 데 도움이 됩니다.
*   `VectorAssembler`: 여러 열을 단일 벡터 열로 결합하는 변환기입니다. 많은 ML 알고리즘은 특징을 단일 벡터로 입력받습니다.
    *   `inputCols`: 결합할 열 이름의 리스트.
    *   `outputCol`: 생성될 벡터 열의 이름.
*   `MinMaxScaler`: 특징을 지정된 최소값과 최대값(기본값은 [0, 1]) 사이로 변환하여 각 특징을 선형적으로 스케일링합니다.
    *   `X_scaled = (X - X_min) / (X_max - X_min)`
    *   `fit()`: 입력 데이터에서 각 특징의 최소값과 최대값을 계산합니다.
    *   `transform()`: 계산된 최소/최대값을 사용하여 데이터를 스케일링합니다.

### Prepare features and labels for training (학습을 위한 특징 및 레이블 준비)

이제 모든 것을 통합하여 모든 특징(인코딩된 범주형 섬 이름과 정규화된 펭귄 측정값)을 포함하는 단일 열과 모델이 예측하도록 학습할 클래스 레이블(펭귄 종)을 포함하는 다른 열을 만듭니다.

1.  다음 코드를 실행합니다:

    ```python
    featVect = VectorAssembler(inputCols=["IslandIdx", "normalizedFeatures"], outputCol="featuresVector")
    preppedData = featVect.transform(scaledData)[col("featuresVector").alias("features"), col("Species").alias("label")]
    display(preppedData)
    ```

    `features` 벡터에는 5개의 값(인코딩된 섬과 정규화된 부리 길이, 부리 깊이, 지느러미 길이, 체질량)이 포함됩니다. `label`에는 펭귄 종의 클래스를 나타내는 간단한 정수 코드가 포함됩니다.

**Note:**
*   Spark MLLib의 많은 알고리즘은 입력 특징을 `features`라는 이름의 단일 벡터 열로, 예측 대상을 `label`이라는 이름의 열로 기대합니다.
*   `alias("new_name")`: 열의 이름을 변경합니다.

## Train a machine learning model (머신러닝 모델 학습)

이제 학습 데이터가 준비되었으므로 이를 사용하여 모델을 학습시킬 수 있습니다. 모델은 특징과 레이블 간의 관계를 설정하려는 알고리즘을 사용하여 학습됩니다. 이 경우 클래스 범주를 예측하는 모델을 학습시키고자 하므로 분류 알고리즘을 사용해야 합니다. 분류에는 많은 알고리즘이 있습니다. 잘 알려진 로지스틱 회귀(logistic regression)부터 시작해 보겠습니다. 이 알고리즘은 각 클래스 레이블 값에 대한 확률을 예측하는 로지스틱 계산에서 특징 데이터에 적용될 수 있는 최적의 계수를 반복적으로 찾으려고 시도합니다. 모델을 학습시키려면 로지스틱 회귀 알고리즘을 학습 데이터에 맞춥니다(fit).

1.  다음 코드를 실행하여 모델을 학습시킵니다.

    ```python
    from pyspark.ml.classification import LogisticRegression

    lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10, regParam=0.3)
    model = lr.fit(preppedData)
    print ("Model trained!")
    ```

    대부분의 알고리즘은 모델 학습 방식을 일부 제어할 수 있는 매개변수를 지원합니다. 이 경우 로지스틱 회귀 알고리즘은 특징 벡터를 포함하는 열과 알려진 레이블을 포함하는 열을 식별해야 하며, 로지스틱 계산을 위한 최적의 계수를 찾는 데 수행되는 최대 반복 횟수와 모델이 과적합(즉, 학습 데이터에서는 잘 작동하지만 새 데이터에 적용할 때 일반화가 잘 되지 않는 로지스틱 계산을 설정하는 것)되는 것을 방지하는 데 사용되는 정규화 매개변수를 지정할 수 있습니다.

**Note:**
*   `LogisticRegression`: 이진 또는 다중 클래스 분류 문제에 사용되는 알고리즘입니다. 이름에 "회귀"가 있지만 분류 알고리즘입니다. 특징들의 선형 결합을 로지스틱(시그모이드) 함수에 통과시켜 클래스에 속할 확률을 예측합니다.
    *   `labelCol`: 레이블 열의 이름.
    *   `featuresCol`: 특징 벡터 열의 이름.
    *   `maxIter`: 최적화 알고리즘의 최대 반복 횟수.
    *   `regParam`: 정규화 매개변수 (L1 또는 L2). 과적합을 방지하는 데 도움이 됩니다. 값이 클수록 정규화 강도가 커집니다.
*   `fit(dataset)`: Estimator(여기서는 `LogisticRegression`)를 입력 데이터셋에 학습시켜 Model(여기서는 `LogisticRegressionModel`)을 생성합니다.

## Test the model (모델 테스트)

이제 학습된 모델이 있으므로 보류했던 데이터로 테스트할 수 있습니다. 이를 수행하기 전에 학습 데이터에 적용했던 것과 동일한 특징 공학 변환(이 경우 섬 이름 인코딩 및 측정값 정규화)을 테스트 데이터에 수행해야 합니다. 그런 다음 모델을 사용하여 테스트 데이터의 특징에 대한 레이블을 예측하고 예측된 레이블을 실제 알려진 레이블과 비교할 수 있습니다.

1.  다음 코드를 사용하여 테스트 데이터를 준비한 다음 예측을 생성합니다:

    ```python
    # Prepare the test data
    indexedTestData = indexer.fit(test).transform(test).drop("Island") # 주의: 실제로는 train 데이터로 fit한 indexer를 사용해야 함
    vectorizedTestData = numericColVector.transform(indexedTestData)
    scaledTestData = minMax.fit(vectorizedTestData).transform(vectorizedTestData) # 주의: 실제로는 train 데이터로 fit한 scaler를 사용해야 함
    preppedTestData = featVect.transform(scaledTestData)[col("featuresVector").alias("features"), col("Species").alias("label")]

    # Get predictions
    prediction = model.transform(preppedTestData)
    predicted = prediction.select("features", "probability", col("prediction").astype("Int"), col("label").alias("trueLabel"))
    display(predicted)
    ```
    **중요 수정 제안**:
    실제 워크플로우에서는 `indexer`와 `minMax` (스케일러)를 테스트 데이터에 대해 다시 `fit`하지 않습니다. 학습 데이터(`train`)로 `fit`된 `indexer`와 `minMax` 객체를 사용하여 테스트 데이터(`test`)를 `transform`해야 합니다. 이는 학습 과정에서 학습된 매핑(예: 어떤 섬 이름이 어떤 인덱스에 매핑되는지, 어떤 최소/최대 값으로 스케일링되는지)을 테스트 데이터에도 동일하게 적용하기 위함입니다. 그렇지 않으면 데이터 누수(data leakage)가 발생하거나 일관성 없는 변환이 이루어질 수 있습니다.
    이 실습에서는 코드가 간결성을 위해 이렇게 작성되었을 수 있지만, 실제 적용 시에는 주의해야 합니다. 파이프라인을 사용하면 이러한 문제를 더 쉽게 관리할 수 있습니다 (아래 섹션 참조).

    결과에는 다음 열이 포함됩니다:

    *   `features`: 테스트 데이터셋에서 준비된 특징 데이터.
    *   `probability`: 각 클래스에 대해 모델이 계산한 확률. 세 개의 클래스가 있으므로 세 개의 확률 값을 포함하는 벡터로 구성되며, 총합은 1.0입니다 (펭귄이 세 종 중 하나에 속할 확률이 100%라고 가정).
    *   `prediction`: 예측된 클래스 레이블 (가장 높은 확률을 가진 레이블).
    *   `trueLabel`: 테스트 데이터의 실제 알려진 레이블 값.

    모델의 효과를 평가하기 위해 이러한 결과에서 예측된 레이블과 실제 레이블을 단순히 비교할 수 있습니다. 그러나 모델 평가기(이 경우 다중 클래스(여러 가능한 클래스 레이블이 있기 때문) 분류 평가기)를 사용하여 더 의미 있는 메트릭을 얻을 수 있습니다.

2.  다음 코드를 사용하여 테스트 데이터의 결과를 기반으로 분류 모델에 대한 평가 메트릭을 가져옵니다:

    ```python
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator

    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

    # Simple accuracy
    accuracy = evaluator.evaluate(prediction, {evaluator.metricName:"accuracy"})
    print("Accuracy:", accuracy)

    # Individual class metrics
    labels = [0,1,2] # Adelie, Gentoo, Chinstrap
    print("\nIndividual class metrics:")
    for label in sorted(labels):
        print ("Class %s" % (label))

        # Precision
        precision = evaluator.evaluate(prediction, {evaluator.metricLabel:label,
                                                    evaluator.metricName:"precisionByLabel"})
        print("\tPrecision:", precision)

        # Recall
        recall = evaluator.evaluate(prediction, {evaluator.metricLabel:label,
                                                 evaluator.metricName:"recallByLabel"})
        print("\tRecall:", recall)

        # F1 score
        f1 = evaluator.evaluate(prediction, {evaluator.metricLabel:label,
                                             evaluator.metricName:"fMeasureByLabel"})
        print("\tF1 Score:", f1)

    # Weighted (overall) metrics
    overallPrecision = evaluator.evaluate(prediction, {evaluator.metricName:"weightedPrecision"})
    print("Overall Precision:", overallPrecision)
    overallRecall = evaluator.evaluate(prediction, {evaluator.metricName:"weightedRecall"})
    print("Overall Recall:", overallRecall)
    overallF1 = evaluator.evaluate(prediction, {evaluator.metricName:"weightedFMeasure"})
    print("Overall F1 Score:", overallF1)
    ```

    다중 클래스 분류에 대해 계산되는 평가 메트릭은 다음과 같습니다:

    *   **Accuracy**: 전체 예측 중 올바른 예측의 비율.
    *   **Per-class metrics (클래스별 메트릭)**:
        *   **Precision (정밀도)**: 해당 클래스로 예측한 것 중 실제로 해당 클래스인 샘플의 비율. (TP / (TP + FP))
        *   **Recall (재현율)**: 실제 해당 클래스인 샘플 중 모델이 올바르게 해당 클래스로 예측한 샘플의 비율. (TP / (TP + FN))
        *   **F1 score (F1 점수)**: 정밀도와 재현율의 조화 평균. (2 \* (Precision \* Recall) / (Precision + Recall))
    *   **Combined (weighted) precision, recall, and F1 metrics for all classes (모든 클래스에 대한 가중 평균 정밀도, 재현율, F1 메트릭).**

    **Note**: 처음에는 전체 정확도 메트릭이 모델의 예측 성능을 평가하는 가장 좋은 방법을 제공하는 것처럼 보일 수 있습니다. 그러나 이것을 고려해 보십시오. 연구 지역의 펭귄 개체 수의 95%가 Gentoo 펭귄이라고 가정해 보겠습니다. 항상 레이블 1(Gentoo 클래스)을 예측하는 모델은 0.95의 정확도를 가질 것입니다. 그렇다고 해서 특징을 기반으로 펭귄 종을 예측하는 훌륭한 모델이라는 의미는 아닙니다! 이것이 데이터 과학자들이 분류 모델이 각 가능한 클래스 레이블에 대해 얼마나 잘 예측하는지 더 잘 이해하기 위해 추가 메트릭을 탐색하는 경향이 있는 이유입니다.

**Note:**
*   `model.transform(dataset)`: 학습된 Model을 사용하여 새 데이터셋에 대한 예측을 생성합니다. 변환기(Transformer)의 일부입니다.
*   `MulticlassClassificationEvaluator`: 다중 클래스 분류 모델의 성능을 평가하는 유틸리티입니다.
    *   `labelCol`: 실제 레이블이 있는 열.
    *   `predictionCol`: 모델이 예측한 레이블이 있는 열.
    *   `metricName`: 평가할 메트릭의 이름 (예: "accuracy", "f1", "weightedPrecision", "weightedRecall").
    *   `metricLabel` (선택 사항): 특정 클래스에 대한 메트릭을 계산할 때 해당 클래스의 레이블 값을 지정합니다.
*   **Accuracy (정확도)**: (TP + TN) / (TP + TN + FP + FN). 전체 샘플 중 올바르게 분류된 샘플의 비율. 데이터가 불균형할 경우(예: 특정 클래스가 매우 드문 경우) مضلل일 수 있습니다.
*   **Precision (정밀도)**: 모델이 "양성(Positive)"이라고 예측한 것들 중 실제로 양성인 것의 비율. (TP / (TP + FP)). 모델의 예측이 얼마나 정확한지를 나타냅니다.
*   **Recall (재현율, Sensitivity)**: 실제 양성인 것들 중 모델이 "양성"이라고 올바르게 예측한 것의 비율. (TP / (TP + FN)). 모델이 실제 양성 샘플을 얼마나 잘 찾아내는지를 나타냅니다.
*   **F1-Score**: 정밀도와 재현율의 조화 평균. 두 지표가 모두 중요할 때 사용되며, 불균형 데이터셋에서 유용합니다.
    (TP: True Positive, TN: True Negative, FP: False Positive, FN: False Negative)

## Use a pipeline (파이프라인 사용)

필요한 특징 공학 단계를 수행한 다음 알고리즘을 데이터에 맞춰 모델을 학습시켰습니다. 예측을 생성하기 위해 (추론(inferencing)이라고 함) 테스트 데이터와 함께 모델을 사용하려면 테스트 데이터에 동일한 특징 공학 단계를 적용해야 했습니다. 모델을 구축하고 사용하는 더 효율적인 방법은 데이터를 준비하는 데 사용되는 변환기(transformers)와 이를 학습하는 데 사용되는 모델(estimator)을 파이프라인(pipeline)에 캡슐화하는 것입니다.

1.  다음 코드를 사용하여 데이터 준비 및 모델 학습 단계를 캡슐화하는 파이프라인을 만듭니다:

    ```python
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import StringIndexer, VectorAssembler, MinMaxScaler
    from pyspark.ml.classification import LogisticRegression

    catFeature = "Island"
    numFeatures = ["CulmenLength", "CulmenDepth", "FlipperLength", "BodyMass"]

    # Define the feature engineering and model training algorithm steps
    catIndexer = StringIndexer(inputCol=catFeature, outputCol=catFeature + "Idx", handleInvalid="keep") # handleInvalid 추가 권장
    numVector = VectorAssembler(inputCols=numFeatures, outputCol="numericFeatures")
    numScaler = MinMaxScaler(inputCol = numVector.getOutputCol(), outputCol="normalizedFeatures")
    featureVector = VectorAssembler(inputCols=["IslandIdx", "normalizedFeatures"], outputCol="Features")
    algo = LogisticRegression(labelCol="Species", featuresCol="Features", maxIter=10, regParam=0.3)

    # Chain the steps as stages in a pipeline
    pipeline = Pipeline(stages=[catIndexer, numVector, numScaler, featureVector, algo])

    # Use the pipeline to prepare data and fit the model algorithm
    model = pipeline.fit(train)
    print ("Model trained!")
    ```

    특징 공학 단계가 이제 파이프라인에 의해 학습된 모델에 캡슐화되었으므로, 각 변환을 적용할 필요 없이 테스트 데이터와 함께 모델을 사용할 수 있습니다 (모델에 의해 자동으로 적용됨).
    **권장 사항**: `StringIndexer`에 `handleInvalid="keep"` (또는 "skip", "error") 옵션을 추가하는 것이 좋습니다. 이는 학습 데이터에 없던 새로운 범주형 값이 테스트 데이터에 나타날 경우 처리 방법을 지정합니다. "keep"은 새 값을 위한 추가 인덱스를 만듭니다.

2.  다음 코드를 사용하여 파이프라인을 테스트 데이터에 적용합니다:

    ```python
    prediction = model.transform(test)
    predicted = prediction.select("Features", "probability", col("prediction").astype("Int"), col("Species").alias("trueLabel"))
    display(predicted)
    ```

**Note:**
*   **Pipeline (파이프라인)**: ML 워크플로우를 구성하는 일련의 단계를 캡슐화합니다. 각 단계는 Transformer(데이터를 변환) 또는 Estimator(데이터로부터 학습하여 Transformer를 생성)입니다.
    *   `stages`: 파이프라인에 포함될 Transformer 및 Estimator 객체의 리스트. 순서대로 실행됩니다.
*   `pipeline.fit(trainingData)`: 전체 파이프라인을 학습 데이터에 맞춥니다. 파이프라인 내의 모든 Estimator가 학습되고, Transformer는 그대로 유지됩니다. 결과로 `PipelineModel`이 반환됩니다.
*   `pipelineModel.transform(testData)`: 학습된 `PipelineModel`을 사용하여 테스트 데이터를 변환하고 예측을 생성합니다. 파이프라인의 모든 단계(학습된 Transformer 및 원본 Transformer)가 순서대로 적용됩니다. 파이프라인을 사용하면 학습 데이터와 테스트 데이터에 동일한 전처리 단계를 일관되게 적용하기가 훨씬 쉬워집니다.

## Try a different algorithm (다른 알고리즘 시도)

지금까지 로지스틱 회귀 알고리즘을 사용하여 분류 모델을 학습시켰습니다. 파이프라인의 해당 단계를 변경하여 다른 알고리즘을 시도해 보겠습니다.

1.  다음 코드를 실행하여 의사 결정 트리(Decision Tree) 알고리즘을 사용하는 파이프라인을 만듭니다:

    ```python
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import StringIndexer, VectorAssembler, MinMaxScaler
    from pyspark.ml.classification import DecisionTreeClassifier # 변경된 부분

    catFeature = "Island"
    numFeatures = ["CulmenLength", "CulmenDepth", "FlipperLength", "BodyMass"]

    # Define the feature engineering and model steps
    catIndexer = StringIndexer(inputCol=catFeature, outputCol=catFeature + "Idx", handleInvalid="keep")
    numVector = VectorAssembler(inputCols=numFeatures, outputCol="numericFeatures")
    numScaler = MinMaxScaler(inputCol = numVector.getOutputCol(), outputCol="normalizedFeatures")
    featureVector = VectorAssembler(inputCols=["IslandIdx", "normalizedFeatures"], outputCol="Features")
    algo = DecisionTreeClassifier(labelCol="Species", featuresCol="Features", maxDepth=10) # 변경된 부분

    # Chain the steps as stages in a pipeline
    pipeline = Pipeline(stages=[catIndexer, numVector, numScaler, featureVector, algo])

    # Use the pipeline to prepare data and fit the model algorithm
    model = pipeline.fit(train)
    print ("Model trained!")
    ```

    이번에는 파이프라인에 이전과 동일한 특징 준비 단계가 포함되지만 의사 결정 트리 알고리즘을 사용하여 모델을 학습시킵니다.

2.  다음 코드를 실행하여 새 파이프라인을 테스트 데이터와 함께 사용합니다:

    ```python
    # Get predictions
    prediction = model.transform(test)
    predicted = prediction.select("Features", "probability", col("prediction").astype("Int"), col("Species").alias("trueLabel"))

    # Generate evaluation metrics
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator

    evaluator = MulticlassClassificationEvaluator(labelCol="Species", predictionCol="prediction") # labelCol을 "Species"로 수정해야 함

    # Simple accuracy
    accuracy = evaluator.evaluate(prediction, {evaluator.metricName:"accuracy"})
    print("Accuracy:", accuracy)

    # Class metrics
    labels = [0,1,2]
    print("\nIndividual class metrics:")
    for label in sorted(labels):
        print ("Class %s" % (label))

        # Precision
        precision = evaluator.evaluate(prediction, {evaluator.metricLabel:label,
                                                        evaluator.metricName:"precisionByLabel"})
        print("\tPrecision:", precision)

        # Recall
        recall = evaluator.evaluate(prediction, {evaluator.metricLabel:label,
                                                 evaluator.metricName:"recallByLabel"})
        print("\tRecall:", recall)

        # F1 score
        f1 = evaluator.evaluate(prediction, {evaluator.metricLabel:label,
                                             evaluator.metricName:"fMeasureByLabel"})
        print("\tF1 Score:", f1)

    # Weighed (overall) metrics
    overallPrecision = evaluator.evaluate(prediction, {evaluator.metricName:"weightedPrecision"})
    print("Overall Precision:", overallPrecision)
    overallRecall = evaluator.evaluate(prediction, {evaluator.metricName:"weightedRecall"})
    print("Overall Recall:", overallRecall)
    overallF1 = evaluator.evaluate(prediction, {evaluator.metricName:"weightedFMeasure"})
    print("Overall F1 Score:", overallF1)
    ```
    **수정 사항**: `MulticlassClassificationEvaluator`의 `labelCol`은 예측 결과 DataFrame에서 실제 레이블이 있는 열의 이름이어야 합니다. `predicted` DataFrame에서는 이 열의 이름이 `trueLabel` (원래 `Species` 열에서 alias됨)이므로 `evaluator = MulticlassClassificationEvaluator(labelCol="trueLabel", predictionCol="prediction")`로 설정하거나, 또는 예측 생성 시 `predicted = prediction.select("Features", "probability", col("prediction").astype("Int"), col("Species"))`로 유지하고 `evaluator = MulticlassClassificationEvaluator(labelCol="Species", predictionCol="prediction")`을 사용해야 합니다. 제공된 코드는 후자를 따르고 있으므로 `labelCol="Species"`가 맞습니다.

**Note:**
*   `DecisionTreeClassifier`: 분류 문제에 사용되는 트리 기반 알고리즘입니다. 데이터를 반복적으로 분할하여 트리 구조를 만듭니다. 각 내부 노드는 특징에 대한 테스트를 나타내고, 각 리프 노드는 클래스 레이블을 나타냅니다.
    *   `maxDepth`: 트리의 최대 깊이. 트리가 너무 깊어지면 과적합될 수 있습니다.

## Save the model (모델 저장)

실제로는 다른 알고리즘(및 매개변수)으로 모델을 반복적으로 학습시켜 데이터에 가장 적합한 모델을 찾을 것입니다. 지금은 학습시킨 의사 결정 트리 모델을 사용하겠습니다. 나중에 새로운 펭귄 관찰에 사용할 수 있도록 저장해 보겠습니다.

1.  다음 코드를 사용하여 모델을 저장합니다:

    ```python
    model.save("/models/penguin.model")
    ```

이제 밖에 나가서 새로운 펭귄을 발견했을 때 저장된 모델을 로드하고 특징 측정값을 기반으로 펭귄의 종을 예측하는 데 사용할 수 있습니다. 새 데이터에서 예측을 생성하기 위해 모델을 사용하는 것을 추론(inferencing)이라고 합니다.

1.  다음 코드를 실행하여 모델을 로드하고 새 펭귄 관찰에 대한 종을 예측하는 데 사용합니다:

    ```python
    from pyspark.ml.pipeline import PipelineModel

    persistedModel = PipelineModel.load("/models/penguin.model")

    newData = spark.createDataFrame ([{"Island": "Biscoe",
                                      "CulmenLength": 47.6,
                                      "CulmenDepth": 14.5,
                                      "FlipperLength": 215.0, # 실수형으로 일치
                                      "BodyMass": 5400.0, # 실수형으로 일치
                                      "Species": 0 # 예측용이므로 이 열은 실제로는 불필요, 모델은 Species를 예측
                                      }])


    predictions = persistedModel.transform(newData)
    display(predictions.select("Island", "CulmenDepth", "CulmenLength", "FlipperLength", "BodyMass", col("prediction").alias("PredictedSpecies")))
    ```
    **수정 사항**: `newData` DataFrame을 생성할 때, 숫자형 특징들은 파이프라인 내 `VectorAssembler`가 숫자형을 기대하므로 float 타입으로 맞춰주는 것이 좋습니다(예: `215.0`, `5400.0`). `Species` 열은 모델이 예측할 대상이므로 `newData`에 포함될 필요는 없지만, 파이프라인의 `LogisticRegression` 또는 `DecisionTreeClassifier`가 `labelCol="Species"`를 사용하므로 스키마 일치를 위해 임의의 값(예: `0` 또는 `None`으로 하고 `IntegerType` 지정)을 넣어주거나, 파이프라인의 마지막 예측 단계에서 `Species` 열을 요구하지 않도록 파이프라인 정의를 수정해야 할 수 있습니다. 이 실습의 맥락에서는 `Species` 열을 제공하면 모델이 이를 무시하고 `featuresCol`만 사용하여 예측을 생성합니다.

**Note:**
*   `model.save(path)`: 학습된 `PipelineModel` (또는 다른 MLLib 모델)을 지정된 경로(일반적으로 DBFS)에 저장합니다. 모델의 메타데이터와 학습된 매개변수가 저장됩니다.
*   `PipelineModel.load(path)`: 저장된 `PipelineModel`을 로드합니다.
*   `spark.createDataFrame(data, schema)`: Python 객체 리스트(예: 딕셔너리 리스트)로부터 Spark DataFrame을 만듭니다. 스키마를 명시적으로 제공하거나 Spark가 추론하도록 할 수 있습니다.
*   **Inferencing (추론)**: 학습된 머신러닝 모델을 사용하여 새로운, 보이지 않는 데이터에 대한 예측을 만드는 과정입니다.

## Clean up (정리)

Azure Databricks 포털의 **Compute** 페이지에서 클러스터를 선택하고 **■ Terminate**를 선택하여 종료합니다.

Azure Databricks 탐색을 마쳤다면 불필요한 Azure 비용을 피하고 구독의 용량을 확보하기 위해 만든 리소스를 삭제할 수 있습니다.

---
이것으로 Azure Databricks 머신러닝 시작하기 실습의 한국어 버전 번역 및 설명이 완료되었습니다. 학습에 도움이 되길 바랍니다!
