

<div align="center">
<img src="https://raw.githubusercontent.com/Marigoldwu/PyDGC/master/assets/logo.png" border="0" width=600px/>
</div>
<div align="center">
  <a href="#PyDGC">Overview</a> |
  <a href="#DGCBench">DGCBench</a> |
  <a href="#Installation">Installation</a> |
  <a href="#Examples">Examples</a> |
  <a href="#Docs">Docs</a> |
  <a href="#Citation">Citation</a> 
</div>

# PyDGC

**PyDGC**, a flexible and extensible Python library  for deep graph clustering (DGC), is compatible with frameworks such as PyG and OGB. It supports the easy integration of new models and datasets, facilitating the rapid development, reproduction, and fair comparison of DGC methods.

## News

- *2025.05*: Release source code of PyDGC.

## What is DGC?

Deep graph clustering, which aims to reveal the underlying graph structure and divide the nodes into different groups, has attracted intensive attention in recent years. 

More details can be found in the [survey](https://arxiv.org/abs/2211.12875) paper. Please click [here](./articles) to view the comprehensive archive of papers.

Timeline of representative models.

<div align="center">
<img src="https://raw.githubusercontent.com/Marigoldwu/PyDGC/master/assets/Roadmap.png" border="0" width=800px/>
</div>

## DGCBench

**DGCBench** encompasses 12 diverse datasets with different characteristics and 12 state-of-the-art methods from all major paradigms. By integrating them into a standardized pipeline, we ensure fair, reproducible, and comprehensive evaluations across multiple dimensions. 

 ## Features

- Integration of multiple deep graph clustering models. [Supported Models](#Supported-Models)
- Support for various graph datasets from PyG and OGB. [Supported Datasets](#Supported-Datasets)
- Model evaluation and visualization capabilities.
- Standardized Pipeline.

### Overview of Pipeline

<div align="center">
<img src="https://raw.githubusercontent.com/Marigoldwu/PyDGC/master/assets/Pipelines.png" border="0" width=800px/>
</div>

# Installation

- Install with Pip

  coming soon...

- Installation for local development

  ```bash
  git clone https://github.com/Marigoldwu/PyDGC.git
  cd PyDGC
  pip install -e .
  ```

# Examples

## Reproduce built-in models

Take GAE as an example:

```bash
cd PyDGC/example/pipelines/gae
python run.py
```

You can also specify arguments in the command line:

```bash
python run.py --dataset_name CORA -eval_each
```

Other optional arguments:

```bash
--cfg_file_path YourPath  # path of corresponding configurations file
--flag FlagContent  # Descriptions
--drop_edge float  # probability of dropping edges
--drop_feature float  # probability of dropping features
--add_edge float  # probability of adding edges
--add_noise float  # standard deviation of Gaussian Noise
-pretrain  # only run the pretraining stage in the model
```

## Develop your own DGC model

```python
from pydgc.models import DGCModel

class MyModel(DGCModel):
    def __init__(self, logger, cfg):
        super(MyModel).__init__(logger, cfg)
        your_model = ...  # Your model
        
        self.loss_curve = []
        self.nmi_curve = []
        self.best_embedding = None
        self.best_predicted_labels = None
        self.best_results = {'ACC': -1}
    
    def forward(self, data):
        ...  # forward process
        return something
    # If needed
    def loss(self, *args, **kwargs):
    # If needed
    def pretrain(self, data, cfg, flag):
    
    def train_model(self, data, cfg, flag):
    
    def get_embedding(self, data):
    
    def clustering(self, data):
        embedding = self.get_embedding(data)
        # clustering
        return embedding, labels_, clustering_centers
    
    def evaluate(self, data):
        embedding, predicted_labels, clustering_centers = self.clustering(data)
        ground_truth = data.y.numpy()
        metric = DGCMetric(ground_truth, predicted_labels.numpy(), embedding, data.edge_index)
        results = metric.evaluate_one_epoch(self.logger, self.cfg.evaluate)
        return embedding, predicted_labels, results

```

## Develop your own DGC pipeline

```python
from pydgc.pipelines import BasePipeline
from pydgc.utils import perturb_data
import MyModel  # import your own model

class MyPipeline(BasePipeline):
    def __init__(self, args):
        super(MyPipeline).__init__(args)
    
    def augmentation(self):
        self.data = perturb_data(self.data, self.cfg.dataset.augmentation)
        # other augmentations if needed
        
    def build_model(self):
        model = MyModel(self.logger, self.cfg)
        self.logger.model_info(model)
        return model
```

# Supported Models

| No.  | Model     | Paper | Source Code |
| :----: | :---------: | :-----: | :-----------: |
| 1    | GAE       |[Variational Graph Auto-Encoders](https://arxiv.org/abs/1611.07308)|[code](https://github.com/tkipf/gae)|
| 2    | GAE_SSC   | - | - |
| 3    | DAEGC     |[Attributed graph clustering: A deep attentional embedding approach](https://arxiv.org/pdf/1906.06532)|[code](https://github.com/Tiger101010/DAEGC)|
| 4    | SDCN      |[Structural Deep Clustering Network](https://arxiv.org/pdf/2002.01633)|[code](https://github.com/bdy9527/SDCN)|
| 5    | DFCN      |[Deep Fusion Clustering Network](https://ojs.aaai.org/index.php/AAAI/article/view/17198/17005)|[code](https://github.com/WxTu/DFCN)|
| 6    | DCRN      |[Deep Graph Clustering via Dual Correlation Reduction](https://www.aaai.org/AAAI22Papers/AAAI-5928.LiuY.pdf)|[code](https://github.com/yueliu1999/DCRN)|
| 7    | AGC-DRR   |[Attributed Graph Clustering with Dual Redundancy Reduction](https://www.ijcai.org/proceedings/2022/0418.pdf)|[code](https://github.com/gongleii/AGC-DRR)|
| 8    | DGCluster |[DGCLUSTER: A Neural Framework for Attributed Graph Clustering via Modularity Maximization](https://ojs.aaai.org/index.php/AAAI/article/view/28983)|[code](https://github.com/pyrobits/DGCluster)|
| 9    | HSAN      |[Hard Sample Aware Network for Contrastive Deep Graph Clustering](https://arxiv.org/abs/2212.08665)|[code](https://github.com/yueliu1999/HSAN)|
| 10   | CCGC      |[Cluster-guided Contrastive Graph Clustering Network](https://arxiv.org/abs/2301.01098)|[code](https://github.com/xihongyang1999/CCGC)|
| 11   | MAGI      |[Revisiting Modularity Maximization for Graph Clustering: A Contrastive Learning Perspective](https://arxiv.org/abs/2406.142886)|[code](https://github.com/EdisonLeeeee/MAGI)|
| 12   | NS4GC     |[Reliable Node Similarity Matrix Guided Contrastive Graph Clustering](https://arxiv.org/pdf/2408.03765?)|[code](https://github.com/Cloudy1225/NS4GC)|

# Supported Datasets

| No.  | Dataset      | #Samples | #Features | #Edges | #Classes | Homo. Ratio |
| :----: | :------------: | --------: | ---------: | ------: | --------: | -----------: |
| 1    | Wiki         |2,405|4,973|17,981|17|0.71|
| 2    | Cora         |2,708|1,433|5,429|7|0.81|
| 3    | ACM          |3,025|1,870|13,128|3|0.82|
| 4    | Citeseer     |3,327|3,703|9,104|6|0.74|
| 5    | DBLP         |4,057|334|3,528|4|0.80|
| 6    | PubMed       |19,717|500|88,648|3|0.80|
| 7    | Ogbn-arXiv   |169,343|128|2,315,598|40|0.65|
| 8    | USPS(3NN)|9,298|256|27,894|10|0.98|
| 9    | HHAR(3NN)|10,299|561|30,897|6|0.95|
| 10   | BlogCatalog  |5,196|8,189|343,486|6|0.40|
| 11   | Flickr       |7,575|12,047|479,476|9|0.24|
| 12   | Roman-empire |22,662|300|65,854|18|0.05|

> More Datasets will be introduced.

# Citation



# Related Repositories

ADGC: [Awesome-Deep-Graph-Clustering](https://github.com/yueliu1999/Awesome-Deep-Graph-Clustering)

Older version of this repository: [A-Unified-Framework-for-Attribute-Graph-Clustering](https://github.com/Marigoldwu/PyDGC/releases/tag/v0.0.1)