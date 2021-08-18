# Adversary-about-RGBD-SOD

a benchmark about adversarial attack&defense on RGBD Saliency Object Detection

## Attack
ml,rosa<br>
二者的区别在于，ml构造的目标预测在迭代过程中是不变的，它优化目标预测和实际预测的距离,在每一步获取梯度;rosa在每一步用(1-实际概率)对应的梯度-(实际概率)对应梯度。<br>
他们都考虑了像素限制，即已经成功修改预测的像素不参与下一步迭代。

## Defense
rosa,...

## :peach:Attack
最近3年，关注点有 **减少噪声冗余（关注图像特定区域）和新噪声（更强而仍然不可感知）**<br>
特别的，对于黑盒，还有 **减少查询次数、提高迁移能力** ;对于白盒，还有 **利用中间层加强攻击**<br>

### 2021

|TITLE|PUBLISHER|GENERATION| |
|---|---|---|---|
|**Adversarial Attack Against Deep Saliency Models Powered by Non-Redundant Priors**|TIP|punish feature activations corresponding to the salient regions;directional gradient estimation|partially white-box & black-box|
|Enhancing the transferability of adversarial attacks through variance tuning<br>[https://github.com/JHL-HUST/VT](https://github.com/JHL-HUST/VT)|CVPR|consider the gradient variance of the previous iteration to tune the current gradient to enhance iterative gradient based attack and improve their attack transferability|black-box|
| | | | |


### 2020

|TITLE|PUBLISHER|GENERATION| |
|---|---|---|---|
|Towards Large yet Imperceptible Adversarial Image Perturbations with Perceptual Color Distance<br />[https://github.com/ZhengyuZhao/PerC-Adversarial](https://github.com/ZhengyuZhao/PerC-Adversarial)|CVPR|PerC-C&W、PerC-AL: replace original penalty with new distance metric|white-box|
|ColorFool: Semantic Adversarial Colorization<br />[https://github.com/smartcameras/ColorFool](https://github.com/smartcameras/ColorFool)|CVPR|modify colors as unrestricted perturbations|black-box|
|Polishing Decision-Based Adversarial Noise With a Customized Sampling|CVPR|CAB: boundary attack with customized sampling|black-box|
|Physically Realizable Adversarial Examples for LiDAR Object Detection|CVPR|3D| |
|**Indirect Local Attacks for Context-Aware Semantic Segmentation Networks**|ECCV|adaptive local attacks using structured sparsity in loss|white-box|
|**ROSA: Robust Salient Object Detection Against Adversarial Attacks<br>[https://github.com/lhaof/ROSA-Robust-Salient-Object-Detection-Against-Adversarial-Attacks](https://github.com/lhaof/ROSA-Robust-Salient-Object-Detection-Against-Adversarial-Attacks)** |**IEEE Transactions on Cybernetics** |**firstly mount successful attack on SOD (refer to DAG)** |**white-box** |



### 2019

|TITLE|PUBLISHER|GENERATION| |
|---|---|---|---|
|Sparse and imperceivable adversarial attacks<br>[https://github.com/fra31/sparse-imperceivable-attacks](https://github.com/fra31/sparse-imperceivable-attacks)|ICCV|restrict region of which pixels are changed and minimize l0-distance to clean img|black-box|
|Subspace attack: Exploiting promising subspaces for query-efficient black-box attacks<br />[https://github.com/ZiangYan/subspace-attack.pytorch](https://github.com/ZiangYan/subspace-attack.pytorch)|NIPS|exploit gradient of a few reference models to reduce the query<br />complexity|black-box|
|Prior convictions: Black-box adversarial attacks with bandits and priors<br>[https://git.io/blackbox-bandits](https://git.io/blackbox-bandits)|ICLR|a bandit optimization-based algorithm using any prior to improve query|black-box|
|**MLAttack: Fooling Semantic Segmentation Networks by Multi-layer Attacks** |**German Conference on Pattern Recognition** |**gradient combination to addtionally match inter. layer response of source and  target img** |**white-box** |



### 2018

|TITLE|PUBLISHER|GENERATION| |
|---|---|---|---|
|Art of Singular Vectors and Universal Adversarial Perturbations|CVPR|compute vectors of Jacobian matrices of hidden layers|white-box|
|Boosting Adversarial Attacks with Momentum|CVPR|more transferable adversarial examples|black-box|
|Decision-based adversarial attacks: Reliable attacks against black-box machine learning models<br />[https://github.com/bethgelab/foolbox](https://github.com/bethgelab/foolbox)|ICLR|Boundary Attack:follow decision boundary,start with big perturbation and reduce it|black-box|
|Towards imperceptible and robust adversarial example attacks against neural networks|AAAI|new distance metric for higher imperceptibility and maximize prob gap for higher attack robustness in physical  world|white-box|



### 2017

|TITLE|PUBLISHER|GENERATION| |
|---|---|---|---|
|Universal Adversarial Perturbations<br />[https://github.com/LTS4/universal](https://github.com/LTS4/universal)|CVPR|iteratly aggregate min perturbation that send data to decision boundary|black-box|
|**Adversarial Examples for Semantic Segmentation and Object Detection** |**ICCV** |**DAG**  **using gradient** |**white-box** |
|Universal Adversarial Perturbations Against Semantic Image Segmentation|ICCV|customized loss gradient averaged over the entire training data|white-box|
|Delving into transferable adversarial examples and black-box attacks|ICLR|attack ensemble of models for better transferability|black-box|



## :peach:Defense
最近3年，通过**新损失函数、新网络块、主动引入新噪声、对抗训练和检测**提高模型鲁棒性<br>
对于对抗训练，关注**减少训练代价、提高泛化能力**，还有**添加分支以pixel-wise地利用对抗样本**<br>
特别的，提高单步训练鲁棒性，从**解决过拟合**和**改进对抗样本**两方面进行<br>

### 2021

|TITLE|PUBLISHER|FOCUS|
|---|---|---|
|**Beating Attackers At Their Own Games:Adversarial Example<br />Detection Using Adversarial Gradient Directions**|AAAI|train classifier exploit AGD and neighbor prototype for detection|
|**Improving Adversarial Robustness via Probabilistically Compact Loss with Logit Constraints**<br />[https://github.com/xinli0928/PC-LC](https://github.com/xinli0928/PC-LC)|AAAI|enlarge the probability gaps between true class and false classes and  prevent the gaps from being melted by a small perturbation|
|**Single-Step Adversarial Training for Semantic Segmentation** |**preprint** |**improve single-step Adv.Train by choosing an appropriate step size** |
|**Improving Adversarial Robustness via Channel-wise Activation Suppressing**<br />[https://github.com/bymavis/CAS_ICLR2021](https://github.com/bymavis/CAS_ICLR2021)|ICLR|add module to suppress redundant adversarial activation for Adv.Train|



### 2020

|TITLE|PUBLISHER|FOCUS|
|---|---|---|
|One Man's Trash Is Another Man's Treasure: Resisting Adversarial Examples by Adversarial Examples|CVPR|finding an adversarial example on a pretrained external model for Adv.Train|
|Achieving Robustness in the Wild via Adversarial Mixing With Disentangled Representations|CVPR          |real-world perturbation used for Adv.Train  |
|Single-Step Adversarial Training With Dropout Scheduling|CVPR|add dropout to single-step Adv.Train|
|Adversarial Vertex Mixup: Toward Better Adversarially Robust Generalization|CVPR|train&test set accuracy gap for Adv.Train|
|**On the Robustness of Semantic Segmentation Models to Adversarial Attacks** |**TPAMI** |**first rigorous evaluation**  |
|**Dynamic Divide-and-Conquer Adversarial Training for Robust Semantic Segmentation**<br>[https://github.com/dvlab-research/Robust-Semantic-Segmentation](https://github.com/dvlab-research/Robust-Semantic-Segmentation)|preprint|add branches to deal with pixels with diverse properties towards adv. perturbation for Adv. Train|
|**Indirect Local Attacks for Context-Aware Semantic Segmentation Networks**|ECCV|train model  using Mahalanobis distance and inter. feature to  detect|
|**ROSA: Robust Salient Object Detection Against Adversarial Attacks<br>[https://github.com/lhaof/ROSA-Robust-Salient-Object-Detection-Against-Adversarial-Attacks](https://github.com/lhaof/ROSA-Robust-Salient-Object-Detection-Against-Adversarial-Attacks)** |**IEEE Transactions on Cybernetics** |**introduce new generic noise to destroy adv. perturbations, learn to predict with introduced noise** |



### 2019

|TITLE|PUBLISHER|FOCUS|
|---|---|---|
|**Feature Denoising for Improving Adversarial Robustness**<br />[https://github.com/facebookresearch/ImageNet-Adversarial-Training](https://github.com/facebookresearch/ImageNet-Adversarial-Training)|CVPR|translate denoise operations to new network block|
|**Adversarial Defense by Restricting the Hidden Space of Deep Neural Networks**|ICCV|customize loss to learn distinct and distant decision regions for each class|



### 2018

|TITLE|PUBLISHER|FOCUS|
|---|---|---|
|Defense Against Universal Adversarial Perturbations|CVPR|add layer to rectify perturbation|
|Ensemble Adversarial Training: Attacks and Defenses|ICLR|augment training data with perturbation transferred from other models for Adv.Train|
|**Towards deep learning models resistant to adversarial attacks** |**ICLR** |**broad and unifying view based robust optimization about robust model** |
|Provable defenses against adversarial examples via the convex outer adversarial polytope<br />[http://github.com/locuslab/convex_adversarial](http://github.com/locuslab/convex_adversarial)|ICML|approximation of the set of activations reachable throuth perturbation to Detection|



### 2017

|TITLE|PUBLISHER|FOCUS|
|---|---|---|
|Adversary Resistant Deep Neural Networks with an Application to Malware Detection|SIGKDD|add a layer to mask pixels randomly |
|On detecting adversarial perturbations|ICLR|add a subnetwork for Detection|
|Adversarial machine learning at scale|ICLR|large dataset for Adv.Train|
