# SAMBA
Pytorch code of Scenario-Adaptive Meta-learning for MmWave Beam Alignment

# Introduction

In millimeter wave communication systems, achieving high-quality data transmission demands efficient and rapid beam alignment. Conventional deep learning-based methods, although promising, often rely on the assumption that training and testing channels share identical distribution. This assumption may not hold in practical settings, potentially leading to significant performance degradation when the deployment environment changes.

To address this issue, we introduce SAMBA, a novel meta-learning-based approach for adaptive beam alignment without requiring Channel State Information (CSI). SAMBA enables swift adaptation to unknown scenarios using a minimal set of newly labeled data. Specifically, we adopt a probing beam-search strategy to obviate the need for CSI. Furthermore, we employ Model-Agnostic Meta-Learning (MAML) for parameter pre-training and fine-tuning to enhance our model's adaptability.  
Confronted with the challenge of numerous beam candidates in the narrow beam selection problem, which complicates the straightforward replication of MAML, we develop a novel training task generation strategy. 

In our experimental assessments, we subjected SAMBA to a wide range of challenging scenarios using ray-tracing simulations. These scenarios encompassed various frequency bands, distinct base station layouts, and outdoor-to-indoor transitions. Our results demonstrate that SAMBA consistently outperforms learning-based baseline models, showcasing its superior domain adaptation capabilities in dynamic and diverse channel settings.


# Model Training 

* TrainedCodebook_MLP_maml.ipynb
  * Model-agnostic Meta-Learning-based beam training. Train and save models.

* TrainedCodebook_MLP_joint.ipynb
  * Traditional method-based beam training. Train and save models. 


# Model Testing

* LSSP_SNR_test.ipynb
  * After saving the trained models (using TrainedCodebook_MLP_maml / TrainedCodebook_MLP_joint), test the performance of models. Performance matrics: Beam selection accuracy and UE-end SNR.
