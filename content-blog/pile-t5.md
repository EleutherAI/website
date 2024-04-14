---
title: "Pile-T5"
date: 2024-04-15T00:00:00+07:00
description: "Trained T5 on the Pile"
author: ["Lintang Sutawika", "Aran Komatsuzaki", "Colin Raffel"]
draft: false
---

The T5 model (Raffel et al, 2019) has been a widely used model in the NLP community. With downloads of its base model from HF being in the millions, it's no doubt that these models have been a community favorite. In this blogpost, we introduce an alternative version of T5 that we hope would be useful for the community.

## Model Description

Our alternative version consists of replacing the pretrained dataset with the Pile and switching out the original T5 tokenizer with the LLAMA tokenizer. Pile-T5 was trained to 2 million steps or 2 trillion tokens in total. We train with the original span corruption method and observe that improvements for finetuning on downstream tasks that users would want to for their usecases. On top of that, Pile-T5 performs much better on code tasks which would benefit extension towards code tasks. Our released models were trained on the same hyperparameters as the original T5, utilizing [T5x](https://github.com/google-research/t5x). We release the our experiments scripts [here](https://github.com/EleutherAI/improved-t5).

These models are accessible from EleutherAI's [huggingface page](https://huggingface.co/collections/EleutherAI/pile-t5-65a76a0d0022dd270b385a66). A notable difference from the original T5 is that the Pile-T5 uses the transformer implementation for [UMT5](https://huggingface.co/docs/transformers/model_doc/umt5) (Chung, Constant, Garcia et al, 2023) because it uses the same scalable implementation in T5x. Inspired by Pythia (Biderman and Shoelkopf et al 2023), we release [intermediate checkpoints](https://huggingface.co/collections/EleutherAI/pile-t5-65a76a0d0022dd270b385a66) that span every 10.000 steps. The `main` branch for these models in their resepective huggingface page is the 2 million step version. In addition, we release the T5x versions of the checkpoints [here](https://huggingface.co/collections/EleutherAI/pile-t5-t5x-checkpoints-660aaab3e8c24412c5f69a6a).

## Going Beyond 1 Trillion Tokens

The Pile-T5 models were evaluated on SuperGLUE, CodeXGLUE, as well as MMLU and Bigbench Hard. The Pile-T5 models are compared with the T5v1.1 where both were finetuned over the same amount of tokens. We also compare Pile-T5 models againts the Flan-T5 models for MMLU and BBH as a loose comparison. All evaluations were done with the [LM-Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) to report model performance over the benchmarks we present here. We release the finetuned checkpoints for [Base](https://huggingface.co/collections/lintang/pile-t5-base-finetuned-models-65d307353ecda975d828ebb8), [Large](https://huggingface.co/collections/lintang/pile-t5-large-finetuned-models-65d307ec0545ab7c1153b804), [XL](https://huggingface.co/collections/lintang/pile-t5-xl-finetuned-models-65d30bbcf5a15aa421099b4e), and [XXL](https://huggingface.co/collections/lintang/pile-t5-xxl-finetuned-models-65a76bbc4908b2676c6b8a94).

### Performance on SuperGLUE

To asses performance on SuperGLUE, we finetune the Pile-T5 (Both the 1 trillion tokens version and the final 2 trillion tokens version) and T5v1.1 models for a batch size of 128 for 263k steps matching the original T5 paper. With all models except for Large, we observe substansial performance increase. 

|  Size |    Variant   |  Average  |   boolq   |     cb    |           |  copa  |  multirc  |           |   record  |           |    rte    |    wic    |    wsc    |
|:-----:|:------------:|:---------:|:---------:|:---------:|:---------:|:------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
|       |              |           |    acc    |     f1    |    acc    |   acc  |     f1    |     em    |     em    |     f1    |    acc    |    acc    |    acc    |
|  Base |   T5-v1.1    |   71.33   |   79.36   |   83.63   |    87.5   |   63   |   73.45   |   33.26   |    69.7   |   68.75   |   78.34   |   65.83   |   75.96   |
|       | Pile-T5 (1T) |   74.85   |   81.46   |   93.69   |   94.64   |   65   |   77.75   |    40.5   |   76.97   |   76.49   |   80.86   |   67.39   |   74.03   |
|       |  **Pile-T5** | **76.13** | **82.45** | **96.07** | **94.64** | **72** | **77.74** | **39.56** | **77.64** | **76.88** | **83.03** | **67.24** | **73.08** |
| Large | **T5-v1.1** | **81.11** | **85.96** | **93.21** | **96.43** | **82** | **81.71** | **48.37** | **82.18** | **81.71** | **85.92** | **71.47** | **81.73** |
|       | Pile-T5 (1T) |   79.18   |    83.7   |   91.85   |   94.64   |   79   |   82.36   |   47.85   |   82.72   |   82.14   |   83.03   |    65.2   |   81.73   |
|       |    Pile-T5   |   79.67   |   85.71   |   88.96   |   94.64   |   74   |    82.6   |   50.47   |    84.1   |    83.7   |   85.19   |   68.49   |   81.73   |
|   XL  |   T5-v1.1    |   81.76   |   86.79   |   81.18   |   91.07   |   84   |   84.03   |   52.89   |   83.92   |    83.5   |   90.25   |   73.04   |   81.73   |
|       | Pile-T5 (1T) |   86.09   |   89.76   |    90.6   |   94.64   |   96   |   88.17   |    63.9   |   91.58   |   91.36   |    93.5   |   72.73   |   86.54   |
|       |  **Pile-T5** | **89.00** |  **90.4** |  **93.1** | **96.43** | **96** | **88.63** | **65.16** | **92.21** | **91.96** | **92.78** | **75.24** | **96.15** |
|  XXL  |   T5-v1.1    |   82.43   |   88.29   |   93.61   |   94.64   |   86   |   75.22   |     51    |   84.67   |   84.55   |   89.17   |   72.41   |   81.73   |
|       | Pile-T5 (1T) |   87.11   |   90.46   |    94.3   |   96.43   |   93   |   80.81   |   56.77   |   91.36   |   91.18   |   92.42   |   70.38   |   95.19   |
|       |  **Pile-T5** | **90.08** | **90.98** | **98.68** | **98.21** | **95** | **89.28** | **67.68** | **93.04** |  **92.7** |  **93.5** | **75.24** | **96.15** |

### Performance on CodeXGlUE

We evaluated on the Code-to-Text subtask of CodeXGLUE (Su et al, 2021). Both Pile-T5 and T5v1.1 were finetune on each programming language variant for 10 epochs with the same method as detailed in the [original repo](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Text/code-to-text).

|  Size |    Version   |  Average  |   Python  |    PHP    |     Go    |    Java   | JavaScript |    Ruby   |
|:-----:|:------------:|:---------:|:---------:|:---------:|:---------:|:---------:|:----------:|:---------:|
|  Base |   T5-v1.1    |   14.34   |   15.55   |   21.72   |   14.71   |   14.89   |    9.25    |    9.9    |
|       | Pile-T5 (1T) |   15.90   |    17.2   |    22.9   |   16.75   |   16.24   |    11.23   |    11.1   |
|       |  **Pile-T5** | **16.37** | **17.78** | **23.12** |  **16.7** | **16.68** |  **11.89** | **12.06** |
| Large |   T5-v1.1    |   11.53   |   12.18   |   14.17   |   12.37   |    12.3   |    8.85    |    9.32   |
|       | Pile-T5 (1T) |   15.74   |   17.09   |    22.8   |   17.16   |   16.33   |    10.75   |   10.31   |
|       |  **Pile-T5** | **16.28** | **17.72** | **22.95** | **17.07** | **16.41** |  **12.05** | **11.45** |
|   XL  |   T5-v1.1    |   16.17   |   17.36   |   21.91   |   16.69   |   17.74   |    11.08   |   12.25   |
|       | Pile-T5 (1T) |   18.01   |   18.61   |   23.75   |   19.04   |   18.43   |    14.27   |   13.93   |
|       |  **Pile-T5** | **18.68** | **19.25** | **24.37** | **19.42** | **19.15** |  **15.1**  | **14.81** |
|  XXL  |   T5-v1.1    |   17.67   |   17.89   |   23.21   |   18.54   |   19.17   |    13.85   |   13.33   |
|       | Pile-T5 (1T) |   18.55   |   19.53   |   24.11   |   19.27   |   18.52   |    15.11   |   14.75   |
|       |  **Pile-T5** | **18.72** | **19.27** | **24.49** |  **19.6** | **18.96** |  **15.1**  | **14.92** |

Due to both the Pile inlcuding code-based data and the LLAMA tokenizer including characters frequently used in code, we observe improvement when on code-based benchmark, specifically the CodeXGlue Code-to-Text benchmark that comprises of 6 programming languages.

## Using Flan Instruction Tuning


We continue by finetuning Pile-T5 models on Flan (Chung, Hou, Longpre et all, 2022) with same training hyperparameters and evaluate on MMLU (Hendrycks et al, 2021) and BigBench Hard (Suzgun et al, 2022). We specifically use the 2 trillion tokens versions of Pile-T5. For fair comparison, We also finetune T5-v1.1 checkpoints with the same procedure. While comparison with FLAN models isn't necesarrily a fair comparison given that it was based on LM-Adapted version of T5v1.1, we include the perfromance score for reference. 

### Performance on Held-In

We observe competitive performance over held-in tasks (tasks that were included in the Flan Instruction Tuning dataset) with a dip in performance at the Large variant similar to SuperGLUE.

| Size  | Version  | Average | ANLI R1 | ANLI R2 | ANLI R3 | Arc Easy | Arc Challange | BoolQ  | RTE    |
| :---: | :------: | :-----: | :-----: | :-----: | :-----: | :------: | :-----------: | :----: | :----: |
| Base  | T5-v1\.1 | 46\.50  | 39\.9   | 34\.96  | 37\.33  | 38\.12   | 28\.23        | 70\.26 | 76\.73 |
|       | Pile-T5  | 46\.37  | 39\.32  | 35\.28  | 37\.53  | 36\.61   | 30\.67        | 71\.87 | 73\.28 |
| Large | T5-v1\.1 | 54\.90  | 52\.46  | 39\.67  | 42\.53  | 50\.6    | 39\.99        | 78\.56 | 80\.5  |
|       | Pile-T5  | 36\.97  | 33      | 33\.03  | 32\.98  | 29\.27   | 21\.21        | 56\.36 | 52\.95 |
| XL    | T5-v1\.1 | 56\.40  | 53\.82  | 40\.22  | 41\.01  | 56\.31   | 39\.08        | 80\.66 | 83\.71 |
|       | Pile-T5  | 64\.41  | 64\.36  | 48\.02  | 49\.18  | 66\.56   | 58\.28        | 85\.16 | 79\.3  |
| XXL   | T5-v1\.1 | 69\.99  | 71\.63  | 55\.81  | 57\.41  | 75\.56   | 62\.30        | 86\.53 | 80\.71 |
|       | Pile-T5  | 69\.21  | 71\.16  | 55\.92  | 55\.19  | 70\.85   | 59\.82        | 87\.55 | 83\.96 |



### Performance on MMLU

Models are evaluated on 2 prompts versions; the original prompt (Hendrycks et al, 2021) and (Chung, Hou, Longpre et all, 2022). In addition, evaluation is not only done in a greedy generation format but also by taking the highest loglikelihood of the available answer choices.

MMLU Prompt
```
The following are multiple choice questions (with answers) about abstract algebra.

Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.
A. 0
B. 4
C. 2
D. 6
Answer:
```

Flan Prompt

```
The following are multiple choice questions (with answers) about abstract algebra.

Q: Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.
(A) 0 (B) 4 (C) 2 (D) 6
A:
```

We see performance gains when using Pile-T5. For MMLU both likelihood-based and generative-based versions are evaluated on. With loglikelihood benifiting mostly zero-shot prompting. Using loglikelihood tends to help model when prompted with 0-shot while the models can struggle to output a proper response when using 0-shot greedy generation. This is due to using strict evaluation and the model frequently generating outputs such as the full answer instead of only the letters. Performance on greedy generation improves when using 5-shot prompting that provides the models example of the correct way to answer. It should be noted that performance can vary significantly depending on the prompt format. Averaging across all variations show that Pile-T5 improves upon v1.1 and is competitive against Flan T5 variants.

| Size | Variant  | Average | Highest Loglikelihood |        |             |        | Greedy Generation |        |             |        |
| :--: | :------: | :-----: | :-------------------: | :----: | :---------: | :----: | :---------------: | :----: | :---------: | :----: |
|      |          |         | Original Prompt       |        | Flan Prompt |        | Original Prompt   |        | Flan Prompt |        |
|      |          |         | 0-Shot                | 5-Shot | 0-Shot      | 5-Shot | 0-Shot            | 5-Shot | 0-Shot      | 5-Shot |
| XL   | Flan-T5  | 42\.45  | 47\.37                | 49\.17 | 47\.83      | 49\.43 | 6\.63             | 48\.8  | 40\.98      | 49\.39 |
|      | T5-v1\.1 | 36\.58  | 38\.59                | 39\.52 | 40\.64      | 39\.79 | 25\.95            | 38\.84 | 29\.66      | 39\.67 |
|      | Pile-T5  | 40\.82  | 46\.04                | 48\.71 | 47\.13      | 48\.18 | 3\.61             | 48\.58 | 35\.53      | 48\.77 |
| XXL  | Flan-T5  | 46\.94  | 51\.47                | 54\.28 | 53\.31      | 53\.85 | 2\.69             | 53\.93 | 52\.15      | 53\.85 |
|      | T5-v1\.1 | 45\.76  | 51\.03                | 51\.15 | 46\.72      | 50\.77 | 31\.00            | 50\.72 | 33\.90      | 50\.78 |
|      | Pile-T5  | 48\.27  | 50\.88                | 53\.35 | 52\.22      | 53\.06 | 35\.8             | 53\.13 | 33\.85      | 53\.84 |

### Performance on BigBench Hard (BBH)

Pile-T5 performs substantially better compared to T5v1.1 on BBH on both Few-shot and Zero-shot settings and comparatively well even against Flan-T5.

| Size | Variant  | Average | Greedy Generation |          |
| :--: | :------: | :-----: | :---------------: | :------: |
|      |          |         | Zero-Shot         | Few-Shot |
| XL   | Flan-T5  | 32\.54  | 24\.71            | 40\.36   |
|      | T5-v1\.1 | 30\.87  | 28\.67            | 33\.06   |
|      | Pile-T5  | 35\.74  | 29\.98            | 41\.49   |
| XXL  | Flan-T5  | 43\.89  | 43\.06            | 44\.72   |
|      | T5-v1\.1 | 37\.49  | 35\.14            | 39\.84   |
|      | Pile-T5  | 44\.16  | 41\.61            | 46\.71   |




## Conclusion

We observe improvements on finetuned benchmarks such as SuperGLUE, CodeXGLUE, MMLU and BBH. Althought Pile-T5 when finetuned on the Flan mixture lags behind Flan-T5, it still performs better compared to T5v1.1. We conclude that Pile-T5 would be a better model for future multitask finetuning and other tasks that benefit from the encoder-decoder architecture. With performance on the Pile-T5 Large being unexpectedly lagging in benchmarks such as SuperGLUE and Flan Held-In tasks, we believe that there may have been a bug and advise users to take caution when using it. In addition, we believe that the intermediate checkpoint release would wide benefit the community for research areas such as interpretability.


## Citation

```
@misc{2024PileT5,
  author  = {Lintang Sutawika and Aran Komatsuzaki and Colin Raffel},
  title   = {Pile-T5},
  year    = {2024},
  url     = {https://blog.eleuther.ai/pile-t5/}
  note    = {Blog post},
}
```


## References

1. Biderman, Stella, Hailey Schoelkopf, Quentin Anthony, Herbie Bradley, Kyle O’Brien, Eric Hallahan, Mohammad Aflah Khan, et al. *Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling*. arXiv [Cs.CL], 2023. arXiv. http://arxiv.org/abs/2304.01373.
2. Chung, Hyung Won, Noah Constant, Xavier Garcia, Adam Roberts, Yi Tay, Sharan Narang, and Orhan Firat. *UniMax: Fairer and More Effective Language Sampling for Large-Scale Multilingual Pretraining*. arXiv [Cs.CL], 2023. arXiv. http://arxiv.org/abs/2304.09151.
3. Chung, Hyung Won, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, et al. *Scaling Instruction-Finetuned Language Models*. arXiv [Cs.LG], 2022. arXiv. http://arxiv.org/abs/2210.11416.
4. Gao, Leo, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, et al. ‘A Framework for Few-Shot Language Model Evaluation’. Zenodo, 12 2023. https://doi.org/10.5281/zenodo.10256836.
5. Hendrycks, Dan, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. ‘Measuring Massive Multitask Language Understanding’. arXiv [Cs.CY], 2021. arXiv. http://arxiv.org/abs/2009.03300.
6. Longpre, Shayne, Le Hou, Tu Vu, Albert Webson, Hyung Won Chung, Yi Tay, Denny Zhou, et al. *The Flan Collection: Designing Data and Methods for Effective Instruction Tuning*. arXiv [Cs.AI], 2023. arXiv. http://arxiv.org/abs/2301.13688.
7. Lu, Shuai, Daya Guo, Shuo Ren, Junjie Huang, Alexey Svyatkovskiy, Ambrosio Blanco, Colin Clement, et al. ‘CodeXGLUE: A Machine Learning Benchmark Dataset for Code Understanding and Generation’. arXiv [Cs.SE], 2021. arXiv. http://arxiv.org/abs/2102.04664.
8. Suzgun, Mirac, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, et al. ‘Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them’. arXiv [Cs.CL], 2022. arXiv. http://arxiv.org/abs/2210.09261.
