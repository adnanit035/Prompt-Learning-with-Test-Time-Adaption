# Prompt-Learning-with-Test-Time-Adaption
Project for the 'Trends and Application of Computer Vision' course of the Master's Degree in Artificial Intelligence Systems at the University of Trento, A.Y. 2023/2024.

## Authors
- Adnan Irshad
- Hira Afzal

## Problem Statement
Recent foundation vision models like CLIP encode a wide range of visual concepts after training on millions of noisy image-text pairs and can be applied to downstream tasks in a zero-shot manner without task-specific training data. This is achieved by using prompts, which are natural language descriptions of the task, to guide the model to produce the desired output. For example, to perform zero-shot classification, we can use a prompt like "a photo of a dog" to guide the model to produce the desired output. However, in applying these models to a downstream task, heavily relies on the quality of the prompts. And prompts can be designed in different ways, like prompt engineering and hand-crafted prompts. But with the increasing number of tasks, these techniques becomes infeasible and requires lot of domain expertise. Therefore, we need to learn prompts for a given task by automating the process of prompt engineering. To address this problem, we explored different techniques to learn prompts for a given task.

## Description
This project divided into two main parts: 
1. **Prompt Learning**: In this part, we explored the different methods e.g. CoOp, CoCoOp to learn prompts for a given task using a small dataset. The details of the methods can be found in another [repository](https://github.com/MisterMandarino/Learning-Prompts-for-Transfer-Learning) by our other team members.
2. **Test-Time Adaptation**: In this part, we explored the different methods like TPT, AlignedPrompts to adapt the learned prompts at test-time to improve the performance of the model.

## Test-Time Adaptation
This repository contains the code for the test-time adaptation part of the project. CoOp applies prompt tuning to CLIP by tuning the prompt on a collection of training data which improves the CLIP’s performance on the downstream tasks but causes lacks in generalization to out-of-distribution data which then CoCoOp proposes to alleviate the problem by making the prompt conditioned on model inputs. Despite being effective on the given task, this line of work requires access to downstream training data with annotations, restricting the zero-shot knowledge transfer of foundation models. To address this problem, we explored different methods to adapt the learned prompts at test-time to improve the performance of the model.

Test-Time Adaptation (TTA) doesn't require any training data and can be applied to any model to improve ZSL performance on given task by learning prompts using target data at test-time.

<br>
<img align="center" src="images/TTA.png">
<p align="center">Figure: Test-Time Adaptation</p>






