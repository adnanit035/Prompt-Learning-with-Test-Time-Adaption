# Prompt-Learning-with-Test-Time-Adaption
Project for the 'Trends and Application of Computer Vision' course of the Master's Degree in Artificial Intelligence Systems at the University of Trento, A.Y. 2023/2024.

## Authors
- Adnan Irshad
- Hira Afzal

## Problem Statement
This project is about **Learning Prompts for Transfer Learning with Test-Time Adaptation** in pre-trained vision-language models like CLIP. Recent foundation vision models like CLIP encode a wide range of visual concepts after training on millions of noisy image-text pairs and can be applied to downstream tasks in a zero-shot manner without task-specific training data. This is achieved by using prompts, which are natural language descriptions of the task, to guide the model to produce the desired output. For example, to perform zero-shot classification, we can use a prompt like "a photo of a dog" to guide the model to produce the desired output. However, in applying these models to a downstream task, heavily relies on the quality of the prompts. And prompts can be designed in different ways, like prompt engineering. But with the increasing number of tasks, prompt engineering becomes infeasible and requires domain expertise. Therefore, we need to learn prompts for a given task by automating the process of prompt engineering with learn-able vectors in prompts.



