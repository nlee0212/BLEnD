# BLEnD

This is the official repository of **BLEnD: A Benchmark for LLMs on Everyday Knowledge in Diverse Cultures and Languages** (Submitted to NeurIPS 2024 Datasets and Benchmarks Track).

## About
![BLEnD Construction & LLM Evaluation Framework](main_figure.png)

Large language models (LLMs) often lack culture-specific everyday knowledge, especially across diverse regions and non-English languages. Existing benchmarks for evaluating LLMs' cultural sensitivities are usually limited to a single language or online sources like Wikipedia, which may not reflect the daily habits, customs, and lifestyles of different regions. That is, information about the food people eat for their birthday celebrations, spices they typically use, musical instruments youngsters play, or the sports they practice in school is not always explicitly written online.
To address this issue, we introduce **BLEnD**, a hand-crafted benchmark designed to evaluate LLMs' everyday knowledge across diverse cultures and languages.
The benchmark comprises 52.6k question-answer pairs from 16 countries/regions, in 13 different languages, including low-resource ones such as Amharic, Assamese, Azerbaijani, Hausa, and Sundanese.
We evaluate LLMs in two formats: short-answer questions, and multiple-choice questions.
We show that LLMs perform better in cultures that are more present online, with a maximum 57.34% difference in GPT-4, the best-performing model, in the short-answer format.
Furthermore, we find that LLMs perform better in their local languages for mid-to-high-resource languages. Interestingly, for languages deemed to be low-resource, LLMs provide better answers in English.
