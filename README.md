# Growth Above All: supplementary material

This repository accompanies the paper "Growth Above All: Goals and Beneficiaries of LLM Silver Linings in Thai and US Settings" (MultiPsyche 2026). Files with Thai text are UTF-8.

- `stimuli.csv`: the 18 schemas in all four cells (US setting in English, US setting in Thai, Thai setting in Thai, Thai setting in English), with text lengths (English in words, Thai in characters without spaces) and the setting-cue nouns of each text.
- `identifiability.csv`: the setting-identifiability check on the final texts, per text and model (five answers each; a text passes when at least four answers name the intended country).
- `prompts.json`: the prompt template, the persona lines of both arms, the three instruction wording pairs, the identifiability question, and the generation and coding settings.
- `model_outputs.jsonl`: all 864 generated lists, one per line, with the parsed items, any preamble or closing text, and parse flags. List ids have the form schema-cell-arm-model-wording, where the cell codes are us_en, us_th, th_th, and th_en (setting then language), the arms are imp and exp, and the models are gpt and ds.
- `codebook.md`: the codebook given to the LLM coders and the human coders, followed by the computed coarse goal level.
- `codes/`: the codes of the two LLM coders, GPT-6.1 Sol and Claude Opus 5.5, for all 5,190 items. Item type "none" means not a lining; goal and beneficiary are filled for linings and consolation type for consolations.
- `known_answer/`: the 72 known-answer items in Thai and English with their intended codes (`type`, `cons`, `goal`, `ben`; `cultural` marks the 12 items with Thai cultural content), and each LLM coder's codes for them.
- `heldout/`: the 36 held-out lists.
- `scripts/`: the scripts that generated the lists (`gen.py`), ran the LLM coders (`llmcode_openai.py`, `llmcode_claude.py`, with `llmcode.py`, `llmcode4.py`, and `coderun.py`), scored the known-answer set (`score_known.py`), and produced the analyses and figures (`analyze.py`, `analyze_coders.py`, `agree.py`, `agree_heldout.py`, `lexical_check.py`, `make_figures_v4.py`). They read the files of the original project folder, so their paths differ from the layout of this repository. API keys are read from the environment and are not included.
