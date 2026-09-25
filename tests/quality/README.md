# Vendored eval data for tests/quality_eval.py

These sets are fixed. Do not edit, reorder or re-sample them. A baseline JSON
is only comparable to a run over the same rows.

## nll_passages.jsonl (public domain)

40 passages of 510-512 tokens (DeepSeek-V4.1 tokenizer, no BOS). 4 per book
from 10 public-domain books. The text was downloaded from Project Gutenberg
(https://www.gutenberg.org). Each passage starts at the first paragraph of at
least 200 chars at 20/40/60/80 % of the book body, then runs to the largest
word prefix that fits in 512 tokens. Whitespace was collapsed. The Project
Gutenberg header, footer and license were removed, and no Project Gutenberg
trademark is used. Each row names its source eBook number, title and author.

Books (eBook #): Pride and Prejudice (1342), Frankenstein (84), Moby Dick
(2701), The Adventures of Sherlock Holmes (1661), A Tale of Two Cities (98),
The Adventures of Tom Sawyer (74), Dracula (345), The Wealth of Nations
(3300), On the Origin of Species (2009), The Prince, tr. W. K. Marriott
(1232).

These books are almost certainly in the training data. Use only the relative
ΔNLL between serve configs. Absolute perplexity means nothing here.

## gsm8k_100.jsonl (MIT)

100 items from the GSM8K test split: rows 0, 13, 26, ... 1287 of
`grade_school_math/data/test.jsonl` from https://github.com/openai/grade-school-math.
`answer` is the integer after `####`, with commas removed.

## mmlu_228.jsonl (MIT)

The first 4 test rows of each of the 57 MMLU subjects, with subjects in
sorted order. Source: https://github.com/hendrycks/test (`data.tar`,
`data/test/*_test.csv`).

## tools30.json

Written for this repo. It has 22 tool-call items with varied JSON schemas
and 8 negatives where the right answer is no tool call.

## Licenses

### GSM8K

MIT License

Copyright (c) 2021 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

### MMLU

MIT License

Copyright (c) 2020 Dan Hendrycks

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

