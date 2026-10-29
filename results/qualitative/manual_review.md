# Manual translation and responsible use review

These observations compare the saved outputs for the same first 500 frozen IITB test IDs. They describe individual examples, not population-level error rates. The full source, reference, raw output, and scored translation remain in each model's `results/predictions/*_test_500_greedy.jsonl` file.

| Concern | Test ID | Observation from actual outputs |
| --- | --- | --- |
| Negation and event meaning | `test_0000000` | The source says they **never believed** the sheriff's account of a death. Transformer and mT5 omit that meaning; Llama copies the English source. Gemma retains disbelief but changes the death event to a form resembling “began to die” and distorts “sheriff.” |
| Names and unsupported additions | `test_0000010` | Transformer introduces Pranab Mukherjee, who is absent from the source, and repeats to the token limit. mT5 drops McGill, Montreal, and Hult Prize. Llama changes McGill to McLean College. Gemma keeps McGill in Latin script but corrupts Montreal and the prize name. |
| Numbers | `test_0000025` | The source says the speaker voted for Obama **twice**. Transformer and mT5 omit the count. Llama uses an awkward mixed-language phrase for it. Gemma preserves “दो बार,” though its closing phrase is unnatural. |
| Gendered wording | `test_0000048` | The source describes a claim about a woman and an attempted complaint. All outputs lose some of the allegation's structure. Llama adds an English preface despite the instruction to return only the translation; Gemma inserts the German word “letzten.” This example is insufficient to infer a systematic gender bias. |
| Cultural wording | `test_0000100` | The source compares haunted houses with Halloween. Transformer and mT5 are repetitive or unrelated; Llama gives a broken explanation rather than a translation. Gemma mentions haunted houses but omits Halloween. The supplied reference is itself awkward, limiting conclusions from automatic overlap scores for this sentence. |
| Sensitive anatomy | `test_0000014` | The source states a medical fact about reproductive anatomy. Llama produces a refusal-like response instead of translating; mT5 repeats unrelated gender terms. Gemma preserves the core contrast but leaves “vagina” in English. Human review is essential for medical wording. |
| Legal consequences | `test_0000122` | The source gives a possible sentence of **10 years to life if convicted**. Transformer and mT5 lose the sentence; Llama copies the English. Gemma keeps the range but leaves “convicted” in English and inserts Korean text, so it cannot be used as a final legal translation. |

Across all 500 test outputs, the automated report records empty, truncated, and no-Devanagari counts separately from BLEU and chrF. A Devanagari check is only a coarse script diagnostic; mixed-language output can still pass it. The models also differ in pretraining, tokenization, precision, and adaptation. None of these observed errors alone establishes a demographic bias rate. For medical, legal, or other consequential use, a qualified Hindi reviewer should verify names, numbers, negation, terminology, and the complete source meaning.

## Practical safeguards

- Check that names, dates, amounts, numerical ranges, and negation in the output match the source before release; flag additions and omissions for review.
- Reject or route outputs that copy the source, refuse a legitimate translation request, mix unrelated scripts, or hit the token limit.
- Maintain domain-specific glossaries and require a qualified reviewer for medical, legal, political, and other sensitive material.
- Audit a larger, balanced set of examples across genders, dialects, and cultural contexts before making any claim about group-level bias.
- Keep the original source beside the translation and allow correction or escalation when meaning is uncertain.
