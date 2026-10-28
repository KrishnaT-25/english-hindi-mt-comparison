# Scratch Transformer: observed test errors

These observations use saved predictions from `transformer_test_greedy.jsonl`. They are examples, not estimates of error prevalence. The full file and frozen IDs support independent review.

| Test row | Source detail | Observed output | Error |
| --- | --- | --- | --- |
| `test_0000000` | “They never believed he died the way the sheriff concluded.” | “और हमने उन्हें कहा, \"ऐ मेरे साथ ही अपने रब की।” | Negation and the sheriff's conclusion disappear; unrelated religious wording appears. |
| `test_0000010` | McGill University students in Montreal received the Hult Prize. | Starts with an unrelated statement about Pranab Mukherjee, then repeats “एक बारी के लिए” to the token limit. | Named entities and award omitted; unsupported named person added; repetitive truncation. |
| `test_0000025` | An Obama voter says they voted twice. | “राष्ट्रपति ने कहा कि मैं इस अवसर पर विचार करने के लिए,” | The name and number are lost. |
| `test_0000500` | Bamford received a 15-day prison term near Phuket. | Generates a repeated `pranabmukherjee.in` URL fragment. | Name, place, and number lost; irrelevant web address added. |

The model's recurring political and religious phrases show an unsupported content risk in these examples. This observation does not establish demographic bias. Such output is unsuitable for factual or sensitive translation without human review.
