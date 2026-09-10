# Faithfulness Report

Judge model: claude-opus-4-8

### flan_t5
SUPPORTED: 5, PARTIALLY_SUPPORTED: 3, UNSUPPORTED: 1, N/A (abstained): 1
pass rate (SUPPORTED / graded): 5/9 (56%)

### claude
SUPPORTED: 9, PARTIALLY_SUPPORTED: 0, UNSUPPORTED: 0, N/A (abstained): 1
pass rate (SUPPORTED / graded): 9/9 (100%)

| provider | id | verdict | explanation |
| --- | --- | --- | --- |
| flan_t5 | eiffel_height | SUPPORTED | All claims in the answer are grounded in the context: the 330m/1,083ft height, the 41 years record, the top platform altitude of 276m/906ft, and the description as a wrought-iron lattice tower in Paris, France. Minor typos (906 ff) don't affect factual grounding. |
| flan_t5 | eiffel_visitors | SUPPORTED | All claims in the answer are grounded: 7 million visitors, the top level's altitude of 276m, and being the most-visited paid monument are all stated in the context. |
| flan_t5 | category_prediction | PARTIALLY_SUPPORTED | The claim that TF-IDF features were used for category prediction is grounded in the context. However, the answer incorrectly attaches the rating-prediction details (global average rating, learned user/book biases, clipping to 1–5 range) to category prediction—these statements describe Task 3, not Task 2. This misattributed linkage (bias-model predictions written to predictions_Category.csv) is not supported by the context, which states the bias model produced predictions_Rating.csv. |
| flan_t5 | read_prediction | SUPPORTED | Every claim in the answer—the popularity-based model, counting how often each book was read, sorting by popularity, choosing the group making up ~70% of reads, predicting 1 if in the popular group and 0 otherwise—matches the context. The output file reference 'predictions_Read.ccv' is a trivial typo of the grounded 'predictions_Read.csv' and not a substantive factual error. |
| flan_t5 | half_shell_helmets | SUPPORTED | Every claim in the answer is directly quoted from the context, including that partial coverage helmets are ejected more often than full-face helmets, that beanie-style helmets look like half-shell helmets but lack impact-absorbing liner, are not designed for motorcycle use, and provide no protection in a crash. |
| flan_t5 | roll_off_test | PARTIALLY_SUPPORTED | The answer states the roll-off test applies to all helmets and is more critical for three-quarter and half-shell helmets, which is grounded. However, it also adds a claim that it is 'more important for three quarter and quarter-shell,' introducing 'quarter-shell' which is not mentioned in the context. This ungrounded claim makes it partially supported. |
| flan_t5 | hirham_model | N/A | no citations returned (system abstained) |
| flan_t5 | rcao_model | PARTIALLY_SUPPORTED | Some claims in the answer are grounded in the context (50-km resolution, IPCC SRES A2 scenario, high emissions from low abatement priorities and high population growth). However, the answer also states the A2 scenario 'assumes a low level of emission,' which directly contradicts the context that says it assumes a high level of emissions. This makes it partially supported. |
| flan_t5 | yelp_liability | SUPPORTED | The answer is a direct quote (with minor typos) of text present in the context regarding the exclusion of special, indirect, incidental, or consequential damages. Every factual element of the answer is grounded in the provided context, so it is supported on grounding grounds regardless of whether it fully answers the question. |
| flan_t5 | yelp_term | UNSUPPORTED | The answer claims the term of the agreement is 'five (5) business days,' but the context's mention of five business days refers to Yelp's timeframe for approving public release, not the term/duration of the Data Agreement. This misapplication is not grounded in the context. |
| claude | eiffel_height | SUPPORTED | Both claims—330 metres (1,083 ft) tall and holding the record for 41 years—are directly stated in the context. |
| claude | eiffel_visitors | SUPPORTED | The answer states approximately 7 million people visit the Eiffel Tower each year, which is directly stated in the context. |
| claude | category_prediction | SUPPORTED | All claims match the context: text classification task, TF-IDF features from train_Category.json.gz, unigrams and bigrams, English stopwords removed, limits on rare/common words, and logistic regression to predict genre ID. |
| claude | read_prediction | SUPPORTED | The context explicitly states 'I built a simple popularity based model' for Task 1: Read Prediction, which directly matches the answer. |
| claude | half_shell_helmets | SUPPORTED | All claims in the answer are directly grounded in the context: half-shell helmets provide the least protection, the need to get one designed for motorcycling with DOT labeling and impact-absorbing liner, and research showing partial coverage helmets are ejected more often than full-face helmets. |
| claude | roll_off_test | SUPPORTED | Every claim in the answer is directly grounded in the context: the roll-off test checks for right fit, the steps (put helmet on, adjust chinstrap, reach hands behind head and rotate forward/down), the failure criterion (comes off or close to coming off), the recommendation to get a different size/model, and the note that it applies to all helmets but is more critical for three-quarter and half-shell helmets. |
| claude | hirham_model | N/A | no citations returned (system abstained) |
| claude | rcao_model | SUPPORTED | The answer 'I don't know' makes no factual claims, so there is nothing that could be ungrounded. The context only mentions the Swedish RCAO model without defining it, so declining to answer is not contradicted. |
| claude | yelp_liability | SUPPORTED | The context explicitly states Yelp's maximum liability shall not exceed US$50.00, which matches the answer exactly. |
| claude | yelp_term | SUPPORTED | The context states the agreement 'shall continue in full force and effect for a term of twelve (12) months from the Effective Date,' which directly supports the answer. |
